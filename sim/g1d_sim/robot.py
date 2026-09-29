"""G1DSim — le G1-D + Dex1-1 dans MuJoCo, réduit à ce dont un VLA a besoin.

Extrait de ``mpc_any`` (``skills/sim_robot.py`` + ``skills/context.py`` + ``config/motion.py``),
mêmes conventions physiques :

* servos ``<position>`` par joint (``scene_g1d.xml``), physique à 0,002 s ;
* feedforward de GRAVITÉ Pinocchio sur les deux bras à chaque pas de contrôle
  (sans lui les bras s'affaissent sous le servo position) ;
* cadence de CONTRÔLE décimée (30 Hz par défaut = le robot réel) : la consigne est TENUE
  pendant les sous-pas de physique — on ne touche jamais à ``m.opt.timestep`` ;
* colonne télescopique, lacet et buste tenus à 0 ; pose de départ = ``tuck`` ;
* offset ``mj_pin`` monde MuJoCo <-> monde Pinocchio (le robot est remonté pour poser
  ses roues au sol), calculé après le settle.

Tout le reste de mpc_any (collision, planif, perception, BT, grasping) est volontairement
laissé de côté.

Caméras, nommées comme les entrées de Hy-VLA :
``top_head`` = ``torso_rgbd``, ``hand_left`` = ``left_wrist_cam``, ``hand_right`` = ``right_wrist_cam``.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

from .camera import SimCamera
from .kinematics import ARM_JOINTS, ArmKinematics

ASSETS = Path(__file__).resolve().parents[1] / "assets"
SCENE_XML = ASSETS / "scene_g1d.xml"
URDF = ASSETS / "g1_d_dex1.urdf"

SIDES = ("left", "right")
CAMERAS = {"top_head": "torso_rgbd", "hand_left": "left_wrist_cam",
           "hand_right": "right_wrist_cam"}

#: colonne télescopique (2 étages) verrouillée en butée basse
LOCKED_JOINTS = {"LZ_mt_Joint": 0.0, "LZ_it_Joint": 0.0}
#: pose de repos, bras le long du corps (config.motion._tuck de mpc_any)
TUCK_Q = np.array([0.3, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0])
#: consigne mors Joint1_1 (m). OUVERT à -0.018 et pas -0.02 (butée : le servo s'y bat) ;
#: FERMÉ = 0.0245 = pleine fermeture, l'objet bloque les mors et l'erreur du servo serre.
GRIP_OPEN, GRIP_CLOSED = -0.018, 0.0245
SETTLE_STEPS = 2000                    # pas de physique au reset (4 s à 0,002 s)

#: orientation pince « par le dessus » (approche verticale) — config.motion.R_DOWN
R_DOWN = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
#: miroir articulaire bras droit -> gauche (config.motion.ARM_MIRROR_SIGNS)
ARM_MIRROR_SIGNS = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
#: posture de référence nullspace, bras droit (config.motion.natural_q)
NATURAL_Q_RIGHT = np.array([-0.7, -0.45, 0.0, 1.1, 0.0, 0.6, 0.0])
#: pose de TRAVAIL (départ des épisodes VLA) : TCP au-dessus de la table, pince vers le
#: bas. ⚠ Le tuck est quasi singulier (bras pendant) : un suivi cartésien en ``follow``
#: depuis le tuck décroche (mesuré : 6/44 sous-cibles convergées) — partir d'ici.
READY_TCP = {"left": np.array([0.35, 0.15, 0.90]), "right": np.array([0.35, -0.15, 0.90])}
#: hauteur du point de passage de ``go_ready`` : au-dessus des murs du carton (0.822)
VIA_Z = 0.95
#: borne de vitesse articulaire de ``track_tcp`` (rad/s)
MAX_JOINT_SPEED = 3.0


class G1DSim:
    def __init__(self, scene_xml: str | Path = SCENE_XML, urdf: str | Path = URDF, *,
                 control_hz: float = 30.0, image_size: tuple[int, int] = (640, 480),
                 depth: bool = False) -> None:
        self.m = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.d = mujoco.MjData(self.m)
        self.control_hz = float(control_hz)
        self.sous_pas = max(1, int(round(1.0 / self.control_hz / self.m.opt.timestep)))
        self._image_size = tuple(image_size)
        self._depth = depth
        self._cams: dict[str, SimCamera] = {}

        self.kin = {s: ArmKinematics(urdf, s, LOCKED_JOINTS) for s in SIDES}
        self._arm_qadr = {s: np.array([self.m.joint(n).qposadr[0] for n in self.kin[s].joint_names])
                          for s in SIDES}
        self._arm_dof = {s: np.array([self.m.joint(n).dofadr[0] for n in self.kin[s].joint_names])
                         for s in SIDES}
        self._arm_act = {s: np.array([self.m.actuator(n).id for n in self.kin[s].joint_names])
                         for s in SIDES}
        self._arm_range = {s: self.m.jnt_range[[self.m.joint(n).id for n in self.kin[s].joint_names]]
                           for s in SIDES}
        self._grip_act = {s: self.m.actuator(f"{s}_gripper_Joint1_1").id for s in SIDES}
        self._grip_qadr = {s: self.m.joint(f"{s}_gripper_Joint1_1").qposadr[0] for s in SIDES}
        self.mj_pin = np.zeros(3)
        self.reset()

    # ------------------------------------------------------------------ cycle de vie
    def reset(self) -> None:
        """Robot en tuck, pinces ouvertes, objets posés par la physique, offset recalé."""
        mujoco.mj_resetData(self.m, self.d)
        for name, q in LOCKED_JOINTS.items():
            self.d.ctrl[self.m.actuator(name).id] = q
        for s in SIDES:
            # le robot APPARAÎT déjà en tuck (la transition qpos0 -> tuck fauche la table)
            self.d.qpos[self._arm_qadr[s]] = TUCK_Q
            self.set_arm_target(s, TUCK_Q)
            self.set_gripper(s, 0.0)
        mujoco.mj_forward(self.m, self.d)
        for _ in range(SETTLE_STEPS):
            self._apply_gravity()
            mujoco.mj_step(self.m, self.d)
        self._refresh_mj_pin()

    def close(self) -> None:
        for cam in self._cams.values():
            cam.close()
        self._cams.clear()

    def _refresh_mj_pin(self) -> None:
        ak = self.kin["right"]
        pin.framesForwardKinematics(ak.model, ak.data, pin.neutral(ak.model))
        self.mj_pin = (self.d.body("torso_link").xpos
                       - ak.data.oMf[ak.model.getFrameId("torso_link")].translation).copy()

    # ------------------------------------------------------------------ physique
    def _apply_gravity(self) -> None:
        for s in SIDES:
            self.d.qfrc_applied[self._arm_dof[s]] = self.kin[s].gravity(self.arm_q(s))

    def step(self, n: int = 1) -> None:
        """``n`` pas de CONTRÔLE (chacun = ``sous_pas`` pas de physique à consigne tenue)."""
        for _ in range(n):
            self._apply_gravity()
            for _ in range(self.sous_pas):
                mujoco.mj_step(self.m, self.d)

    @property
    def time(self) -> float:
        return float(self.d.time)

    # ------------------------------------------------------------------ bras
    def arm_q(self, side: str) -> np.ndarray:
        """Angles MESURÉS [7] du bras (ordre ``ARM_JOINTS``)."""
        return self.d.qpos[self._arm_qadr[side]].copy()

    def set_arm_target(self, side: str, q) -> None:
        """Consigne de position [7], bornée aux butées du modèle."""
        lo, hi = self._arm_range[side].T
        self.d.ctrl[self._arm_act[side]] = np.clip(np.asarray(q, float), lo, hi)

    def tcp_pose(self, side: str, q=None) -> np.ndarray:
        """Pose 4×4 du TCP en repère MONDE MuJoCo (FK Pinocchio + ``mj_pin``)."""
        fk = self.kin[side].fk(self.arm_q(side) if q is None else q)
        T = np.eye(4)
        T[:3, :3] = fk.rotation
        T[:3, 3] = fk.translation + self.mj_pin
        return T

    def solve_ik(self, side: str, T_world: np.ndarray, q0=None, *,
                 follow: bool = True) -> tuple[np.ndarray, bool]:
        """IK vers une pose TCP 4×4 en repère MONDE. ``follow=True`` (défaut) = une seule
        descente depuis ``q0`` (ou la mesure) : continuité de branche le long d'un chunk."""
        T_world = np.asarray(T_world, float)
        target = pin.SE3(T_world[:3, :3], T_world[:3, 3] - self.mj_pin)
        return self.kin[side].ik(target, self.arm_q(side) if q0 is None else q0,
                                 follow=follow)

    def track_tcp(self, side: str, T_world: np.ndarray,
                  max_joint_speed: float = MAX_JOINT_SPEED) -> bool:
        """Consigne CARTÉSIENNE sûre pour un pas de contrôle — ce qu'un exécuteur de
        chunks (VLA) doit appeler. IK ``follow`` depuis la CONSIGNE courante (pas la mesure :
        la branche suivie reste celle qu'on commande) ; si elle ne converge pas, la consigne
        est TENUE et on rend False. Le pas articulaire est borné à ``max_joint_speed``.

        Pourquoi : hors d'atteinte, la DLS rend son meilleur effort — mesuré jusqu'à
        4,5 rad en un pas de 33 ms ; les servos kp=300 l'exécutent, le bras percute et la
        physique diverge (NaN). Une cible irréalisable ne doit jamais devenir un saut."""
        q_cmd = self.d.ctrl[self._arm_act[side]].copy()
        q, ok = self.solve_ik(side, T_world, q0=q_cmd, follow=True)
        if not ok:
            return False
        dq_max = max_joint_speed / self.control_hz
        self.set_arm_target(side, q_cmd + np.clip(q - q_cmd, -dq_max, dq_max))
        return True

    def natural_q(self, side: str) -> np.ndarray:
        return NATURAL_Q_RIGHT if side == "right" else NATURAL_Q_RIGHT * ARM_MIRROR_SIGNS

    def move_arms(self, targets: dict[str, np.ndarray], duration_s: float = 2.0) -> None:
        """Rampe ARTICULAIRE linéaire des bras donnés vers ``targets`` (+ 0,5 s de
        stabilisation). Pour les transitions entre poses, pas pour le suivi de chunks."""
        start = {s: self.arm_q(s) for s in targets}
        n = max(1, int(duration_s * self.control_hz))
        for a in np.linspace(0.0, 1.0, n + 1)[1:]:
            for s, q in targets.items():
                self.set_arm_target(s, (1 - a) * start[s] + a * np.asarray(q, float))
            self.step()
        self.step(int(0.5 * self.control_hz))

    def go_ready(self, poses: dict[str, np.ndarray] | None = None,
                 duration_s: float = 2.0) -> None:
        """Les deux bras en pose de travail, pinces ouvertes. ``poses`` = TCP 4×4 monde par
        bras ; défaut = ``READY_TCP`` pince vers le bas."""
        # Deux temps : MONTER sur place (la main du tuck est derrière le bord de table, sous
        # le plateau), PUIS avancer au-dessus du décor. Une rampe articulaire directe depuis
        # le tuck fait passer la pince droite À TRAVERS le carton (mesuré : bloquée dans
        # carton_mur2, TCP 12 cm à côté de sa cible) — le corridor que mpc_any signalait.
        vias, targets = {}, {}
        for s in SIDES:
            if poses is not None:
                T = np.asarray(poses[s], float)
            else:
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = R_DOWN, READY_TCP[s]
            q, ok = self.kin[s].ik(pin.SE3(T[:3, :3], T[:3, 3] - self.mj_pin), self.arm_q(s),
                                   posture=self.natural_q(s))
            if not ok:
                raise RuntimeError(f"{s}: pose de travail {np.round(T[:3, 3], 3)} hors d'atteinte")
            targets[s] = q
            p_via = self.tcp_pose(s)[:3, 3].copy()
            p_via[2] = max(T[2, 3], VIA_Z)
            q_via, ok = self.kin[s].ik(pin.SE3(T[:3, :3], p_via - self.mj_pin), q,
                                       posture=self.natural_q(s))
            if ok:
                vias[s] = q_via
            self.set_gripper(s, 0.0)
        if vias:
            self.move_arms(vias, duration_s / 2)
        self.move_arms(targets, duration_s)

    # ------------------------------------------------------------------ pinces
    def set_gripper(self, side: str, closure: float) -> None:
        """``closure`` ∈ [0, 1] : 0 = ouverte, 1 = consigne de fermeture complète."""
        c = float(np.clip(closure, 0.0, 1.0))
        self.d.ctrl[self._grip_act[side]] = GRIP_OPEN + c * (GRIP_CLOSED - GRIP_OPEN)

    def gripper(self, side: str) -> float:
        """Fermeture MESURÉE ∈ [0, 1] (bloquée à la largeur de l'objet s'il est pris)."""
        q = float(self.d.qpos[self._grip_qadr[side]])
        return float(np.clip((q - GRIP_OPEN) / (GRIP_CLOSED - GRIP_OPEN), 0.0, 1.0))

    # ------------------------------------------------------------------ caméras
    def camera(self, name: str) -> SimCamera:
        """``name`` = clé de ``CAMERAS`` (top_head, hand_left, hand_right) ou nom MJCF."""
        mj_name = CAMERAS.get(name, name)
        if mj_name not in self._cams:
            w, h = self._image_size
            self._cams[mj_name] = SimCamera(self.m, self.d, mj_name, w, h, depth=self._depth)
        return self._cams[mj_name]

    def render(self, name: str) -> np.ndarray:
        return self.camera(name).rgb()

    def render_all(self) -> dict[str, np.ndarray]:
        """Les trois vues du VLA, clés ``top_head`` / ``hand_left`` / ``hand_right``."""
        return {k: self.render(k) for k in CAMERAS}

    # ------------------------------------------------------------------ monde / état
    def object_pose(self, body: str = "piece") -> np.ndarray:
        """⚠ VÉRITÉ TERRAIN (sim seulement) — pour le scoring, jamais pour la politique."""
        T = np.eye(4)
        T[:3, :3] = self.d.body(body).xmat.reshape(3, 3)
        T[:3, 3] = self.d.body(body).xpos
        return T

    def save_state(self) -> np.ndarray:
        """Instantané complet (qpos, qvel, act, ctrl, forces appliquées…) — le
        « rollback » de FlowPRO se fait en rechargeant cet état."""
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        out = np.empty(mujoco.mj_stateSize(self.m, spec))
        mujoco.mj_getState(self.m, self.d, out, spec)
        return out

    def restore_state(self, state: np.ndarray) -> None:
        mujoco.mj_setState(self.m, self.d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.m, self.d)


__all__ = ["G1DSim", "CAMERAS", "SIDES", "ARM_JOINTS", "SCENE_XML", "URDF"]
