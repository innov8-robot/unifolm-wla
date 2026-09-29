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

Caméras, nommées comme les rôles image de UnifoLM-WLA (``configs/unitree.yaml``) :
``head_left`` = ``head_left_cam`` (œil gauche rectifié de la stéréo de tête),
``cam_wrist_left`` = ``left_wrist_cam``, ``cam_wrist_right`` = ``right_wrist_cam``.
RGB seulement : WLA n'utilise pas de profondeur.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

from .camera import SimCamera
from .kinematics import ARM_JOINTS, ArmKinematics

# Constantes du contrat iso WLA : source unique dans g1d_wla (racine du dépôt), partagée avec le
# convertisseur d'enregistrements. Voir docs/G1D_Constats.md §9.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from g1d_wla.frames import (G1_PELVIS_TO_TORSO_XYZ, G1_STANDING_LEGS,  # noqa: E402
                            TORSO_PITCH_TRAINING as TORSO_PITCH, WLA_EE_IN_WRIST)

ASSETS = Path(__file__).resolve().parents[1] / "assets"
SCENE_XML = ASSETS / "scene_g1d.xml"
URDF = ASSETS / "g1_d_dex1.urdf"

SIDES = ("left", "right")
CAMERAS = {"head_left": "head_left_cam", "cam_wrist_left": "left_wrist_cam",
           "cam_wrist_right": "right_wrist_cam"}
#: caméra MJCF servant le rôle ``head_left`` selon la vue voulue :
#: ``rec`` = œil gauche rectifié (config Dex1 du dépôt), ``raw`` = œil gauche brut (config WBT,
#: et seule vue des 32 datasets UniBot : ~3/4 des frames publiques avec pose effecteur).
HEAD_VIEWS = {"rec": "head_left_cam", "raw": "head_left_raw_cam"}

#: joint de tangage du buste. Il s'appelle ``Yaw_Joint`` dans la scène mais son axe est y.
TORSO_JOINT = "Yaw_Joint"
#: joints hors bras tenus fixes : colonne télescopique (2 étages) en butée basse, buste DROIT
#: au reset. Le buste ne s'incline qu'une fois les bras au-dessus de la table (``go_ready``) :
#: incliné en tuck, les poignets entrent dans la table et la physique diverge (mesuré).
LOCKED_JOINTS = {"LZ_mt_Joint": 0.0, "LZ_it_Joint": 0.0, TORSO_JOINT: 0.0}
#: pose de DÉPART des épisodes G1 Dex1 : angles médians des bras à la frame 0 (40 épisodes de
#: G1_Dex1_Stack_Block). Les bras G1 et G1-D étant identiques (au poignet près, 5 mm), ces angles
#: redonnent la pose effecteur de départ du G1 dans la base WLA (vérifié : ~5 mm, ~0.05 rad).
G1_START_Q = {"left": np.array([-0.235, 0.848, 0.25, 0.115, -0.55, 0.201, -1.32]),
              "right": np.array([-0.164, -0.972, -0.262, -0.371, 0.619, 0.625, 1.135])}
#: posture de PASSAGE tuck -> départ, bras droit (gauche = miroir) : bras écartés, coude fléchi,
#: avant-bras relevé au-dessus du bord de table. Trouvée par recherche de collisions le long des
#: deux rampes articulaires (scène actuelle, buste à TORSO_PITCH). La rampe directe fait taper
#: les poignets dans le bord de la table et dans le carton (mesuré).
VIA_Q_RIGHT = np.array([0.041, -1.3, -0.982, 1.667, 1.753, 0.556, 1.166])
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
#: pince « par le dessus » : ``R_DOWN`` reste disponible pour des poses IK sur mesure.
#: ⚠ Le tuck est quasi singulier (bras pendant) : un suivi cartésien en ``follow`` depuis le
#: tuck décroche (mesuré : 6/44 sous-cibles convergées) — toujours passer par ``go_ready``.
#: borne de vitesse articulaire de ``track_tcp`` (rad/s)
MAX_JOINT_SPEED = 3.0


class G1DSim:
    def __init__(self, scene_xml: str | Path = SCENE_XML, urdf: str | Path = URDF, *,
                 control_hz: float = 30.0, image_size: tuple[int, int] = (640, 480),
                 head_view: str = "rec") -> None:
        if head_view not in HEAD_VIEWS:
            raise ValueError(f"head_view='{head_view}' (attendu : {list(HEAD_VIEWS)})")
        self.cameras = dict(CAMERAS, head_left=HEAD_VIEWS[head_view])
        self.m = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.d = mujoco.MjData(self.m)
        self.control_hz = float(control_hz)
        self.sous_pas = max(1, int(round(1.0 / self.control_hz / self.m.opt.timestep)))
        self._image_size = tuple(image_size)
        self._cams: dict[str, SimCamera] = {}

        self._urdf = urdf
        self.kin = {s: ArmKinematics(urdf, s, LOCKED_JOINTS) for s in SIDES}
        self._arm_qadr = {s: np.array([self.m.joint(n).qposadr[0] for n in self.kin[s].joint_names])
                          for s in SIDES}
        self._arm_dof = {s: np.array([self.m.joint(n).dofadr[0] for n in self.kin[s].joint_names])
                         for s in SIDES}
        self._arm_act = {s: np.array([self.m.actuator(n).id for n in self.kin[s].joint_names])
                         for s in SIDES}
        self._arm_range = {s: self.m.jnt_range[[self.m.joint(n).id for n in self.kin[s].joint_names]]
                           for s in SIDES}
        self._torso_act = self.m.actuator(TORSO_JOINT).id
        self._torso_dof = self.m.joint(TORSO_JOINT).dofadr[0]
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
        self._refresh_kinematics()

    def close(self) -> None:
        for cam in self._cams.values():
            cam.close()
        self._cams.clear()

    def _refresh_kinematics(self) -> None:
        """Reconstruit la FK/IK avec les angles RÉELS des joints hors bras (colonne, buste),
        puis recale ``mj_pin``. À appeler après tout changement de ces joints."""
        measured = {n: float(self.d.qpos[self.m.joint(n).qposadr[0]]) for n in LOCKED_JOINTS}
        self.kin = {s: ArmKinematics(self._urdf, s, measured) for s in SIDES}
        self._refresh_mj_pin()

    def torso_pitch(self) -> float:
        """Tangage MESURÉ du buste (rad)."""
        return float(self.d.qpos[self.m.joint(TORSO_JOINT).qposadr[0]])

    def set_torso_pitch(self, pitch: float, duration_s: float = 1.0, hold_tcp: bool = True) -> None:
        """Rampe du tangage du buste, FK recalée à chaque pas (elle dépend du buste).
        ``hold_tcp=True`` : les deux TCP sont TENUS en repère monde (``track_tcp``) ;
        sinon les consignes articulaires des bras sont tenues et les mains suivent le buste."""
        if not hold_tcp:
            start = float(self.d.ctrl[self._torso_act])
            n = max(1, int(duration_s * self.control_hz))
            for a in np.linspace(0.0, 1.0, n + 1)[1:]:
                self.d.ctrl[self._torso_act] = (1 - a) * start + a * pitch
                self.step()
            self.step(int(0.5 * self.control_hz))
            self._refresh_kinematics()
            return
        hold = {s: self.tcp_pose(s) for s in SIDES}
        start = float(self.d.ctrl[self._torso_act])
        n = max(1, int(duration_s * self.control_hz))
        for a in np.linspace(0.0, 1.0, n + 1)[1:]:
            self.d.ctrl[self._torso_act] = (1 - a) * start + a * pitch
            self.step()
            self._refresh_kinematics()
            for s in SIDES:
                self.track_tcp(s, hold[s])
        for _ in range(int(0.5 * self.control_hz)):
            self.step()
            self._refresh_kinematics()
            for s in SIDES:
                self.track_tcp(s, hold[s])

    def _refresh_mj_pin(self) -> None:
        ak = self.kin["right"]
        pin.framesForwardKinematics(ak.model, ak.data, pin.neutral(ak.model))
        self.mj_pin = (self.d.body("torso_link").xpos
                       - ak.data.oMf[ak.model.getFrameId("torso_link")].translation).copy()

    # ------------------------------------------------------------------ physique
    def _apply_gravity(self) -> None:
        for s in SIDES:
            self.d.qfrc_applied[self._arm_dof[s]] = self.kin[s].gravity(self.arm_q(s))
        # buste : compensation par le biais MuJoCo (gravité de tout le haut du corps), sinon
        # le servo kp=400 fléchit de ~0.03 rad et la FK ment dès que les bras bougent
        self.d.qfrc_applied[self._torso_dof] = self.d.qfrc_bias[self._torso_dof]

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
                 duration_s: float = 2.0, torso_pitch: float = TORSO_PITCH) -> None:
        """Pose de départ des épisodes, pinces ouvertes, buste incliné à ``torso_pitch``.

        Par défaut, les bras vont aux angles de départ du G1 (``G1_START_Q``) : même pose
        effecteur dans la base WLA qu'à l'entraînement. ``poses`` (TCP 4×4 monde par bras)
        remplace ces angles par une IK.

        Ordre : buste incliné D'ABORD (bras en tuck, consignes articulaires tenues), puis
        rampe articulaire vers la pose : les angles de départ du G1 sont définis buste penché.
        """
        for s in SIDES:
            self.set_gripper(s, 0.0)
        if abs(torso_pitch - self.torso_pitch()) > 1e-3:
            self.set_torso_pitch(torso_pitch, hold_tcp=False)
        if poses is None:
            self.move_arms({"right": VIA_Q_RIGHT, "left": VIA_Q_RIGHT * ARM_MIRROR_SIGNS},
                           duration_s / 2)
            targets = {s: G1_START_Q[s].copy() for s in SIDES}
        else:
            targets = {}
            for s in SIDES:
                T = np.asarray(poses[s], float)
                q, ok = self.kin[s].ik(pin.SE3(T[:3, :3], T[:3, 3] - self.mj_pin), self.arm_q(s),
                                       posture=self.natural_q(s))
                if not ok:
                    raise RuntimeError(f"{s}: pose {np.round(T[:3, 3], 3)} hors d'atteinte")
                targets[s] = q
        self.move_arms(targets, duration_s)

    # ------------------------------------------------------------------ repères WLA
    def base_pose_wla(self) -> np.ndarray:
        """Repère base WLA en MONDE : bassin virtuel du G1 placé sous le buste du G1-D, tel
        que torse/base = celui du G1 (translation ``G1_PELVIS_TO_TORSO_XYZ`` puis tangage du
        buste, porté chez le G1 par le tangage de la taille)."""
        T_wt = np.eye(4)
        T_wt[:3, :3] = self.d.body("torso_link").xmat.reshape(3, 3)
        T_wt[:3, 3] = self.d.body("torso_link").xpos
        T_bt = np.eye(4)
        T_bt[:3, :3] = pin.utils.rpyToMatrix(0.0, self.torso_pitch(), 0.0)
        T_bt[:3, 3] = G1_PELVIS_TO_TORSO_XYZ
        return T_wt @ np.linalg.inv(T_bt)

    def _tcp_to_ee(self, side: str) -> np.ndarray:
        """Transformation fixe TCP sim -> effecteur WLA (4×4)."""
        k = self.kin[side]
        q = pin.neutral(k.model)
        pin.framesForwardKinematics(k.model, k.data, q)
        T_w = k.data.oMf[k.model.getFrameId(f"{side}_wrist_yaw_link")]
        T_wrist_tcp = T_w.actInv(k.fk(q)).homogeneous
        T_wrist_ee = np.eye(4)
        T_wrist_ee[:3, 3] = WLA_EE_IN_WRIST[side]
        return np.linalg.inv(T_wrist_tcp) @ T_wrist_ee

    def ee_pose_wla(self, side: str) -> np.ndarray:
        """Pose 4×4 de l'effecteur WLA dans le repère base WLA — ce que le modèle attend dans
        ``observation.state.*_ee_pose_gripper_base``."""
        T_we = self.tcp_pose(side) @ self._tcp_to_ee(side)
        return np.linalg.inv(self.base_pose_wla()) @ T_we

    def track_ee_wla(self, side: str, T_base_ee: np.ndarray, **kw) -> bool:
        """``track_tcp`` avec une cible effecteur WLA exprimée dans le repère base WLA
        (sortie absolue du serveur après composition)."""
        T_we = self.base_pose_wla() @ np.asarray(T_base_ee, float)
        return self.track_tcp(side, T_we @ np.linalg.inv(self._tcp_to_ee(side)), **kw)

    def lower_body_wla(self) -> np.ndarray:
        """``observation.state.lower_body`` du serveur WLA (15) : jambe gauche (6), jambe
        droite (6), taille [yaw, roll, pitch] (3). Jambes = G1 debout, taille = buste G1-D."""
        return np.concatenate([G1_STANDING_LEGS["left"], G1_STANDING_LEGS["right"], self.waist_wla()])

    def waist_wla(self) -> np.ndarray:
        """Slot taille WLA [yaw, roll, pitch] : chez le G1, le tangage de la taille EST le
        tangage du buste (égalité vérifiée sur un épisode). Le G1-D n'a que le tangage."""
        return np.array([0.0, 0.0, self.torso_pitch()])

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
        """``name`` = clé de ``CAMERAS`` (head_left, cam_wrist_left, cam_wrist_right) ou nom MJCF."""
        mj_name = self.cameras.get(name, name)
        if mj_name not in self._cams:
            w, h = self._image_size
            self._cams[mj_name] = SimCamera(self.m, self.d, mj_name, w, h)
        return self._cams[mj_name]

    def render(self, name: str) -> np.ndarray:
        return self.camera(name).rgb()

    def render_all(self) -> dict[str, np.ndarray]:
        """Les trois vues du VLA, clés ``head_left`` / ``cam_wrist_left`` / ``cam_wrist_right``."""
        return {k: self.render(k) for k in self.cameras}

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


__all__ = ["G1DSim", "CAMERAS", "SIDES", "ARM_JOINTS", "SCENE_XML", "URDF", "TORSO_PITCH", "G1_START_Q", "G1_STANDING_LEGS", "HEAD_VIEWS"]
