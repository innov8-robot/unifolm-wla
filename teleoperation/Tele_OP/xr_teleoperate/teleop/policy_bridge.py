"""Pont téléop <-> serveur UnifoLM-WLA, avec CORRECTION EN DELTA par l'opérateur (G1-D, RECAP / Delta-0).

La politique WLA pilote les bras ; l'opérateur, casque sur la tête, peut à tout moment AJOUTER un
décalage au geste prévu, sans reprendre le contrôle bas niveau :

* bouton de GRIP (squeeze) d'une manette maintenu : le décalage de cette manette depuis l'appui
  (position, et rotation si ``rot_delta``) s'ajoute à la cible de la politique pour le bras de ce côté ;
* gâchette pressée pendant la correction : l'opérateur commande aussi la pince de ce côté ;
* chaque pas où une correction est active est enregistré ``intervention = 1`` ;
* au relâchement, la politique REPLANIFIE depuis l'état corrigé (nouveau chunk).

Conventions de repères (vérifiées par ``test_policy_bridge.py``) :

* le serveur WLA parle en « base WLA » = bassin (virtuel) du G1 sous le buste : torse = base ·
  translation (-0.004, 0, 0.044) · tangage du buste ; effecteur WLA = ``wrist_yaw`` + (0.105, ±0.003, 0) ;
* l'IK de la téléop (``G1_29_ArmIK``) vise ``L_ee`` / ``R_ee`` = ``wrist_yaw`` + (0.05, 0, 0) dans le repère
  de son modèle G1 à taille verrouillée à 0, soit la base WLA à buste DROIT ;
* observation : FK des angles MESURÉS avec l'URDF du G1-D (``g1d_wla``), exprimée dans la base WLA.

⚠ Non validé sur le robot. Premiers essais : vitesses bridées (``max_speed``), zone dégagée, arrêt
d'urgence à portée. L'indice du tangage du buste dans ``body.qpos`` (13) est une HYPOTHÈSE à vérifier.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[4]
for p in (REPO, REPO / "model_server"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from g1d_wla import G1_STANDING_LEGS, SIDES, ArmFK, base_T_torso, wrist_T_ee  # noqa: E402
from tools import msgpack_numpy  # noqa: E402

#: décalage de ``L_ee`` / ``R_ee`` de l'IK de la téléop dans ``*_wrist_yaw_link`` (robot_arm_ik.py)
IK_EE_OFFSET = np.array([0.05, 0.0, 0.0])
DEX1_OPEN = 5.4
#: la téléop convertit la gâchette (10 relâchée -> 0 pressée) en angle Dex1 par interp([5, 7] -> [0, 5.4])
TRIGGER_MIN, TRIGGER_MAX = 5.0, 7.0


def _T(R=None, p=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def xyz_rpy_to_T(x: np.ndarray) -> np.ndarray:
    return _T(Rotation.from_euler("xyz", x[3:6]).as_matrix(), x[:3])


def T_to_xyz_rot6d(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[:3, 3], T[:3, 0], T[:3, 1]]).astype(np.float32)


def dex1_to_trigger(g: float) -> float:
    """Angle Dex1 voulu (0 fermée -> 5.4 ouverte) -> valeur « gâchette » attendue par la téléop."""
    return float(np.clip(TRIGGER_MIN + (g / DEX1_OPEN) * (TRIGGER_MAX - TRIGGER_MIN), TRIGGER_MIN, TRIGGER_MAX))


class FrameConverter:
    """Base WLA <-> repère de l'IK de la téléop, pour un tangage de buste donné."""

    def __init__(self) -> None:
        self.fk = {s: ArmFK(s) for s in SIDES}
        self.T_ik_torso = base_T_torso(0.0)             # IK : modèle G1 à taille verrouillée à 0

    def ee_wla(self, side: str, arm_q: np.ndarray, torso_pitch: float) -> np.ndarray:
        """FK (angles mesurés, 7) -> effecteur WLA dans la base WLA (4×4)."""
        return base_T_torso(torso_pitch) @ self.fk[side].fk(arm_q)[0] @ wrist_T_ee(side)

    def wla_to_ik(self, side: str, T_base_ee: np.ndarray, torso_pitch: float) -> np.ndarray:
        """Effecteur WLA (base WLA) -> cible ``L_ee`` / ``R_ee`` de l'IK (repère IK)."""
        T_torso_wrist = np.linalg.inv(base_T_torso(torso_pitch)) @ T_base_ee @ np.linalg.inv(wrist_T_ee(side))
        return self.T_ik_torso @ T_torso_wrist @ _T(p=IK_EE_OFFSET)

    def ik_to_wla(self, side: str, T_ik: np.ndarray, torso_pitch: float) -> np.ndarray:
        T_torso_wrist = np.linalg.inv(self.T_ik_torso) @ T_ik @ np.linalg.inv(_T(p=IK_EE_OFFSET))
        return base_T_torso(torso_pitch) @ T_torso_wrist @ wrist_T_ee(side)


class PolicyBridge:
    def __init__(self, uri: str, instruction: str, unnorm_key: str = "UnifoLM_G1_Dex1",
                 advantage: str | None = None, exec_steps: int = 30, frequency: float = 30.0,
                 max_speed: float = 0.15, max_rot_speed: float = 1.0, delta_scale: float = 1.0,
                 rot_delta: bool = True, torso_pitch_index: int | None = 13, torso_pitch: float | None = None,
                 control_left: bool = True, control_right: bool = True) -> None:
        from websockets.sync.client import connect
        self.ws = connect(uri, max_size=None, open_timeout=10)
        self.packer = msgpack_numpy.Packer()
        self.meta = msgpack_numpy.unpackb(self.ws.recv())
        self.instruction, self.unnorm_key, self.advantage = instruction, unnorm_key, advantage
        self.exec_steps, self.freq = exec_steps, frequency
        self.max_step = max_speed / frequency
        self.max_rot_step = max_rot_speed / frequency
        self.delta_scale, self.rot_delta = delta_scale, rot_delta
        self.torso_pitch_index, self.torso_pitch_const = torso_pitch_index, torso_pitch
        self.control = {"left": control_left, "right": control_right}
        self.conv = FrameConverter()
        self.reset()

    # ------------------------------------------------------------------ cycle de vie
    def reset(self) -> None:
        """Début d'essai : vide le chunk et la mémoire du serveur (policy_reset)."""
        self.chunk, self.k = None, 0
        self.last_target = {s: None for s in SIDES}
        self.corr = {s: None for s in SIDES}          # (pose manette à l'appui) si correction active
        self.step_count = 0
        self.ws.send(self.packer.pack({"type": "policy_reset"}))
        self.ws.recv()

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass

    def _pitch(self, body_q) -> float:
        if self.torso_pitch_const is not None:
            return float(self.torso_pitch_const)
        if body_q is None or self.torso_pitch_index is None or len(body_q) <= self.torso_pitch_index:
            return 0.0
        return float(body_q[self.torso_pitch_index])

    # ------------------------------------------------------------------ requête au modèle
    def _query(self, images_bgr: dict, arm_q: np.ndarray, grip_dex1: np.ndarray, pitch: float) -> None:
        obs = {
            "observation.images.cam_left_high": images_bgr["head"],
            "observation.images.cam_left_wrist": images_bgr["left_wrist"],
            "observation.images.cam_right_wrist": images_bgr["right_wrist"],
            "observation.state.lower_body": np.concatenate(
                [G1_STANDING_LEGS["left"], G1_STANDING_LEGS["right"], [0.0, 0.0, pitch]]).astype(np.float32),
            "instruction": self.instruction,
            "unnorm_key": self.unnorm_key,
        }
        for i, s in enumerate(SIDES):
            q = arm_q[:7] if s == "left" else arm_q[7:14]
            obs[f"observation.state.{s}_ee_6d"] = T_to_xyz_rot6d(self.conv.ee_wla(s, q, pitch))
            obs[f"observation.state.{s}_gripper"] = np.array([grip_dex1[i]], np.float32)
        if self.advantage:
            obs["advantage"] = self.advantage
        t0 = time.perf_counter()
        self.ws.send(self.packer.pack({"type": "get_action", "obs": obs}))
        raw = self.ws.recv()
        if isinstance(raw, str):
            raise RuntimeError(f"erreur du serveur WLA :\n{raw}")
        act = msgpack_numpy.unpackb(raw)
        self.chunk = {s: np.asarray(act[f"action.{s}_ee_rpy"])[0] for s in SIDES}
        self.chunk_grip = {s: np.asarray(act[f"action.{s}_gripper"])[0, :, 0] for s in SIDES}
        self.k = 0
        self.last_latency = time.perf_counter() - t0

    # ------------------------------------------------------------------ un pas de contrôle
    def step(self, images_bgr: dict, arm_q: np.ndarray, grip_dex1: np.ndarray, body_q, tele_data) -> dict:
        """-> {"left"/"right": cible IK 4×4, "trigger": {côté: valeur gâchette}, "intervention": bool}.

        ``images_bgr`` : {"head": œil gauche 480×640 BGR, "left_wrist", "right_wrist"} ; ``arm_q`` (14) mesuré ;
        ``grip_dex1`` (2) mesuré (unité Dex1) ; ``body_q`` (35) ; ``tele_data`` : TeleData ou None."""
        pitch = self._pitch(body_q)
        squeeze = {s: bool(tele_data is not None and getattr(tele_data, f"{s}_ctrl_squeezeValue", 0.0) > 0.5)
                   for s in SIDES}
        # fin de correction -> replanifier depuis l'état corrigé
        if any(self.corr[s] is not None and not squeeze[s] for s in SIDES):
            self.chunk = None
        if self.chunk is None or self.k >= min(self.exec_steps, len(self.chunk["left"])):
            self._query(images_bgr, arm_q, grip_dex1, pitch)

        out = {"trigger": {}, "intervention": False}
        for i, s in enumerate(SIDES):
            q_meas = arm_q[:7] if s == "left" else arm_q[7:14]
            if not self.control[s]:
                out[s] = self.conv.wla_to_ik(s, self.conv.ee_wla(s, q_meas, pitch), pitch)   # tenir la pose
                continue
            T = self.conv.wla_to_ik(s, xyz_rpy_to_T(self.chunk[s][self.k]), pitch)
            g = float(self.chunk_grip[s][self.k])
            if squeeze[s]:
                P = np.asarray(getattr(tele_data, f"{s}_wrist_pose"), float)
                if self.corr[s] is None:
                    self.corr[s] = P.copy()
                P0 = self.corr[s]
                T = T.copy()
                T[:3, 3] += self.delta_scale * (P[:3, 3] - P0[:3, 3])
                if self.rot_delta:
                    T[:3, :3] = (P[:3, :3] @ P0[:3, :3].T) @ T[:3, :3]
                trig = float(getattr(tele_data, f"{s}_ctrl_triggerValue", 10.0))
                if trig < 9.0:                                       # gâchette pressée : l'opérateur tient la pince
                    g = float(np.interp(trig, [TRIGGER_MIN, TRIGGER_MAX], [0.0, DEX1_OPEN]))
                out["intervention"] = True
            else:
                self.corr[s] = None
            out[s] = self._limit(s, T)
            out["trigger"][s] = dex1_to_trigger(g)
        self.k += 1
        self.step_count += 1
        return out

    def _limit(self, side: str, T: np.ndarray) -> np.ndarray:
        """Borne la vitesse de la cible (translation et rotation par pas)."""
        prev = self.last_target[side]
        if prev is not None:
            d = T[:3, 3] - prev[:3, 3]
            n = float(np.linalg.norm(d))
            if n > self.max_step:
                T = T.copy()
                T[:3, 3] = prev[:3, 3] + d * (self.max_step / n)
            ang = float(np.linalg.norm(Rotation.from_matrix(prev[:3, :3].T @ T[:3, :3]).as_rotvec()))
            if ang > self.max_rot_step:
                T = T.copy()
                sl = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([prev[:3, :3], T[:3, :3]])))
                T[:3, :3] = sl(self.max_rot_step / ang).as_matrix()
        self.last_target[side] = T
        return T
