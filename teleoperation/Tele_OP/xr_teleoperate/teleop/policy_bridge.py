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

Protections (audit du 29/09) : borne de vitesse partant de la pose MESURÉE (premier pas, reprise) ;
décalage de correction remis à zéro à chaque nouveau chunk (le chunk part de la pose déjà corrigée) ;
décalage borné et pose manette suivie dans le repère MONDE (indépendante des mouvements de tête) ;
saut de manette (perte de suivi) -> décalage figé ; délai max sur les requêtes ; actions non finies
refusées (``PolicyError``) ; hystérésis sur le grip ; la gâchette ne prend la pince qu'au-delà de 30 %.

⚠ Non validé sur le robot. Premiers essais : vitesses bridées (``max_speed``), zone dégagée, arrêt
d'urgence à portée. L'indice du tangage du buste dans ``body.qpos`` est une HYPOTHÈSE à vérifier : dans
l'énumération G1_29 de la téléop, 12 = lacet, 13 = roulis, 14 = tangage de la taille ; la chaîne de
données (sim, convertisseur) utilise 13 par cohérence.
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
from g1d_wla import (G1_STANDING_LEGS, SIDES, ArmFK, base_T_torso, torso_from_waist,  # noqa: E402
                     waist_from_torso, wrist_T_ee)
from tools import msgpack_numpy  # noqa: E402

#: décalage de ``L_ee`` / ``R_ee`` de l'IK de la téléop dans ``*_wrist_yaw_link`` (robot_arm_ik.py)
IK_EE_OFFSET = np.array([0.05, 0.0, 0.0])
DEX1_OPEN = 5.4
#: la téléop convertit la gâchette (10 relâchée -> 0 pressée) en angle Dex1 par interp([5, 7] -> [0, 5.4])
TRIGGER_MIN, TRIGGER_MAX = 5.0, 7.0
#: grip : correction active au-delà de 0.5, relâchée en dessous de 0.3 (hystérésis)
GRIP_ON, GRIP_OFF = 0.5, 0.3
#: bornes du décalage de correction (depuis le début du chunk courant)
MAX_DELTA_POS, MAX_DELTA_ROT = 0.15, 0.6
#: saut de la manette entre deux pas au-delà duquel on suppose une perte de suivi
TRACKING_JUMP = 0.05
#: délai max d'une requête au serveur (s)
RECV_TIMEOUT = 3.0


class PolicyError(RuntimeError):
    """Le serveur n'a pas répondu à temps, a renvoyé une erreur, ou des actions invalides."""


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
    """Base WLA <-> repère de l'IK de la téléop, pour un buste donné (tangage, rotation gauche-droite).
    L'IK est résolue dans un repère lié au TORSE : la rotation du buste emporte les bras, et une
    cible fixe en base WLA reste atteinte pendant que le buste tourne (conversion au buste mesuré)."""

    def __init__(self) -> None:
        self.fk = {s: ArmFK(s) for s in SIDES}
        self.T_ik_torso = base_T_torso(0.0)             # IK : modèle G1 à taille verrouillée à 0

    def ee_wla(self, side: str, arm_q: np.ndarray, torso_pitch: float, torso_yaw: float = 0.0) -> np.ndarray:
        """FK (angles mesurés, 7) -> effecteur WLA dans la base WLA (4×4)."""
        return base_T_torso(torso_pitch, torso_yaw) @ self.fk[side].fk(arm_q)[0] @ wrist_T_ee(side)

    def wla_to_ik(self, side: str, T_base_ee: np.ndarray, torso_pitch: float, torso_yaw: float = 0.0) -> np.ndarray:
        """Effecteur WLA (base WLA) -> cible ``L_ee`` / ``R_ee`` de l'IK (repère IK)."""
        T_torso_wrist = (np.linalg.inv(base_T_torso(torso_pitch, torso_yaw)) @ T_base_ee
                         @ np.linalg.inv(wrist_T_ee(side)))
        return self.T_ik_torso @ T_torso_wrist @ _T(p=IK_EE_OFFSET)

    def ik_to_wla(self, side: str, T_ik: np.ndarray, torso_pitch: float, torso_yaw: float = 0.0) -> np.ndarray:
        T_torso_wrist = np.linalg.inv(self.T_ik_torso) @ T_ik @ np.linalg.inv(_T(p=IK_EE_OFFSET))
        return base_T_torso(torso_pitch, torso_yaw) @ T_torso_wrist @ wrist_T_ee(side)


class PolicyBridge:
    def __init__(self, uri: str, instruction: str, unnorm_key: str = "UnifoLM_G1_Dex1",
                 advantage: str | None = None, exec_steps: int = 30, frequency: float = 30.0,
                 max_speed: float = 0.15, max_rot_speed: float = 1.0, delta_scale: float = 1.0,
                 rot_delta: bool = True, torso_pitch_index: int | None = 13, torso_pitch: float | None = None,
                 control_left: bool = True, control_right: bool = True,
                 torso_yaw_index: int | None = None) -> None:
        from websockets.sync.client import connect
        self.ws = connect(uri, max_size=None, open_timeout=10)
        self.packer = msgpack_numpy.Packer()
        self.meta = msgpack_numpy.unpackb(self.ws.recv(timeout=RECV_TIMEOUT))
        self.instruction, self.unnorm_key, self.advantage = instruction, unnorm_key, advantage
        self.exec_steps, self.freq = exec_steps, frequency
        self.max_step = max_speed / frequency
        self.max_rot_step = max_rot_speed / frequency
        self.delta_scale, self.rot_delta = delta_scale, rot_delta
        self.torso_pitch_index, self.torso_pitch_const = torso_pitch_index, torso_pitch
        self.torso_yaw_index = torso_yaw_index         # None : buste supposé non tourné, rotation non pilotée
        self.control = {"left": control_left, "right": control_right}
        self.conv = FrameConverter()
        self.uri = uri
        self.reset()

    # ------------------------------------------------------------------ cycle de vie
    def reset(self) -> None:
        """Début d'essai : vide le chunk et la mémoire du serveur (policy_reset). Reconnecte si la
        connexion précédente a été perdue."""
        self.pause()
        self.step_count = 0
        try:
            self.ws.send(self.packer.pack({"type": "policy_reset"}))
            self.ws.recv(timeout=RECV_TIMEOUT)
        except Exception:
            from websockets.sync.client import connect
            self.close()
            self.ws = connect(self.uri, max_size=None, open_timeout=10)
            self.meta = msgpack_numpy.unpackb(self.ws.recv(timeout=RECV_TIMEOUT))
            self.ws.send(self.packer.pack({"type": "policy_reset"}))
            self.ws.recv(timeout=RECV_TIMEOUT)

    def pause(self) -> None:
        """La politique cesse de piloter (image absente, erreur, fin d'essai) : à la reprise, borne de
        vitesse repartant de la pose MESURÉE, nouveau chunk, pas de correction en cours."""
        self.chunk, self.k = None, 0
        self.last_target = {s: None for s in SIDES}
        self.corr = {s: None for s in SIDES}          # {"P0": pose manette de référence, "prev": pose au pas précédent}
        self.grip_on = {s: False for s in SIDES}

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

    def _yaw(self, body_q) -> float:
        if body_q is None or self.torso_yaw_index is None or len(body_q) <= self.torso_yaw_index:
            return 0.0
        return float(body_q[self.torso_yaw_index])

    # ------------------------------------------------------------------ requête au modèle
    def _query(self, images_bgr: dict, arm_q: np.ndarray, grip_dex1: np.ndarray, pitch: float, yaw: float = 0.0) -> None:
        obs = {
            "observation.images.cam_left_high": images_bgr["head"],
            "observation.images.cam_left_wrist": images_bgr["left_wrist"],
            "observation.images.cam_right_wrist": images_bgr["right_wrist"],
            "observation.state.lower_body": np.concatenate(
                [G1_STANDING_LEGS["left"], G1_STANDING_LEGS["right"], waist_from_torso(pitch, yaw)]).astype(np.float32),
            "instruction": self.instruction,
            "unnorm_key": self.unnorm_key,
        }
        for i, s in enumerate(SIDES):
            q = arm_q[:7] if s == "left" else arm_q[7:14]
            obs[f"observation.state.{s}_ee_6d"] = T_to_xyz_rot6d(self.conv.ee_wla(s, q, pitch, yaw))
            obs[f"observation.state.{s}_gripper"] = np.array([grip_dex1[i]], np.float32)
        if self.advantage:
            obs["advantage"] = self.advantage
        t0 = time.perf_counter()
        try:
            self.ws.send(self.packer.pack({"type": "get_action", "obs": obs}))
            raw = self.ws.recv(timeout=RECV_TIMEOUT)
        except Exception as e:
            raise PolicyError(f"serveur WLA sans réponse ({type(e).__name__}: {e})") from e
        if isinstance(raw, str):
            raise PolicyError(f"erreur du serveur WLA :\n{raw}")
        act = msgpack_numpy.unpackb(raw)
        try:
            chunk = {s: np.asarray(act[f"action.{s}_ee_rpy"], float)[0] for s in SIDES}
            grip = {s: np.asarray(act[f"action.{s}_gripper"], float)[0, :, 0] for s in SIDES}
            lb = act.get("action.lower_body")
            waist = None if lb is None else np.asarray(lb, float)[0, :, 12:15]
        except Exception as e:
            raise PolicyError(f"réponse du serveur mal formée ({e})") from e
        if not all(np.isfinite(chunk[s]).all() and np.isfinite(grip[s]).all() for s in SIDES) \
                or (waist is not None and not np.isfinite(waist).all()):
            raise PolicyError("actions non finies (NaN/inf) renvoyées par le modèle")
        self.chunk, self.chunk_grip, self.k = chunk, grip, 0
        # rotation du buste G1-D prédite (T,) ; None si le serveur ne renvoie pas la taille
        self.chunk_yaw = None if waist is None else torso_from_waist(waist)[1]
        self.last_latency = time.perf_counter() - t0

    # ------------------------------------------------------------------ un pas de contrôle
    def measured_ik(self, arm_q: np.ndarray, body_q) -> dict:
        """Poses IK correspondant aux angles MESURÉS (pour tenir la pose ou borner une reprise)."""
        pitch, yaw = self._pitch(body_q), self._yaw(body_q)
        return {s: self.conv.wla_to_ik(s, self.conv.ee_wla(s, arm_q[:7] if s == "left" else arm_q[7:14], pitch, yaw),
                                       pitch, yaw)
                for s in SIDES}

    def _controller_world(self, tele_data, side: str) -> np.ndarray:
        """Pose de la manette en repère MONDE : la téléop la rend relative à la position de la tête
        (translation seulement) ; on rajoute la tête pour que se pencher ne déplace pas le bras."""
        P = np.asarray(getattr(tele_data, f"{side}_wrist_pose"), float).copy()
        head = getattr(tele_data, "head_pose", None)
        if head is not None:
            P[:3, 3] += np.asarray(head, float)[:3, 3]
        return P

    def step(self, images_bgr: dict, arm_q: np.ndarray, grip_dex1: np.ndarray, body_q, tele_data) -> dict:
        """-> {"left"/"right": cible IK 4×4, "trigger": {côté: valeur gâchette}, "intervention": bool}.

        ``images_bgr`` : {"head": œil gauche 480×640 BGR, "left_wrist", "right_wrist"} ; ``arm_q`` (14) mesuré ;
        ``grip_dex1`` (2) mesuré (unité Dex1) ; ``body_q`` (35) ; ``tele_data`` : TeleData ou None.
        Lève ``PolicyError`` si le serveur ne répond pas ou répond des actions invalides.
        ``out["torso_yaw"]`` : rotation du buste prédite (rad), None si la rotation n'est pas pilotée."""
        pitch, yaw = self._pitch(body_q), self._yaw(body_q)
        meas = self.measured_ik(arm_q, body_q)
        for s in SIDES:                                    # reprise : la borne part de la pose mesurée
            if self.last_target[s] is None:
                self.last_target[s] = meas[s]
            v = float(getattr(tele_data, f"{s}_ctrl_squeezeValue", 0.0)) if tele_data is not None else 0.0
            self.grip_on[s] = v > GRIP_OFF if self.grip_on[s] else v > GRIP_ON
        # fin de correction -> replanifier depuis l'état corrigé
        if any(self.corr[s] is not None and not self.grip_on[s] for s in SIDES):
            self.chunk = None
        new_chunk = self.chunk is None or self.k >= min(self.exec_steps, len(self.chunk["left"]))
        if new_chunk:
            self._query(images_bgr, arm_q, grip_dex1, pitch, yaw)

        out = {"trigger": {}, "intervention": False,
               "torso_yaw": (float(self.chunk_yaw[self.k])
                             if self.torso_yaw_index is not None and self.chunk_yaw is not None else None)}
        for i, s in enumerate(SIDES):
            if not self.control[s]:
                out[s] = meas[s]                           # bras non piloté : tenir la pose mesurée
                out["trigger"][s] = dex1_to_trigger(float(grip_dex1[i]))
                continue
            T = self.conv.wla_to_ik(s, xyz_rpy_to_T(self.chunk[s][self.k]), pitch, yaw)
            g = float(self.chunk_grip[s][self.k])
            if self.grip_on[s]:
                P = self._controller_world(tele_data, s)
                c = self.corr[s]
                if c is None or new_chunk:
                    # début de correction, ou nouveau chunk (déjà parti de la pose corrigée) : le
                    # décalage repart de zéro depuis la pose actuelle de la manette
                    c = self.corr[s] = {"P0": P.copy(), "prev": P.copy()}
                elif np.linalg.norm(P[:3, 3] - c["prev"][:3, 3]) > TRACKING_JUMP:
                    P = c["prev"]                          # saut de manette : perte de suivi supposée
                c["prev"] = P.copy()
                d = self.delta_scale * (P[:3, 3] - c["P0"][:3, 3])
                n = float(np.linalg.norm(d))
                if n > MAX_DELTA_POS:
                    d *= MAX_DELTA_POS / n
                T = T.copy()
                T[:3, 3] += d
                if self.rot_delta:
                    rv = Rotation.from_matrix(P[:3, :3] @ c["P0"][:3, :3].T).as_rotvec()
                    a = float(np.linalg.norm(rv))
                    if a > MAX_DELTA_ROT:
                        rv *= MAX_DELTA_ROT / a
                    T[:3, :3] = Rotation.from_rotvec(rv).as_matrix() @ T[:3, :3]
                trig = float(getattr(tele_data, f"{s}_ctrl_triggerValue", 10.0))
                if trig < TRIGGER_MAX:                     # gâchette pressée à plus de 30 % : pince à l'opérateur
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
