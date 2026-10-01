"""Constantes et transformations du contrat iso WLA (docs/G1D_Constats.md §9)."""
from __future__ import annotations

import numpy as np

SIDES = ("left", "right")

#: bassin du G1 -> torso_link, tailles yaw = roll = 0 : translation fixe puis tangage de la taille
#: (URDF officiel g1_29dof_mode_15_with_dex1_1, vérifié par FK). La base WLA du G1-D est un
#: bassin VIRTUEL placé sous le buste avec cette même relation.
G1_PELVIS_TO_TORSO_XYZ = np.array([-0.0039635, 0.0, 0.044])

#: effecteur WLA (``*_ee_pose_gripper_base``) dans ``*_wrist_yaw_link`` : wrist_yaw + 0.105 m en x,
#: sans rotation (exact sur le G1). Sur le G1-D, même point par rapport à la pince (±3 mm en y).
WLA_EE_IN_WRIST = {"left": np.array([0.105, 0.003, 0.0]), "right": np.array([0.105, -0.003, 0.0])}

#: pince Dex1 en unité moteur : 0 fermée -> DEX1_OPEN ouverte (état et action)
DEX1_OPEN = 5.4

#: indices du buste dans ``body.qpos`` (35 moteurs, disposition G1_29) écrits par la sim ; sur le
#: robot G1-D, HYPOTHÈSES à vérifier (12 = lacet de taille du G1_29, 13 = roulis chez le G1_29)
TORSO_YAW_INDEX = 12
TORSO_PITCH_INDEX = 13

#: tangage médian du buste à l'entraînement (``state_torso``, 54 M frames), 0.13–0.18 selon la tâche
TORSO_PITCH_TRAINING = 0.166

#: hauteur de bassin commandée du G1 debout dans les tâches de table Dex1 (``action.base_command[3]``)
G1_BASE_HEIGHT = 0.732

#: jambes du G1 debout pendant les tâches de table Dex1, [hip_pitch, hip_roll, hip_yaw, knee,
#: ankle_pitch, ankle_roll] (médianes Stack_Block + Wipe_Table). Slots VALIDES à l'entraînement.
G1_STANDING_LEGS = {"left": np.array([-0.408, 0.027, -0.032, 0.664, -0.25, -0.016]),
                    "right": np.array([-0.44, 0.005, 0.026, 0.639, -0.192, 0.007])}


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def base_T_torso(torso_pitch: float, torso_yaw: float = 0.0) -> np.ndarray:
    """Pose 4×4 de ``torso_link`` dans la base WLA. Chaîne du G1-D : tangage du buste
    (``Yaw_Joint``, axe y malgré son nom) PUIS rotation gauche-droite (``torso_Joint``, axe z).
    Le pivot du lacet est à l'origine du torse : la base WLA reste fixe quand le buste tourne. Le
    pivot du tangage est 0,104 m plus bas : la base virtuelle bougerait si le tangage variait en
    cours d'épisode (il est constant en pratique)."""
    T = np.eye(4)
    T[:3, :3] = _rot_y(float(torso_pitch)) @ _rot_z(float(torso_yaw))
    T[:3, 3] = G1_PELVIS_TO_TORSO_XYZ
    return T


def waist_from_torso(torso_pitch, torso_yaw=0.0) -> np.ndarray:
    """Slot taille WLA [yaw, roll, pitch] du G1 (chaîne Rz·Rx·Ry) qui donne la MÊME orientation
    de torse que le G1-D (Ry(tangage)·Rz(lacet)). Lacet nul -> [0, 0, tangage]. Vectorisé."""
    p, y = np.broadcast_arrays(np.asarray(torso_pitch, float), np.asarray(torso_yaw, float))
    cp, sp, cy, sy = np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    # R = Ry(p) Rz(y) : colonnes utiles pour Rz(a) Rx(b) Ry(c)
    r01, r11 = -cp * sy, cy
    r20, r21, r22 = -sp * cy, sp * sy, cp
    return np.stack([np.arctan2(-r01, r11), np.arcsin(np.clip(r21, -1, 1)), np.arctan2(-r20, r22)], axis=-1)


def torso_from_waist(waist) -> tuple[np.ndarray, np.ndarray]:
    """Inverse (au roulis résiduel près) : taille WLA [yaw, roll, pitch] -> (tangage, lacet) G1-D."""
    w = np.asarray(waist, float)
    a, b, c = w[..., 0], w[..., 1], w[..., 2]
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    # R = Rz(a) Rx(b) Ry(c) ; G1-D : R = Ry(p) Rz(y) -> p = atan2(R02, R22), y = atan2(R10, R11)
    r02 = ca * sc + sa * sb * cc
    r22 = cb * cc
    r10 = sa * cc + ca * sb * sc
    r11 = ca * cb
    return np.arctan2(r02, r22), np.arctan2(r10, r11)


def wrist_T_ee(side: str) -> np.ndarray:
    """Pose 4×4 de l'effecteur WLA dans ``*_wrist_yaw_link``."""
    T = np.eye(4)
    T[:3, 3] = WLA_EE_IN_WRIST[side]
    return T


def matrix_to_xyz_rpy(T: np.ndarray) -> np.ndarray:
    """(…, 4, 4) -> (…, 6) xyz + euler ``xyz`` EXTRINSÈQUE (convention des datasets G1 Dex1,
    identique à ``scipy.spatial.transform.Rotation.as_euler('xyz')``)."""
    T = np.asarray(T, float)
    R = T[..., :3, :3]
    # R = Rz(yaw) · Ry(pitch) · Rx(roll)
    pitch = np.arcsin(np.clip(-R[..., 2, 0], -1.0, 1.0))
    roll = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.concatenate([T[..., :3, 3], np.stack([roll, pitch, yaw], axis=-1)], axis=-1)
