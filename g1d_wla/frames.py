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


def base_T_torso(torso_pitch: float) -> np.ndarray:
    """Pose 4×4 de ``torso_link`` dans la base WLA pour un tangage de buste donné."""
    T = np.eye(4)
    T[:3, :3] = _rot_y(float(torso_pitch))
    T[:3, 3] = G1_PELVIS_TO_TORSO_XYZ
    return T


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
