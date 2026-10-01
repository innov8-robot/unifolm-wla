"""Repères et conventions UnifoLM-WLA pour le Unitree G1-D (pur numpy, sans mujoco ni pinocchio).

Source unique des constantes utilisées par la sim (``sim/g1d_sim``) et par le convertisseur
d'enregistrements (``g1d_wla.convert_teleop``). Voir ``docs/G1D_Constats.md`` §9.
"""
from .frames import (G1_BASE_HEIGHT, G1_PELVIS_TO_TORSO_XYZ, G1_STANDING_LEGS, SIDES,
                     TORSO_PITCH_TRAINING, WLA_EE_IN_WRIST, base_T_torso, matrix_to_xyz_rpy,
                     torso_from_waist, waist_from_torso, wrist_T_ee)
from .urdf_fk import G1D_URDF, ArmFK

__all__ = ["G1_BASE_HEIGHT", "G1_PELVIS_TO_TORSO_XYZ", "G1_STANDING_LEGS", "SIDES",
           "TORSO_PITCH_TRAINING", "WLA_EE_IN_WRIST", "base_T_torso", "matrix_to_xyz_rpy",
           "torso_from_waist", "waist_from_torso", "wrist_T_ee", "G1D_URDF", "ArmFK"]
