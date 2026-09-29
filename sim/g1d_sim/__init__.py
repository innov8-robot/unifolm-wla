"""Sim MuJoCo minimale du G1-D + Dex1-1 (extraite de mpc_any) pour brancher Hy-VLA."""

from .camera import SimCamera
from .kinematics import ARM_JOINTS, ArmKinematics
from .robot import CAMERAS, HEAD_VIEWS, SCENE_XML, SIDES, URDF, G1DSim

__all__ = ["G1DSim", "ArmKinematics", "SimCamera", "CAMERAS", "SIDES", "ARM_JOINTS",
           "SCENE_XML", "URDF", "HEAD_VIEWS"]
