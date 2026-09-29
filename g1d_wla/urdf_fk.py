"""Cinématique directe torso_link -> *_wrist_yaw_link du G1-D, lue dans l'URDF (numpy seul).

La chaîne du bras part de ``torso_link`` : elle ne dépend ni de la colonne ni du tangage du buste,
qui n'interviennent que dans ``base_T_torso``. Validée contre pinocchio (voir ``sim/smoke.py``).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

G1D_URDF = Path(__file__).resolve().parents[1] / "sim" / "assets" / "g1_d_dex1.urdf"

#: ordre des 7 joints d'un bras, identique à xr_teleoperate (G1_29_JointArmIndex) et aux datasets
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw")


def _rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(N,) angles -> (N, 3, 3) rotations autour d'``axis`` (unitaire)."""
    x, y, z = axis
    c, s = np.cos(q), np.sin(q)
    C = 1.0 - c
    return np.stack([np.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1),
                     np.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1),
                     np.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1)], -2)


class ArmFK:
    """FK vectorisée d'un bras : ``fk(q)`` avec q (N, 7) -> (N, 4, 4) wrist_yaw dans torso_link."""

    def __init__(self, side: str, urdf: str | Path = G1D_URDF, root: str = "torso_link") -> None:
        joints = {}
        for j in ET.parse(str(urdf)).getroot().findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = np.array([float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
            a = j.find("axis")
            axis = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else None
            joints[j.find("child").get("link")] = (j.get("name"), j.get("type"), j.find("parent").get("link"),
                                                   xyz, rpy, axis)
        chain, link = [], f"{side}_wrist_yaw_link"
        while link != root:
            if link not in joints:
                raise ValueError(f"{link} n'est pas relié à {root} dans {urdf}")
            chain.append(joints[link])
            link = joints[link][2]
        self.chain = chain[::-1]
        self.names = [c[0] for c in self.chain if c[1] == "revolute"]
        expected = [f"{side}_{n}_joint" for n in ARM_JOINTS]
        if self.names != expected:
            raise ValueError(f"chaîne inattendue {self.names}, attendu {expected}")

    def fk(self, q: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(np.asarray(q, float))
        T = np.tile(np.eye(4), (q.shape[0], 1, 1))
        k = 0
        for _name, jtype, _parent, xyz, rpy, axis in self.chain:
            O = np.eye(4)
            O[:3, :3], O[:3, 3] = _rpy_matrix(*rpy), xyz
            T = T @ O
            if jtype == "revolute":
                Rq = np.tile(np.eye(4), (q.shape[0], 1, 1))
                Rq[:, :3, :3] = _axis_angle(axis / np.linalg.norm(axis), q[:, k])
                T = T @ Rq
                k += 1
        return T
