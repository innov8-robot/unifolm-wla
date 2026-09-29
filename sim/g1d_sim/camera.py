"""SimCamera — rendu RGB (+ depth métrique) d'une caméra MJCF.

Extrait de ``mpc_any/src/oc_svfp/sim/sim_camera.py`` : même contrat que ``OrbbecClient``,
``read() -> (rgb HxWx3 uint8, depth_mm HxW uint16)`` et ``intrinsics = (fx, fy, cx, cy)``.
La depth MuJoCo est métrique ; convertie en uint16 mm, 0 = invalide (> ``max_range_m``).

⚠ Rendu hors écran : lancer avec ``MUJOCO_GL=egl`` sur une machine sans affichage.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np


class SimCamera:
    def __init__(self, model, data, camera: str, width: int = 640, height: int = 480,
                 max_range_m: float = 10.0, depth: bool = True) -> None:
        self._data = data
        self._camera = camera
        self._max_range = max_range_m
        self._rgb = mujoco.Renderer(model, height, width)
        self._depth = None
        if depth:
            self._depth = mujoco.Renderer(model, height, width)
            self._depth.enable_depth_rendering()
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            raise ValueError(f"caméra '{camera}' absente de la scène")
        # pinhole idéal depuis le fovy MJCF (pixels carrés, centre optique au centre)
        f = 0.5 * height / math.tan(math.radians(model.cam_fovy[cam_id]) / 2)
        self.intrinsics: tuple[float, float, float, float] = (f, f, width / 2, height / 2)

    def rgb(self) -> np.ndarray:
        """Image RGB uint8 HxWx3 de l'état courant (après mj_forward/mj_step)."""
        self._rgb.update_scene(self._data, camera=self._camera)
        return self._rgb.render().copy()

    def read(self) -> tuple[np.ndarray, np.ndarray | None]:
        rgb = self.rgb()
        if self._depth is None:
            return rgb, None
        self._depth.update_scene(self._data, camera=self._camera)
        z = self._depth.render()                                   # float32, mètres
        valid = np.isfinite(z) & (z > 0) & (z < self._max_range)
        return rgb, np.where(valid, z * 1000.0, 0.0).astype(np.uint16)

    def close(self) -> None:
        self._rgb.close()
        if self._depth is not None:
            self._depth.close()
