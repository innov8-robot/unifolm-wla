"""SimCamera — rendu RGB d'une caméra MJCF.

Extrait de ``mpc_any/src/oc_svfp/sim/sim_camera.py``, réduit au RGB : WLA ne prend que des
images couleur. ``rgb() -> HxWx3 uint8`` et ``intrinsics = (fx, fy, cx, cy)``.

⚠ Rendu hors écran : lancer avec ``MUJOCO_GL=egl`` sur une machine sans affichage.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np


class SimCamera:
    def __init__(self, model, data, camera: str, width: int = 640, height: int = 480) -> None:
        self._data = data
        self._camera = camera
        self._rgb = mujoco.Renderer(model, height, width)
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

    def close(self) -> None:
        self._rgb.close()
