"""Écriture d'un épisode de sim au format xr_teleoperate (JSON + JPEG), partagée par l'enregistreur de
démos expertes (``record_sim_demos.py``) et les rollouts RECAP (``recap_rollouts.py``).

Même format et même ordre qu'un enregistrement réel du G1-D : ``record(t, ...)`` est appelé APRÈS
avoir posé les commandes du pas et AVANT ``sim.step()`` ; l'état est donc MESURÉ et l'action est la
consigne servo du pas (angles commandés après IK), comme dans xr_teleoperate.

* images : ``color_0`` tête (œil gauche), ``color_1`` poignet gauche, ``color_2`` poignet droit ;
* ``states`` / ``actions`` : bras, pinces en unité Dex1, ``body.qpos`` à 35 moteurs (tangage du buste
  à l'indice 13, bras en 15–28) ;
* champs par pas OPTIONNELS (``extra``), p. ex. ``intervention`` (RECAP) ; champs d'épisode dans
  ``info`` (``success_step``, ``outcome``...).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from g1d_sim import SIDES, G1DSim
from g1d_wla.frames import DEX1_OPEN, TORSO_PITCH_INDEX, TORSO_YAW_INDEX  # (source unique ; g1d_sim met le dépôt dans le chemin)

ARM_BODY_SLICE = {"left": slice(15, 22), "right": slice(22, 29)}
VIEWS = ("head_left", "cam_wrist_left", "cam_wrist_right")


def closure_to_dex1(c: float) -> float:
    return DEX1_OPEN * (1.0 - float(np.clip(c, 0.0, 1.0)))


class SimEpisodeWriter:
    def __init__(self, sim: G1DSim, ep_dir: Path, jpeg_quality: int = 95) -> None:
        self.sim, self.ep_dir, self.q = sim, Path(ep_dir), jpeg_quality
        (self.ep_dir / "colors").mkdir(parents=True)
        self.steps = []

    def record(self, t: int, closure_cmd: dict, extra: dict | None = None) -> None:
        sim = self.sim
        imgs = sim.render_all()
        colors = {}
        for k, view in enumerate(VIEWS):
            name = f"colors/{t:06d}_color_{k}.jpg"
            Image.fromarray(imgs[view]).save(self.ep_dir / name, quality=self.q)
            colors[f"color_{k}"] = name
        body = np.zeros(35)
        body[TORSO_PITCH_INDEX] = sim.torso_pitch()
        body[TORSO_YAW_INDEX] = sim.torso_yaw()
        states, actions = {}, {}
        for s in SIDES:
            q = sim.arm_q(s)
            body[ARM_BODY_SLICE[s]] = q
            states[f"{s}_arm"] = {"qpos": q.tolist(), "qvel": [], "torque": []}
            actions[f"{s}_arm"] = {"qpos": sim.d.ctrl[sim._arm_act[s]].tolist(), "qvel": [], "torque": []}
            states[f"{s}_ee"] = {"qpos": [closure_to_dex1(sim.gripper(s))], "qvel": [], "torque": []}
            actions[f"{s}_ee"] = {"qpos": [closure_to_dex1(closure_cmd[s])], "qvel": [], "torque": []}
        states["body"] = {"qpos": body.tolist()}
        actions["body"] = {"qpos": [0.0, 0.0, 0.0]}
        step = {"idx": t, "colors": colors, "depths": {}, "states": states, "actions": actions,
                "tactiles": {}, "audios": {}, "sim_state": ""}
        if extra:
            step.update(extra)
        self.steps.append(step)

    def save(self, goal: str, desc: str, source: str, info_extra: dict | None = None) -> None:
        info = {"version": "1.0.0", "author": "g1d_sim", "image": {"width": 640, "height": 480, "fps": 30.0},
                "source": source,
                "body_layout": {"torso_pitch": TORSO_PITCH_INDEX, "torso_yaw": TORSO_YAW_INDEX}}
        if info_extra:
            info.update(info_extra)
        doc = {"info": info, "text": {"goal": goal, "desc": desc, "steps": ""}, "data": self.steps}
        (self.ep_dir / "data.json").write_text(json.dumps(doc))
