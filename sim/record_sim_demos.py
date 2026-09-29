"""Enregistre des démos expertes d'une tâche de sim (``--task cube|novares``) au format xr_teleoperate.

Même format et même ordre qu'un enregistrement réel du G1-D, pour passer ensuite par le MÊME
convertisseur (``g1d_wla.convert_teleop``) que les vraies démos :

* images : disposition monoculaire de la téléop, ``color_0`` tête (œil gauche, vue ``--head-view``),
  ``color_1`` poignet gauche, ``color_2`` poignet droit, 640×480 ;
* ``states`` : angles MESURÉS des bras, pinces en unité Dex1, ``body.qpos`` à 35 moteurs avec le
  tangage du buste à l'indice 13 (hypothèse du convertisseur) et les bras en 15–28 ;
* ``actions`` : angles COMMANDÉS (consignes servo après IK), pinces commandées, base à l'arrêt.

Usage (env de la sim) ::

    MUJOCO_GL=egl python sim/record_sim_demos.py --n 150 \\
        --out teleoperation/Tele_OP/xr_teleoperate/teleop/utils/data/sim_cube
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from g1d_sim import SIDES, G1DSim  # noqa: E402
from sim_tasks import get_task  # noqa: E402
from wla_client import closure_to_dex1  # noqa: E402

TORSO_PITCH_INDEX = 13
ARM_BODY_SLICE = {"left": slice(15, 22), "right": slice(22, 29)}
VIEWS = ("head_left", "cam_wrist_left", "cam_wrist_right")


def record_episode(sim: G1DSim, rng: np.random.Generator, ep_dir: Path, jpeg_quality: int, task) -> dict:
    colors_dir = ep_dir / "colors"
    colors_dir.mkdir(parents=True)
    steps = []

    def on_step(t: int, closure_cmd: dict) -> None:
        imgs = sim.render_all()
        colors = {}
        for k, view in enumerate(VIEWS):
            name = f"colors/{t:06d}_color_{k}.jpg"
            Image.fromarray(imgs[view]).save(ep_dir / name, quality=jpeg_quality)
            colors[f"color_{k}"] = name
        body = np.zeros(35)
        body[TORSO_PITCH_INDEX] = sim.torso_pitch()
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
        steps.append({"idx": t, "colors": colors, "depths": {}, "states": states, "actions": actions,
                      "tactiles": {}, "audios": {}, "sim_state": ""})

    res = task.expert(sim, rng, on_step=on_step)
    doc = {"info": {"version": "1.0.0", "author": "g1d_sim", "image": {"width": 640, "height": 480, "fps": 30.0},
                    "source": "sim/record_sim_demos.py"},
           "text": {"goal": task.instruction, "desc": f"expert scripté, sim MuJoCo G1-D, tâche {task.name}", "steps": ""},
           "data": steps}
    (ep_dir / "data.json").write_text(json.dumps(doc))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150, help="nombre d'épisodes RÉUSSIS à garder")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--task", choices=["cube", "novares"], default="cube")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--head-view", dest="head_view", choices=["raw", "rec"], default="raw")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.out.exists():
        if not a.overwrite:
            raise SystemExit(f"{a.out} existe déjà (--overwrite)")
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    rng = np.random.default_rng(a.seed)
    task = get_task(a.task)
    sim = G1DSim(scene_xml=task.scene_xml, head_view=a.head_view)
    kept, tried, t0 = 0, 0, time.time()
    summary = []
    while kept < a.n:
        sim.reset()
        sim.go_ready()
        task.place(sim, rng)
        ep_dir = a.out / f"episode_{kept:04d}"
        res = record_episode(sim, rng, ep_dir, a.jpeg_quality, task)
        tried += 1
        if res["success"] and res["ik_refused"] == 0:
            kept += 1
            summary.append(res)
            print(f"garde {kept}/{a.n} (essai {tried}) soulevé {res['lifted']*100:.1f} cm "
                  f"| {time.time() - t0:.0f} s", flush=True)
        else:
            shutil.rmtree(ep_dir)
            print(f"rejet essai {tried} : succès={res['success']} IK refusées={res['ik_refused']}", flush=True)
    (a.out / "summary.json").write_text(json.dumps({"kept": kept, "tried": tried, "episodes": summary}, indent=1))
    print(f"{kept} épisodes gardés sur {tried} essais, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
