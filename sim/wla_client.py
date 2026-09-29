"""Boucle fermée sim G1-D <-> serveur UnifoLM-WLA : premier test zero-shot sans robot.

1. lancer le serveur (env uv du projet, racine du dépôt) ::

       .venv/bin/python -m model_server.action_server_wbc_msgpack_unitree \\
           --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \\
           --unnorm_key UnifoLM_G1_Dex1 --port 8600

2. lancer ce client (env de la sim) ::

       MUJOCO_GL=egl python sim/wla_client.py --instruction "pick up the black part" --chunks 10

À chaque cycle : observation au format WLA (3 images BGR, effecteurs xyz + rot6d dans la base WLA,
pinces en unité Dex1, bas du corps), puis exécution des ``--exec-steps`` premiers pas du chunk à
30 Hz. Sorties dans ``--out`` : vidéo des 3 vues, trajectoires et journal JSON.

⚠ Correspondance pince sim <-> Dex1 : LINÉAIRE SUPPOSÉE (0 = fermée, DEX1_OPEN = ouverte), à
calibrer sur le robot (TODO de MYREADME.md).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "model_server"))
from g1d_sim import SIDES, G1DSim  # noqa: E402
from tools import msgpack_numpy  # noqa: E402

#: pince Dex1 grande ouverte, unité moteur (xr_teleoperate : 0 fermée -> 5.4 ouverte)
DEX1_OPEN = 5.4
ROLE_TO_OBS = {"head_left": "observation.images.cam_left_high",
               "cam_wrist_left": "observation.images.cam_left_wrist",
               "cam_wrist_right": "observation.images.cam_right_wrist"}


def closure_to_dex1(c: float) -> float:
    return DEX1_OPEN * (1.0 - float(np.clip(c, 0.0, 1.0)))


def dex1_to_closure(g: float) -> float:
    return float(np.clip(1.0 - g / DEX1_OPEN, 0.0, 1.0))


def ee_to_xyz_rot6d(T: np.ndarray) -> np.ndarray:
    """4×4 -> (9,) xyz + rot6d = deux premières COLONNES [R00,R10,R20,R01,R11,R21]."""
    return np.concatenate([T[:3, 3], T[:3, 0], T[:3, 1]]).astype(np.float32)


def xyz_rpy_to_matrix(x: np.ndarray) -> np.ndarray:
    """(6,) xyz + euler ``xyz`` extrinsèque -> 4×4 (convention du serveur)."""
    r, p, y = x[3:6]
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    T = np.eye(4)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    T[:3, 3] = x[:3]
    return T


def build_obs(sim: G1DSim, instruction: str, unnorm_key: str | None) -> dict:
    imgs = sim.render_all()
    obs = {ROLE_TO_OBS[r]: np.ascontiguousarray(img[..., ::-1]) for r, img in imgs.items()}  # RGB -> BGR
    for s in SIDES:
        obs[f"observation.state.{s}_ee_6d"] = ee_to_xyz_rot6d(sim.ee_pose_wla(s))
        obs[f"observation.state.{s}_gripper"] = np.array([closure_to_dex1(sim.gripper(s))], np.float32)
    obs["observation.state.lower_body"] = sim.lower_body_wla().astype(np.float32)
    obs["instruction"] = instruction
    if unnorm_key:
        obs["unnorm_key"] = unnorm_key
    return obs, imgs


async def run(args) -> dict:
    import websockets

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sim = G1DSim(head_view=args.head_view)
    sim.go_ready()
    frames, log = [], {"args": vars(args), "chunks": []}
    packer = msgpack_numpy.Packer()
    async with websockets.connect(args.uri, max_size=None, ping_interval=None) as ws:
        meta = msgpack_numpy.unpackb(await ws.recv())
        print(f"serveur : {meta.get('env')} chunk={meta.get('action_chunk_size')} "
              f"clés norm={meta.get('available_unnorm_keys')}", flush=True)
        for c in range(args.chunks):
            obs, imgs = build_obs(sim, args.instruction, args.unnorm_key)
            t0 = time.perf_counter()
            await ws.send(packer.pack({"type": "get_action", "obs": obs}))
            raw = await ws.recv()
            if isinstance(raw, str):
                raise RuntimeError(f"erreur serveur :\n{raw}")
            act = msgpack_numpy.unpackb(raw)
            dt = time.perf_counter() - t0
            chunk = {s: np.asarray(act[f"action.{s}_ee_rpy"])[0] for s in SIDES}
            grip = {s: np.asarray(act[f"action.{s}_gripper"])[0, :, 0] for s in SIDES}
            n_exec = min(args.exec_steps, len(chunk["left"]))
            refused = {s: 0 for s in SIDES}
            ee_before = {s: sim.ee_pose_wla(s)[:3, 3].copy() for s in SIDES}
            for t in range(n_exec):
                for s in SIDES:
                    refused[s] += not sim.track_ee_wla(s, xyz_rpy_to_matrix(chunk[s][t]))
                    sim.set_gripper(s, dex1_to_closure(grip[s][t]))
                sim.step()
                if t % args.video_every == 0:
                    v = sim.render_all()
                    frames.append(np.concatenate([v["head_left"], v["cam_wrist_left"], v["cam_wrist_right"]], axis=1))
            entry = {
                "chunk": c, "inference_s": round(dt, 3), "exec_steps": n_exec, "ik_refused": refused,
                "ee_start": {s: np.round(ee_before[s], 4).tolist() for s in SIDES},
                "ee_target_end": {s: np.round(chunk[s][n_exec - 1, :3], 4).tolist() for s in SIDES},
                "ee_reached": {s: np.round(sim.ee_pose_wla(s)[:3, 3], 4).tolist() for s in SIDES},
                "gripper_cmd_dex1": {s: [round(float(grip[s][0]), 3), round(float(grip[s][n_exec - 1]), 3)] for s in SIDES},
                "piece_world": np.round(sim.object_pose()[:3, 3], 4).tolist(),
            }
            log["chunks"].append(entry)
            print(f"chunk {c}: inférence {dt:.2f}s | IK refusées {refused} | "
                  f"EE droit {entry['ee_start']['right']} -> {entry['ee_reached']['right']} | "
                  f"pince D {entry['gripper_cmd_dex1']['right']}", flush=True)
    (out / "log.json").write_text(json.dumps(log, indent=2))
    try:
        import imageio.v3 as iio
        iio.imwrite(out / "rollout.mp4", np.stack(frames), fps=30 / args.video_every)
        print(f"vidéo : {out / 'rollout.mp4'} ({len(frames)} images)")
    except Exception as e:  # imageio/ffmpeg absent : on garde au moins la première et la dernière vue
        print(f"vidéo non écrite ({e}) ; images de début et de fin sauvegardées")
        import imageio.v3 as iio
        iio.imwrite(out / "first.png", frames[0])
        iio.imwrite(out / "last.png", frames[-1])
    sim.close()
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default="ws://127.0.0.1:8600")
    ap.add_argument("--instruction", default="pick up the black part and put it in the box")
    ap.add_argument("--unnorm_key", default="UnifoLM_G1_Dex1")
    ap.add_argument("--head-view", dest="head_view", choices=["rec", "raw"], default="rec")
    ap.add_argument("--chunks", type=int, default=10, help="nombre de requêtes au modèle")
    ap.add_argument("--exec-steps", dest="exec_steps", type=int, default=20,
                    help="pas exécutés par chunk (30 = chunk entier)")
    ap.add_argument("--video-every", dest="video_every", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "wla_out"))
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
