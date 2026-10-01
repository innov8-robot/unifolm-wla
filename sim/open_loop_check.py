"""Contrôle en BOUCLE OUVERTE : le modèle reproduit-il les actions des démos à partir de leurs images ?

Pour des épisodes d'un dataset converti (état et actions : parquet LeRobot ; images : JPEG bruts
xr_teleoperate du même épisode), envoie au serveur WLA l'observation enregistrée au pas t et compare
le chunk prédit aux 30 actions enregistrées [t, t+30). Référence : « ne rien faire » (tenir la pose
courante). Aucune simulation n'est jouée : une bonne boucle ouverte avec une mauvaise boucle fermée
désigne l'accumulation d'erreurs ; une mauvaise boucle ouverte désigne l'apprentissage ou les données.

Usage (env de la sim ; serveur WLA lancé sur le checkpoint) ::

    python sim/open_loop_check.py --dataset playground/Datasets/g1d_sim_stack_n100/sim_stack \\
        --raw playground/sim_raw/stack_n100 --episodes 0 1 2 --instruction "stack the black part on the other one"
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "model_server"))
from g1d_sim import SIDES  # noqa: E402
from g1d_wla.frames import G1_STANDING_LEGS  # noqa: E402
from tools import msgpack_numpy  # noqa: E402
from wla_client import ee_to_xyz_rot6d, xyz_rpy_to_matrix  # noqa: E402

#: JPEG bruts -> clés d'image attendues par le serveur, selon la disposition (3 images : sim ;
#: 4 images : robot réel, tête stéréo gauche/droite puis poignets ; seul l'œil GAUCHE va au modèle)
RAW_TO_OBS = {3: {"color_0": "observation.images.cam_left_high",
                  "color_1": "observation.images.cam_left_wrist",
                  "color_2": "observation.images.cam_right_wrist"},
              4: {"color_0": "observation.images.cam_left_high",
                  "color_2": "observation.images.cam_left_wrist",
                  "color_3": "observation.images.cam_right_wrist"}}


def build_obs(row, ep_raw: Path, item: dict, instruction: str, unnorm_key: str, lower_body) -> dict:
    mapping = RAW_TO_OBS[len(item["colors"])]
    obs = {mapping[k]: cv2.imread(str(ep_raw / p)) for k, p in item["colors"].items() if k in mapping}  # BGR
    for s in SIDES:
        T = xyz_rpy_to_matrix(np.asarray(row[f"observation.state.{s}_ee_pose_gripper_base"], float))
        obs[f"observation.state.{s}_ee_6d"] = ee_to_xyz_rot6d(T)
        obs[f"observation.state.{s}_gripper"] = np.array([row[f"observation.state.{s}_gripper"]], np.float32).reshape(1)
    obs["observation.state.lower_body"] = lower_body
    obs["instruction"] = instruction
    obs["unnorm_key"] = unnorm_key
    return obs


async def main_async(a) -> None:
    import websockets
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.dataset}/data/*/*.parquet"))])
    legs = np.concatenate([G1_STANDING_LEGS["left"], G1_STANDING_LEGS["right"]]).astype(np.float32)
    packer = msgpack_numpy.Packer()
    rows_out = []
    raw_dirs = sorted(p for p in Path(a.raw).glob("episode_*") if (p / "data.json").exists())
    async with websockets.connect(a.uri, max_size=None, ping_interval=None) as ws:
        await ws.recv()
        for e in a.episodes:
            ep = df[df.episode_index == e].sort_values("frame_index").reset_index(drop=True)
            # épisode e du dataset = e-ième dossier brut TRIÉ (les enregistrements réels commencent à 0001)
            ep_raw = raw_dirs[e]
            items = json.loads((ep_raw / "data.json").read_text())["data"]
            assert len(items) == len(ep), f"épisode {e} : {len(items)} JPEG pour {len(ep)} lignes"
            act = np.stack(ep[f"action.{a.side}_ee_pose_gripper_base"].to_numpy())
            grip = ep[f"action.{a.side}_gripper"].to_numpy().astype(float).reshape(-1)
            for t in range(0, len(ep) - 30, a.every):
                await ws.send(packer.pack({"type": "policy_reset"}))
                await ws.recv()
                # taille ENREGISTRÉE (tangage, lacet) : une sim neuve sans go_ready donnait [0, 0, 0] (audit du 1/10)
                waist = np.asarray(ep.iloc[t]["observation.state.waist_state_joint"], np.float32)
                obs = build_obs(ep.iloc[t], ep_raw, items[t], a.instruction, a.unnorm_key,
                                np.concatenate([legs, waist]))
                await ws.send(packer.pack({"type": "get_action", "obs": obs}))
                raw = await ws.recv()
                if isinstance(raw, str):
                    raise RuntimeError(raw)
                out = msgpack_numpy.unpackb(raw)
                pred = np.asarray(out[f"action.{a.side}_ee_rpy"])[0, :30]
                pg = np.asarray(out[f"action.{a.side}_gripper"])[0, :30, 0]
                true, tg = act[t:t + 30], grip[t:t + 30]
                err = np.linalg.norm(pred[:, :3] - true[:, :3], axis=1) * 1000
                hold = np.linalg.norm(true[:, :3] - act[t, :3], axis=1) * 1000
                r = {"ep": e, "t": t, "err_mm_mean": float(err.mean()), "err_mm_end": float(err[-1]),
                     "hold_mm_end": float(hold[-1]), "grip_err": float(np.abs(pg - tg).mean())}
                rows_out.append(r)
                print(f"ép {e} pas {t:3d} | écart position moyen {r['err_mm_mean']:5.1f} mm, au pas 30 "
                      f"{r['err_mm_end']:5.1f} mm (tenir la pose : {r['hold_mm_end']:5.1f}) | pince {r['grip_err']:.2f}",
                      flush=True)
    em = np.array([r["err_mm_end"] for r in rows_out])
    hm = np.array([r["hold_mm_end"] for r in rows_out])
    print(f"BILAN : écart au pas 30 médian {np.median(em):.1f} mm (tenir la pose : {np.median(hm):.1f} mm), "
          f"moyen {em.mean():.1f} mm, sur {len(em)} requêtes")
    if a.out:
        Path(a.out).write_text(json.dumps(rows_out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default="ws://127.0.0.1:8600")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw", required=True, help="enregistrements bruts du MÊME dataset (JPEG)")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--side", default="right")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--unnorm_key", default="UnifoLM_G1_Dex1")
    ap.add_argument("--out", default=None)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
