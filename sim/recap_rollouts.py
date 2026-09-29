"""Rollouts RECAP en sim : la politique WLA joue seule, un OPÉRATEUR SIMULÉ corrige quand elle échoue.

Boucle « à la RECAP / Delta-0 », étape 2 : le robot essaie, un humain intervient. Chaque épisode est
enregistré au format xr_teleoperate (``SimEpisodeWriter``) avec, par pas, ``intervention`` = 0 (action
de la politique) ou 1 (correction de l'opérateur) ; l'en-tête porte ``success_step``, ``outcome``,
``takeover_step``. Les épisodes passent ensuite par le modèle de valeur (``g1d_wla.recap``) qui étiquette
l'avantage de chaque morceau d'action, puis par le convertisseur.

Opérateur simulé (sim seulement : il voit la vraie pose de l'objet) : il prend la main
* si l'objet a été poussé de plus de ``--knock-cm`` sans être soulevé, ou
* si la tâche n'est pas réussie au pas ``--takeover-step``.
Il rejoue alors l'expert scripté DEPUIS L'ÉTAT COURANT, jusqu'à la réussite. Sur le robot, ce rôle est
tenu par l'opérateur en téléop (correction en delta, voir docs/G1D_Constats.md §15).

Usage (env de la sim ; serveur WLA lancé sur le checkpoint à améliorer) ::

    MUJOCO_GL=egl python sim/recap_rollouts.py --task novares --episodes 40 \\
        --out playground/sim_raw/recap_novares_r1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "model_server"))
from g1d_sim import SIDES, G1DSim  # noqa: E402
from sim_episode_writer import SimEpisodeWriter  # noqa: E402
from sim_tasks import get_task  # noqa: E402
from tools import msgpack_numpy  # noqa: E402
from wla_client import build_obs, dex1_to_closure, xyz_rpy_to_matrix  # noqa: E402

TABLE_DROP = 0.05      # objet tombé de la table : plus de correction possible


async def run_episode(ws, packer, sim: G1DSim, task, rng, args, w: SimEpisodeWriter) -> dict:
    task.place(sim, rng)
    z0 = sim.object_pose(task.body)[2, 3]
    xy0 = sim.object_pose(task.body)[:2, 3].copy()
    await ws.send(packer.pack({"type": "policy_reset"}))
    await ws.recv()
    t, success_step, takeover_step, prev_exec = 0, None, None, None

    def lifted() -> float:
        return float(sim.object_pose(task.body)[2, 3] - z0)

    def check(tt: int) -> None:
        nonlocal success_step
        if success_step is None and lifted() > task.lift_success:
            success_step = tt

    # 1. la politique joue
    while t < args.max_steps and success_step is None:
        obs, _ = build_obs(sim, args.instruction, args.unnorm_key)
        if args.advantage:
            obs["advantage"] = args.advantage
        if args.rtc_prefix > 0 and prev_exec is not None:
            obs["rtc_executed"], obs["rtc_prefix"] = prev_exec, args.rtc_prefix
            if args.rtc_soft:
                obs["rtc_soft"] = args.rtc_soft
        await ws.send(packer.pack({"type": "get_action", "obs": obs}))
        raw = await ws.recv()
        if isinstance(raw, str):
            raise RuntimeError(f"erreur serveur :\n{raw}")
        act = msgpack_numpy.unpackb(raw)
        chunk = {s: np.asarray(act[f"action.{s}_ee_rpy"])[0] for s in SIDES}
        grip = {s: np.asarray(act[f"action.{s}_gripper"])[0, :, 0] for s in SIDES}
        n = min(args.exec_steps, len(chunk["left"]))
        knocked = False
        for k in range(n):
            closure = {}
            for s in SIDES:
                sim.track_ee_wla(s, xyz_rpy_to_matrix(chunk[s][k]))
                closure[s] = dex1_to_closure(grip[s][k])
                sim.set_gripper(s, closure[s])
            w.record(t, closure, {"intervention": 0})
            sim.step()
            check(t)
            t += 1
            moved = float(np.linalg.norm(sim.object_pose(task.body)[:2, 3] - xy0))
            if success_step is None and moved > args.knock_cm / 100 and lifted() < 0.02:
                knocked = True
                break
            if success_step is not None or t >= args.max_steps:
                break
        prev_exec = n
        if knocked or (success_step is None and t >= args.takeover_step):
            break

    # 2. l'opérateur corrige si besoin (et si l'objet est encore sur la table)
    if success_step is None and args.operator and lifted() > -TABLE_DROP:
        takeover_step = t
        t0 = t

        def on_step(k: int, closure_cmd: dict) -> None:
            w.record(t0 + k, closure_cmd, {"intervention": 1})
            check(t0 + k)

        z0_before = z0
        res = task.expert(sim, rng, on_step=on_step)
        t = t0 + int(res.get("steps", 0))
        z0 = z0_before

    # 3. quelques pas de tenue après la réussite (fin d'épisode propre)
    outcome = "success" if success_step is not None and lifted() > task.lift_success else "failure"
    return {"success_step": success_step, "takeover_step": takeover_step, "outcome": outcome,
            "steps": t, "lifted": round(lifted(), 4)}


async def main_async(args) -> None:
    import websockets

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} existe déjà (--overwrite)")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    task = get_task(args.task)
    sim = G1DSim(scene_xml=task.scene_xml, head_view=args.head_view)
    rng = np.random.default_rng(args.seed)
    packer = msgpack_numpy.Packer()
    summary, t_start = [], time.time()
    async with websockets.connect(args.uri, max_size=None, ping_interval=None) as ws:
        meta = msgpack_numpy.unpackb(await ws.recv())
        print(f"serveur : {meta.get('ckpt_path')}", flush=True)
        for e in range(args.episodes):
            sim.reset()
            sim.go_ready()
            w = SimEpisodeWriter(sim, out / f"episode_{e:04d}", args.jpeg_quality)
            r = await run_episode(ws, packer, sim, task, rng, args, w)
            w.save(args.instruction, f"rollout RECAP, sim MuJoCo G1-D, tâche {task.name}", "sim/recap_rollouts.py",
                   {k: r[k] for k in ("success_step", "takeover_step", "outcome")})
            summary.append(r)
            who = "politique" if r["takeover_step"] is None else f"opérateur au pas {r['takeover_step']}"
            print(f"épisode {e}: {r['outcome']} ({who}), {r['steps']} pas, réussite au pas {r['success_step']}"
                  f" | {time.time() - t_start:.0f} s", flush=True)
    n_pol = sum(r["outcome"] == "success" and r["takeover_step"] is None for r in summary)
    n_op = sum(r["takeover_step"] is not None for r in summary)
    (out / "summary.json").write_text(json.dumps({"args": vars(args), "episodes": summary}, indent=1))
    print(f"réussites de la politique seule {n_pol}/{len(summary)} | corrections de l'opérateur {n_op}")
    sim.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default="ws://127.0.0.1:8600")
    ap.add_argument("--task", choices=["cube", "novares", "novares_shift"], default="novares")
    ap.add_argument("--instruction", default=None, help="défaut : instruction de la tâche")
    ap.add_argument("--unnorm_key", default="UnifoLM_G1_Dex1")
    ap.add_argument("--advantage", default=None, help="condition envoyée au serveur (modèle RECAP)")
    ap.add_argument("--head-view", dest="head_view", choices=["rec", "raw"], default="raw")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=300)
    ap.add_argument("--takeover-step", dest="takeover_step", type=int, default=220,
                    help="l'opérateur prend la main si pas de réussite à ce pas")
    ap.add_argument("--knock-cm", dest="knock_cm", type=float, default=3.0)
    ap.add_argument("--no-operator", dest="operator", action="store_false",
                    help="pas de corrections (rollouts purement autonomes)")
    ap.add_argument("--exec-steps", dest="exec_steps", type=int, default=30)
    ap.add_argument("--rtc-prefix", dest="rtc_prefix", type=int, default=0)
    ap.add_argument("--rtc-soft", dest="rtc_soft", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.instruction is None:
        args.instruction = get_task(args.task).instruction
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
