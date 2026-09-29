"""Enregistrements xr_teleoperate (JSON + JPEG) du G1-D -> dataset LeRobot v3 au format WLA.

Produit les mêmes clés que les datasets G1 Dex1 de l'entraînement (``docs/G1D_Constats.md`` §9),
pour que la config de données WLA les lise sans adaptation :

* effecteurs ``*_ee_pose_gripper_base`` : FK des angles MESURÉS (état) et COMMANDÉS (action), dans
  la base WLA (bassin virtuel du G1), point effecteur WLA, xyz + euler ``xyz`` extrinsèque ;
* pinces : unité moteur Dex1 (0 fermée → ~5.4 ouverte), identique à l'entraînement : copiée telle
  quelle ;
* taille ``[yaw, roll, pitch]`` = [0, 0, tangage du buste] ; jambes = G1 debout (slots valides) ;
* ``action.base_command`` = [vx, vy, vyaw, hauteur de bassin équivalente] ;
* images : œil gauche BRUT -> ``head_stereo_left`` (+ œil droit), poignets -> ``wrist_left/right``.

Usage (env uv du projet, depuis la racine du dépôt) ::

    .venv/bin/python -m g1d_wla.convert_teleop \\
        --raw-dir teleoperation/Tele_OP/xr_teleoperate/teleop/utils/data/mon_test \\
        --out-dir playground/Datasets/g1d/mon_test --repo-id innov8/g1d_mon_test

⚠ Hypothèses à valider sur le robot (TODO de MYREADME.md) : l'indice moteur du tangage du buste
(``--torso-pitch-index``) et la hauteur de bassin équivalente (``--base-height``).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .frames import (G1_BASE_HEIGHT, G1_STANDING_LEGS, SIDES, base_T_torso, matrix_to_xyz_rpy,
                     wrist_T_ee)
from .urdf_fk import ArmFK

log = logging.getLogger("g1d_wla.convert")

FPS = 30
IMG_SHAPE = (480, 640, 3)
XYZ_RPY = ["x", "y", "z", "roll", "pitch", "yaw"]
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
LEG = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]

#: images xr_teleoperate -> clés des datasets G1 Dex1. La téléop numérote les vues dans l'ordre
#: tête (1 ou 2 yeux) puis poignets : 4 images = tête binoculaire, 3 = tête monoculaire (sim).
IMAGE_LAYOUTS = {
    4: {"color_0": "observation.images.head_stereo_left",
        "color_1": "observation.images.head_stereo_right",
        "color_2": "observation.images.wrist_left",
        "color_3": "observation.images.wrist_right"},
    3: {"color_0": "observation.images.head_stereo_left",
        "color_1": "observation.images.wrist_left",
        "color_2": "observation.images.wrist_right"},
}


def _vec(n: int, names: list[str]) -> dict:
    return {"dtype": "float32", "shape": (n,), "names": names}


def build_features(image_keys: dict, with_advantage: bool = False) -> dict:
    img = {"dtype": "video", "shape": IMG_SHAPE, "names": ["height", "width", "channels"]}
    f = {k: dict(img) for k in image_keys.values()}
    if with_advantage:        # RECAP : 1 = bon morceau d'action, 0 = mauvais ; 1 = pas corrigé par l'opérateur
        f["advantage"] = _vec(1, ["advantage"])
        f["intervention"] = _vec(1, ["intervention"])
    for kind in ("observation.state", "action"):
        for s in SIDES:
            f[f"{kind}.{s}_arm"] = _vec(7, ARM)
            f[f"{kind}.{s}_ee_pose_gripper_base"] = _vec(6, XYZ_RPY)
            f[f"{kind}.{s}_gripper"] = _vec(1, ["gripper_pos"])
            f[f"{kind}.{s}_leg"] = _vec(6, LEG)
    f["observation.state.waist_state_joint"] = _vec(3, ["yaw", "roll", "pitch"])
    f["observation.state.state_torso"] = _vec(6, XYZ_RPY)
    f["action.waist_action_joint"] = _vec(3, ["yaw", "roll", "pitch"])
    f["action.base_command"] = _vec(4, ["vx", "vy", "angle_z", "height"])
    return f


def load_episode(ep_dir: Path) -> dict:
    d = json.loads((ep_dir / "data.json").read_text())
    return {"text": d.get("text", {}), "steps": d["data"]}


def convert_episode(ep: dict, fk: dict, torso_pitch_index: int | None, torso_pitch_const: float | None,
                    base_height: float) -> dict:
    """Un épisode -> dict de tableaux (T, …) aux clés WLA (sans les images)."""
    steps = ep["steps"]
    T = len(steps)
    out = {}
    body = np.array([s["states"]["body"]["qpos"] for s in steps], float) if steps[0]["states"]["body"]["qpos"] else None
    if torso_pitch_const is not None:
        pitch = np.full(T, torso_pitch_const)
    elif body is not None and torso_pitch_index is not None:
        pitch = body[:, torso_pitch_index]
    else:
        raise ValueError("tangage du buste introuvable : body.qpos vide, passer --torso-pitch")
    B_torso = np.stack([base_T_torso(p) for p in pitch])                     # (T, 4, 4)

    for kind, key in (("observation.state", "states"), ("action", "actions")):
        for s in SIDES:
            q = np.array([st[key][f"{s}_arm"]["qpos"] for st in steps], float)   # (T, 7)
            ee = B_torso @ fk[s].fk(q) @ wrist_T_ee(s)
            out[f"{kind}.{s}_arm"] = q
            out[f"{kind}.{s}_ee_pose_gripper_base"] = matrix_to_xyz_rpy(ee)
            out[f"{kind}.{s}_gripper"] = np.array([st[key][f"{s}_ee"]["qpos"][:1] for st in steps], float)
            out[f"{kind}.{s}_leg"] = np.tile(G1_STANDING_LEGS[s], (T, 1))
    waist = np.stack([np.zeros(T), np.zeros(T), pitch], axis=1)
    out["observation.state.waist_state_joint"] = waist
    out["action.waist_action_joint"] = waist
    out["observation.state.state_torso"] = matrix_to_xyz_rpy(B_torso)
    vel = np.array([(st["actions"]["body"]["qpos"] or [0.0, 0.0, 0.0])[:3] for st in steps], float)
    out["action.base_command"] = np.concatenate([vel, np.full((T, 1), base_height)], axis=1)
    return {k: v.astype(np.float32) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, type=Path, help="dossier <task>/ contenant episode_XXXX/")
    ap.add_argument("--out-dir", required=True, type=Path, help="racine du dataset LeRobot à créer")
    ap.add_argument("--repo-id", required=True, help="identifiant du dataset, ex. innov8/g1d_mon_test")
    ap.add_argument("--task", default=None, help="instruction ; défaut = text.goal de chaque épisode")
    ap.add_argument("--torso-pitch-index", type=int, default=13,
                    help="indice du tangage du buste dans states.body.qpos (35 moteurs). HYPOTHÈSE : 13, à vérifier")
    ap.add_argument("--torso-pitch", type=float, default=None, help="tangage constant (rad), remplace l'indice")
    ap.add_argument("--base-height", type=float, default=G1_BASE_HEIGHT,
                    help="hauteur de bassin G1 équivalente pour action.base_command (défaut : médiane G1)")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="indices d'épisodes à convertir")
    ap.add_argument("--vcodec", default="libsvtav1")
    ap.add_argument("--advantage", choices=["auto", "on", "off"], default="auto",
                    help="colonnes advantage/intervention (RECAP) : auto = si un épisode porte step['advantage'] ; "
                         "les pas sans étiquette valent --default-advantage")
    ap.add_argument("--default-advantage", dest="default_advantage", type=float, default=1.0,
                    help="avantage des pas non étiquetés (1 : démos expertes = bonnes actions)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # import tardif : lourd

    ep_dirs = sorted(p for p in args.raw_dir.glob("episode_*") if (p / "data.json").exists())
    if args.episodes is not None:
        keep = {f"episode_{i:04d}" for i in args.episodes}
        ep_dirs = [p for p in ep_dirs if p.name in keep]
    if not ep_dirs:
        raise SystemExit(f"aucun épisode dans {args.raw_dir}")
    if args.out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out_dir} existe déjà (--overwrite pour le remplacer)")
        shutil.rmtree(args.out_dir)
    if args.torso_pitch is None:
        log.warning("tangage du buste lu à l'indice %d de body.qpos : HYPOTHÈSE à valider sur le robot",
                    args.torso_pitch_index)

    fk = {s: ArmFK(s) for s in SIDES}
    n_img = len(load_episode(ep_dirs[0])["steps"][0]["colors"])
    if n_img not in IMAGE_LAYOUTS:
        raise SystemExit(f"{n_img} images par pas : dispositions connues {sorted(IMAGE_LAYOUTS)}")
    image_keys = IMAGE_LAYOUTS[n_img]
    log.info("%d images par pas : %s", n_img, list(image_keys.values()))
    with_adv = args.advantage == "on" or (
        args.advantage == "auto" and any("advantage" in load_episode(p)["steps"][0] for p in ep_dirs))
    if with_adv:
        log.info("colonnes advantage / intervention écrites (défaut %.1f)", args.default_advantage)
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=build_features(image_keys, with_adv), root=args.out_dir,
                               robot_type="unitree_g1d", use_videos=True, vcodec=args.vcodec,
                               image_writer_threads=4)
    for n, ep_dir in enumerate(ep_dirs):
        ep = load_episode(ep_dir)
        task = args.task or ep["text"].get("goal") or "manipulation"
        arrays = convert_episode(ep, fk, args.torso_pitch_index, args.torso_pitch, args.base_height)
        for t, st in enumerate(ep["steps"]):
            frame = {k: v[t] for k, v in arrays.items()}
            for ck, fk_name in image_keys.items():
                path = st["colors"].get(ck)
                if path is None:
                    raise ValueError(f"{ep_dir.name} pas {t} : image {ck} absente")
                img = np.asarray(Image.open(ep_dir / path).convert("RGB"))
                if img.shape != IMG_SHAPE:
                    raise ValueError(f"{ep_dir.name} {ck} : {img.shape}, attendu {IMG_SHAPE}")
                frame[fk_name] = img
            if with_adv:
                frame["advantage"] = np.array([st.get("advantage", args.default_advantage)], np.float32)
                frame["intervention"] = np.array([float(st.get("intervention", 0))], np.float32)
            frame["task"] = task
            ds.add_frame(frame)
        ds.save_episode()
        log.info("épisode %d/%d : %s, %d pas, tâche %r", n + 1, len(ep_dirs), ep_dir.name, len(ep["steps"]), task)
    ds.finalize()
    log.info("dataset écrit dans %s", args.out_dir)


if __name__ == "__main__":
    main()
