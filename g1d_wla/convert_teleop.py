"""Enregistrements xr_teleoperate (JSON + JPEG) du G1-D -> dataset LeRobot v3 au format WLA.

Produit les mêmes clés que les datasets G1 Dex1 de l'entraînement (``docs/G1D_Constats.md`` §9),
pour que la config de données WLA les lise sans adaptation :

* effecteurs ``*_ee_pose_gripper_base`` : FK des angles MESURÉS (état) et COMMANDÉS (action), dans
  la base WLA (bassin virtuel du G1), point effecteur WLA, xyz + euler ``xyz`` extrinsèque ;
* pinces : unité moteur Dex1 (0 fermée → ~5.4 ouverte), identique à l'entraînement : copiée telle
  quelle ;
* taille ``[yaw, roll, pitch]`` = taille G1 équivalente au buste G1-D (tangage, rotation : ``waist_from_torso``) ;
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

from .frames import (G1_BASE_HEIGHT, G1_STANDING_LEGS, SIDES, base_T_torso, matrix_to_xyz_rpy, waist_from_torso,
                     wrist_T_ee)
from .urdf_fk import ArmFK

log = logging.getLogger("g1d_wla.convert")

FPS = 30
IMG_SHAPE = (480, 640, 3)
XYZ_RPY = ["x", "y", "z", "roll", "pitch", "yaw"]
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
LEG = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]

#: images xr_teleoperate -> clés des datasets G1 Dex1. La téléop numérote les vues dans l'ordre
#: tête (1 ou 2 yeux) puis poignets ACTIFS : le nombre d'images ne suffit pas à trouver la disposition
#: (tête monoculaire de la sim et --right-only avec tête binoculaire donnent tous deux 3 images).
IMAGE_LAYOUTS = {
    "binocular": {"color_0": "observation.images.head_stereo_left",
                  "color_1": "observation.images.head_stereo_right",
                  "color_2": "observation.images.wrist_left",
                  "color_3": "observation.images.wrist_right"},
    "mono": {"color_0": "observation.images.head_stereo_left",          # sim : œil gauche seul
             "color_1": "observation.images.wrist_left",
             "color_2": "observation.images.wrist_right"},
    "right-only": {"color_0": "observation.images.head_stereo_left",    # téléop --right-only
                   "color_1": "observation.images.head_stereo_right",
                   "color_2": "observation.images.wrist_right",
                   None: "observation.images.wrist_left"},              # poignet gauche absent : image noire
}
N_IMAGES = {"binocular": 4, "mono": 3, "right-only": 3}


def episode_layout(doc: dict, forced: str) -> str:
    """Disposition des images d'un épisode : forcée, sinon d'après l'en-tête (sim) et le nombre."""
    n = len(doc["data"][0]["colors"])
    if forced != "auto":
        lay = forced
    elif n == 4:
        lay = "binocular"
    elif n == 3 and str(doc.get("info", {}).get("source", "")).startswith("sim/"):
        lay = "mono"
    else:
        raise ValueError(f"{n} images par pas sans en-tête de sim : préciser --layout "
                         f"(right-only pour un enregistrement --right-only)")
    if n != N_IMAGES[lay]:
        raise ValueError(f"disposition {lay} : {N_IMAGES[lay]} images attendues, {n} trouvées")
    return lay


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
    return {"text": d.get("text", {}), "steps": d["data"], "info": d.get("info", {})}


def convert_episode(ep: dict, fk: dict, torso_pitch_index: int | None, torso_pitch_const: float | None,
                    base_height: float, torso_yaw_index: int | None = None) -> dict:
    """Un épisode -> dict de tableaux (T, …) aux clés WLA (sans les images)."""
    steps = ep["steps"]
    T = len(steps)
    out = {}
    has_body = [bool(s["states"].get("body", {}).get("qpos")) for s in steps]
    if any(has_body) and not all(has_body):
        raise ValueError(f"body.qpos présent sur {sum(has_body)}/{T} pas seulement : enregistrement incohérent")
    body = np.array([s["states"]["body"]["qpos"] for s in steps], float) if all(has_body) else None
    # info.body_layout (écrit par l'enregistreur) prioritaire, sauf ses valeurs None (audit du 1/10)
    layout = {k: v for k, v in (ep.get("info", {}).get("body_layout") or {}).items() if v is not None}
    if torso_pitch_const is None and "torso_pitch_const" in layout:
        torso_pitch_const = float(layout["torso_pitch_const"])   # tangage CONSTANT utilisé à l'enregistrement
    torso_pitch_index = layout.get("torso_pitch", torso_pitch_index)
    torso_yaw_index = layout.get("torso_yaw", torso_yaw_index)
    yaw = body[:, torso_yaw_index] if body is not None and torso_yaw_index is not None else np.zeros(T)
    if torso_pitch_const is not None:
        pitch = np.full(T, torso_pitch_const)
    elif body is not None and torso_pitch_index is not None:
        pitch = body[:, torso_pitch_index]
    else:
        raise ValueError("tangage du buste introuvable : body.qpos vide, passer --torso-pitch")
    B_torso = np.stack([base_T_torso(p, y) for p, y in zip(pitch, yaw)])     # (T, 4, 4)

    for kind, key in (("observation.state", "states"), ("action", "actions")):
        for s in SIDES:
            q = np.array([st[key][f"{s}_arm"]["qpos"] for st in steps], float)   # (T, 7)
            ee = B_torso @ fk[s].fk(q) @ wrist_T_ee(s)
            out[f"{kind}.{s}_arm"] = q
            out[f"{kind}.{s}_ee_pose_gripper_base"] = matrix_to_xyz_rpy(ee)
            out[f"{kind}.{s}_gripper"] = np.array([st[key][f"{s}_ee"]["qpos"][:1] for st in steps], float)
            out[f"{kind}.{s}_leg"] = np.tile(G1_STANDING_LEGS[s], (T, 1))
    waist = waist_from_torso(pitch, yaw)                                     # [yaw, roll, pitch] G1
    out["observation.state.waist_state_joint"] = waist
    out["action.waist_action_joint"] = waist
    out["observation.state.state_torso"] = matrix_to_xyz_rpy(B_torso)
    vel = np.array([(st["actions"]["body"]["qpos"] or [0.0, 0.0, 0.0])[:3] for st in steps], float)
    # hauteur : G1-D avec colonne enregistrée (téléop --column) -> hauteur de bassin G1 équivalente =
    # base_height + hauteur au-dessus de la butée basse. HYPOTHÈSE : colonne en butée basse ≈ G1 debout
    # (le G1 ne fait que s'accroupir : au-delà de ~+6 cm, hors des données d'entraînement, q99 = 0,794)
    if all("column" in st["states"] for st in steps):
        col = np.array([st["states"]["column"]["qpos"][0] for st in steps], float)
        height = base_height + col
    else:
        height = np.full(T, base_height)
    out["action.base_command"] = np.concatenate([vel, height[:, None]], axis=1)
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
    ap.add_argument("--torso-yaw-index", type=int, default=None,
                    help="indice de la rotation du buste dans body.qpos ; défaut : info.body_layout de l'épisode, "
                         "sinon buste non tourné")
    ap.add_argument("--base-height", type=float, default=G1_BASE_HEIGHT,
                    help="hauteur de bassin G1 équivalente pour action.base_command (défaut : médiane G1)")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="indices d'épisodes à convertir")
    ap.add_argument("--vcodec", default="libsvtav1")
    ap.add_argument("--layout", choices=["auto", "binocular", "mono", "right-only"], default="auto",
                    help="disposition des images : auto = 4 images -> binoculaire, 3 images d'un épisode de sim "
                         "-> mono ; right-only obligatoire pour la téléop --right-only")
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
    if args.torso_pitch is None and not all(
            (json.loads((p / "data.json").read_text()).get("info", {}).get("body_layout") or {}).get("torso_pitch_const")
            is not None for p in ep_dirs):
        log.warning("tangage du buste lu à l'indice %d de body.qpos (pas de tangage constant enregistré) : "
                    "sur le G1-D, le tangage n'est PAS dans les 35 moteurs -> passer --torso-pitch", args.torso_pitch_index)

    fk = {s: ArmFK(s) for s in SIDES}
    # disposition vérifiée sur TOUS les épisodes avant d'écrire quoi que ce soit
    layouts = {p.name: episode_layout(json.loads((p / "data.json").read_text()), args.layout) for p in ep_dirs}
    kinds = sorted(set(layouts.values()))
    image_keys = {v: v for v in IMAGE_LAYOUTS["binocular"].values()}   # mêmes clés de sortie pour toutes
    if "mono" in kinds:
        image_keys = {v: v for v in IMAGE_LAYOUTS["mono"].values()}
        if len(kinds) > 1:
            raise SystemExit(f"dispositions mélangées {kinds} : convertir séparément la sim et le réel")
    log.info("disposition des images : %s -> %s", kinds, list(image_keys.values()))
    with_adv = args.advantage == "on" or (
        args.advantage == "auto" and any("advantage" in load_episode(p)["steps"][0] for p in ep_dirs))
    if with_adv:
        log.info("colonnes advantage / intervention écrites (défaut %.1f)", args.default_advantage)
        unlabeled = [p.name for p in ep_dirs if "advantage" not in load_episode(p)["steps"][0]]
        if unlabeled:      # un lot oublié à l'étiquetage deviendrait entièrement « positif » (audit du 1/10)
            log.warning("%d épisode(s) SANS étiquette d'avantage -> %.1f partout : %s", len(unlabeled),
                        args.default_advantage, ", ".join(unlabeled[:10]) + (" ..." if len(unlabeled) > 10 else ""))
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=build_features(
                                   {k: k for k in image_keys.values()}, with_adv), root=args.out_dir,
                               robot_type="unitree_g1d", use_videos=True, vcodec=args.vcodec,
                               image_writer_threads=4)
    for n, ep_dir in enumerate(ep_dirs):
        ep = load_episode(ep_dir)
        task = args.task or ep["text"].get("goal") or "manipulation"
        arrays = convert_episode(ep, fk, args.torso_pitch_index, args.torso_pitch, args.base_height,
                                 args.torso_yaw_index)
        for t, st in enumerate(ep["steps"]):
            frame = {k: v[t] for k, v in arrays.items()}
            for ck, fk_name in IMAGE_LAYOUTS[layouts[ep_dir.name]].items():
                if ck is None:
                    frame[fk_name] = np.zeros(IMG_SHAPE, np.uint8)
                    continue
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
