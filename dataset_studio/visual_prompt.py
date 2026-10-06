"""Visual prompt : dessine la SOURCE (vert) et la CIBLE (rouge) dans l'image de la caméra de tête.

Le MÊME rendu doit servir à l'entraînement (export ci-dessous) et sur le robot (suivi en direct) : c'est
la condition pour que le modèle voie au robot ce qu'il a vu dans ses données. Ne pas changer couleurs,
opacité ou contour sans réexporter les données.

Export : ``python -m dataset_studio.visual_prompt export <tâche> [--out <dossier>] [--object piece]`` crée
un dossier de tâche ``<tâche>_couleurs`` : images de tête (color_0) redessinées, autres caméras en liens
physiques, data.json copiés. Les originaux ne sont pas modifiés. Seuls les épisodes suivis sont exportés.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

SOURCE_BGR = (0, 200, 0)          # vert
TARGET_BGR = (0, 0, 220)          # rouge
ALPHA = 0.45                      # opacité du remplissage
CONTOUR = 2                       # épaisseur du contour (px)
HEAD_KEY = "color_0"              # œil gauche de la tête = l'image du modèle
JPEG_QUALITY = 95


def overlay(img_bgr: np.ndarray, source: np.ndarray | None, target: np.ndarray | None) -> np.ndarray:
    """Remplissage semi-transparent + contour, cible d'abord puis source par-dessus."""
    out = img_bgr.copy()
    for m, col in ((target, TARGET_BGR), (source, SOURCE_BGR)):
        if m is None or not m.any():
            continue
        fill = out.copy()
        fill[m] = col
        out = cv2.addWeighted(fill, ALPHA, out, 1 - ALPHA, 0)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, col, CONTOUR)
    return out


def _link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export(task: Path, out: Path, name: str = "piece") -> int:
    if out.exists():
        raise SystemExit(f"{out} existe déjà : le supprimer ou choisir --out")
    tmp = out.with_name(f".export_{out.name}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    eps = sorted(p for p in task.glob("episode_*") if (p / "objects" / f"{name}_tracks.npz").exists())
    skipped = sorted(p.name for p in task.glob("episode_*") if p not in eps)
    for i, ep in enumerate(eps):
        doc = json.loads((ep / "data.json").read_text())
        d = np.load(ep / "objects" / f"{name}_tracks.npz")
        W = int(d["shape"][-1])
        packed = d["masks"]
        dst = tmp / f"episode_{i:04d}"
        dst.mkdir()
        for sub in [p for p in ep.iterdir() if p.is_dir() and p.name != "objects"]:
            (dst / sub.name).mkdir()
            for f in sub.iterdir():
                if f.is_file() and not (sub.name == "colors" and f"_{HEAD_KEY}." in f.name):
                    _link_or_copy(f, dst / sub.name / f.name)
        for fi, st in enumerate(doc["data"]):
            p = st["colors"].get(HEAD_KEY)
            if p is None:
                continue
            img = cv2.imread(str(ep / p))
            if img is None:
                continue
            if fi < len(packed):
                m = np.unpackbits(packed[fi], axis=-1)[..., :W].astype(bool)
                img = overlay(img, m[0], m[1])
            cv2.imwrite(str(dst / p), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        doc.setdefault("info", {})["visual_prompt"] = {"source": str(ep), "object": name,
                                                       "colors": {"source": "vert", "cible": "rouge"}}
        (dst / "data.json").write_text(json.dumps(doc, ensure_ascii=False))
        if (i + 1) % 10 == 0 or i + 1 == len(eps):
            print(f"  {i + 1}/{len(eps)} épisodes exportés", flush=True)
    tmp.rename(out)
    if skipped:
        print(f"épisodes sans suivi, non exportés : {skipped}")
    return len(eps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("task")
    e.add_argument("--out")
    e.add_argument("--object", default="piece")
    a = ap.parse_args()
    task = Path(a.task).resolve()
    out = Path(a.out).resolve() if a.out else task.with_name(task.name + "_couleurs")
    n = export(task, out, a.object)
    print(f"export : {n} épisodes -> {out}")


if __name__ == "__main__":
    sys.exit(main())
