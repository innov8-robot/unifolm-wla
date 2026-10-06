"""Calculs de l'onglet « Objets » du Dataset Studio (OPTIONNEL) : signature DINOv3, détection des
instances sans texte, suivi SAM2 des deux masques choisis (source, cible), aperçu vidéo.

Tourne dans un environnement qui a torch, transformers (DINOv3) et sam2 — par défaut
``unitree_lerobot`` — lancé par le studio (``QProcess``) ou à la main. Le studio n'importe jamais ce
module : sans cet environnement, il marche comme avant.

Déroulé :

1. l'opérateur ENCADRE l'objet sur quelques images -> ``objects/objects.json`` ;
2. ``signature`` : plongements DINOv3 des découpes encadrées (pos) et de fonds (neg) ;
3. ``detect`` : sur une image choisie (main loin des pièces), SAM2 « tout segmenter » propose tous les
   segments, la signature garde ceux qui ressemblent à l'objet (6 pièces identiques = 6 masques) ;
4. l'opérateur choisit dans le studio le masque SOURCE (vert) et le masque CIBLE (rouge), les autres
   sont écartés -> ``episode_XXXX/objects/<nom>_selection.json`` (un choix par image de départ) ;
5. ``track`` : seuls ces deux masques sont suivis, de leur image jusqu'au choix suivant ;
6. ``preview`` : vidéo de contrôle.

Reconnaissance par DÉCOUPES : chaque candidat est découpé, agrandi à 224 px et plongé par DINOv3. (La
carte de chaleur dense sur l'image entière échoue sur la pièce Novares, trop petite : mesuré dans
mpc_any, perception/signature.py.)

Suivi CAUSAL (propagation SAM2 vers l'avant seulement) : mêmes défauts que le suivi en direct sur le
robot, donc mêmes entrées à l'entraînement et à l'exécution.

Fichiers (dossier de tâche) ::

    objects/objects.json                              {"objects": [{"name", "examples": [{episode, frame, cam, box}]}]}
    objects/<nom>_signature.npz                       pos, neg, loo
    episode_XXXX/objects/<nom>_candidates_<image>.npz masques candidats (bits), scores, image
    episode_XXXX/objects/<nom>_selection.json         {"choices": [{frame, source, target}]} (indices de candidats)
    episode_XXXX/objects/<nom>_tracks.npz             masques suivis (T, 2, H, W) : 0 = source, 1 = cible

Usage ::

    python -m dataset_studio.objects_worker signature --task <dossier> --object piece
    python -m dataset_studio.objects_worker detect --task <dossier> --object piece --episodes episode_0000 --frame 0
    python -m dataset_studio.objects_worker track --task <dossier> --object piece --episodes episode_0000
    python -m dataset_studio.objects_worker preview --task <dossier> --object piece --episode episode_0000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

CKPT = Path(os.environ.get("STUDIO_OBJECTS_CKPT", "/home/thomas/Documents/project/MPC/mpc_any/checkpoints"))
HEAD_CAM = "color_0"
CROP = 224
COLORS_BGR = {"source": (60, 200, 60), "cible": (60, 60, 230)}


def say(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ fichiers
def objects_file(task: Path) -> Path:
    return task / "objects" / "objects.json"


def load_objects(task: Path) -> dict:
    f = objects_file(task)
    return json.loads(f.read_text()) if f.exists() else {"objects": []}


def object_entry(task: Path, name: str) -> dict:
    for o in load_objects(task)["objects"]:
        if o["name"] == name:
            return o
    raise SystemExit(f"objet « {name} » absent de {objects_file(task)}")


def frame_path(task: Path, episode: str, frame: int, cam: str = HEAD_CAM) -> Path:
    doc = json.loads((task / episode / "data.json").read_text())
    return task / episode / doc["data"][frame]["colors"][cam]


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


# ------------------------------------------------------------------ DINOv3 sur découpes
class CropEmbedder:
    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        p = CKPT / "dinov3-vits16"
        self.torch = torch
        self.model = AutoModel.from_pretrained(p).to(device).eval()
        self.proc = AutoImageProcessor.from_pretrained(p)
        self.device = device

    @staticmethod
    def crop(rgb: np.ndarray, box, grow: float = 1.4) -> np.ndarray:
        """Découpe CARRÉE autour de la boîte (agrandie de ``grow``), redimensionnée à CROP px."""
        from PIL import Image
        x0, y0, x1, y1 = [float(v) for v in box]
        cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0, 8) * grow / 2
        h, w = rgb.shape[:2]
        a, b = int(max(0, cx - s)), int(max(0, cy - s))
        c, d = int(min(w, cx + s)), int(min(h, cy + s))
        return np.asarray(Image.fromarray(rgb[b:d, a:c]).resize((CROP, CROP)))

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 384), np.float32)
        out = []
        for i in range(0, len(crops), 64):
            x = self.proc(images=crops[i:i + 64], return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                o = self.model(**x)
            f = self.torch.cat([o.pooler_output, o.last_hidden_state[:, 1:].mean(1)], 1)   # CLS + moyenne
            out.append(self.torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def score(emb: np.ndarray, sig: dict) -> np.ndarray:
    """Ressemblance aux exemples moins ressemblance aux fonds (top-3 moyens)."""
    def topk(a, b, k=3):
        s = a @ b.T
        k = min(k, s.shape[1])
        return np.sort(s, 1)[:, -k:].mean(1)
    return topk(emb, sig["pos"]) - topk(emb, sig["neg"])


def box_of(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None


def iou_box(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


# ------------------------------------------------------------------ signature
def cmd_signature(a) -> None:
    task = Path(a.task)
    obj = object_entry(task, a.object)
    ex = obj.get("examples", [])
    if len(ex) < 3:
        raise SystemExit(f"il faut au moins 3 exemples encadrés (il y en a {len(ex)})")
    emb = CropEmbedder()
    rng = random.Random(0)
    pos_crops, neg_crops = [], []
    for e in ex:
        rgb = read_rgb(frame_path(task, e["episode"], e["frame"], e.get("cam", HEAD_CAM)))
        pos_crops.append(emb.crop(rgb, e["box"]))
        bw, bh = e["box"][2] - e["box"][0], e["box"][3] - e["box"][1]
        h, w = rgb.shape[:2]
        for _ in range(12):                                      # fonds : même taille, ailleurs
            x0, y0 = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
            nb = (x0, y0, x0 + bw, y0 + bh)
            if all(iou_box(nb, o["box"]) < 0.05 for o in ex if o["episode"] == e["episode"] and o["frame"] == e["frame"]):
                neg_crops.append(emb.crop(rgb, nb))
    pos, neg = emb.embed(pos_crops), emb.embed(neg_crops)
    sig = {"pos": pos, "neg": neg}
    # validation : chaque exemple contre la signature construite SANS lui
    loo = []
    for i in range(len(pos)):
        s = {"pos": np.delete(pos, i, 0), "neg": neg}
        loo.append(float(score(pos[i:i + 1], s)[0]))
    neg_s = score(neg, sig)
    out = task / "objects" / f"{a.object}_signature.npz"
    np.savez_compressed(out, pos=pos, neg=neg, loo=np.array(loo))
    say(f"signature « {a.object} » : {len(pos)} exemples, {len(neg)} fonds -> {out}")
    say(f"  validation (exemple retiré) : score min {min(loo):+.3f}, médian {np.median(loo):+.3f} ; "
        f"fonds : max {neg_s.max():+.3f}, médian {np.median(neg_s):+.3f}")
    say("  (un exemple au score < 0 ressemble plus au fond qu'aux autres exemples : ajouter des vues variées)")


# ------------------------------------------------------------------ détection + suivi + rôles
def cmd_detect(a) -> None:
    """Candidats SANS TEXTE sur l'image ``--frame`` : SAM2 « tout segmenter », puis tri par la signature
    DINOv3 (découpes). Écrit ``<nom>_candidates_<image>.npz`` : l'opérateur y choisit source et cible."""
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    task = Path(a.task)
    sigf = task / "objects" / f"{a.object}_signature.npz"
    if not sigf.exists():
        raise SystemExit("signature absente : lancer d'abord « signature »")
    sig = dict(np.load(sigf))
    eps = a.episodes or sorted(p.name for p in task.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    emb = CropEmbedder()
    amg = SAM2AutomaticMaskGenerator(
        build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(CKPT / "sam2.1_hiera_large.pt"), device="cuda"),
        points_per_side=32, pred_iou_thresh=0.7, stability_score_thresh=0.85, min_mask_region_area=30)
    for n_ep, ep in enumerate(eps):
        doc = json.loads((task / ep / "data.json").read_text())
        f0 = min(a.frame, len(doc["data"]) - 1)
        rgb = read_rgb(task / ep / doc["data"][f0]["colors"][HEAD_CAM])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            props = amg.generate(rgb)
        cand = [p["segmentation"] for p in props if 30 < p["segmentation"].sum() < 0.15 * rgb.shape[0] * rgb.shape[1]]
        boxes = [box_of(m) for m in cand]
        sc = score(emb.embed([emb.crop(rgb, b) for b in boxes]), sig) if cand else np.zeros(0)
        order = [i for i in np.argsort(-sc) if sc[i] > a.threshold]
        keep = []                                   # sans doublon : SAM2 rend parfois la pièce ET une partie
        for i in order:
            if all(iou_box(boxes[i], boxes[j]) < 0.5 for j in keep):
                keep.append(i)
        keep = keep[: a.max_instances]
        od = task / ep / "objects"
        od.mkdir(exist_ok=True)
        masks = np.stack([cand[i] for i in keep]) if keep else np.zeros((0,) + rgb.shape[:2], bool)
        np.savez_compressed(od / f"{a.object}_candidates_{f0:05d}.npz", masks=np.packbits(masks, axis=-1),
                            shape=np.array(masks.shape), scores=sc[keep] if keep else np.zeros(0), frame=f0)
        say(f"[{n_ep + 1}/{len(eps)}] {ep} image {f0} : {len(props)} segments SAM2, {len(keep)} « {a.object} » "
            f"(scores {np.round(sc[keep], 2).tolist() if keep else []})")


def unpack(d) -> np.ndarray:
    shp = tuple(d["shape"])
    return np.unpackbits(d["masks"], axis=-1)[..., : shp[-1]].astype(bool).reshape(shp)


def cmd_track(a) -> None:
    """Suit SEULEMENT les masques choisis (``<nom>_selection.json``) : pour chaque choix, de son image
    jusqu'au choix suivant (ou la fin), propagation SAM2 vers l'avant. Écrit ``<nom>_tracks.npz`` :
    masques (T, 2, H, W), canal 0 = source, 1 = cible ; vide hors des plages choisies."""
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    task = Path(a.task)
    pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", str(CKPT / "sam2.1_hiera_large.pt"),
                                      device="cuda")
    eps = a.episodes or sorted(p.name for p in task.iterdir()
                               if (p / "objects" / f"{a.object}_selection.json").exists())
    for n_ep, ep in enumerate(eps):
        od = task / ep / "objects"
        sf = od / f"{a.object}_selection.json"
        if not sf.exists():
            say(f"[{n_ep + 1}/{len(eps)}] {ep} : aucun choix source / cible enregistré — épisode passé "
                f"(détecter, puis cliquer sur les masques)")
            continue
        sel = json.loads(sf.read_text())["choices"]
        if not sel:
            say(f"[{n_ep + 1}/{len(eps)}] {ep} : liste de choix vide — épisode passé")
            continue
        sel = sorted(sel, key=lambda c: c["frame"])
        doc = json.loads((task / ep / "data.json").read_text())
        steps = doc["data"]
        H, W = read_rgb(task / ep / steps[0]["colors"][HEAD_CAM]).shape[:2]
        T = len(steps)
        out = np.zeros((T, 2, H, W), bool)
        for ci, ch in enumerate(sel):
            f0 = ch["frame"]
            f1 = sel[ci + 1]["frame"] - 1 if ci + 1 < len(sel) else T - 1
            cm = unpack(np.load(od / f"{a.object}_candidates_{f0:05d}.npz"))
            tmp = Path(tempfile.mkdtemp(prefix="studio_obj_"))
            try:
                for i in range(f0, f1 + 1):
                    os.symlink(task / ep / steps[i]["colors"][HEAD_CAM], tmp / f"{i - f0:05d}.jpg")
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    vs = pred.init_state(video_path=str(tmp), offload_video_to_cpu=True)
                    for oid, key in enumerate(("source", "target")):
                        if ch.get(key) is not None:
                            pred.add_new_mask(vs, frame_idx=0, obj_id=oid, mask=cm[ch[key]])
                    for fi, oids, logits in pred.propagate_in_video(vs):
                        m = (logits[:, 0] > 0).cpu().numpy()
                        for j, oid in enumerate(oids):
                            out[f0 + fi, oid] = m[j]
                    pred.reset_state(vs)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        np.savez_compressed(od / f"{a.object}_tracks.npz", masks=np.packbits(out, axis=-1), shape=np.array(out.shape))
        say(f"[{n_ep + 1}/{len(eps)}] {ep} : {len(sel)} choix suivis (images {[c['frame'] for c in sel]})")


def load_tracks(task: Path, episode: str, name: str) -> np.ndarray:
    return unpack(np.load(task / episode / "objects" / f"{name}_tracks.npz"))


def cmd_preview(a) -> None:
    import cv2
    task = Path(a.task)
    doc = json.loads((task / a.episode / "data.json").read_text())
    masks = load_tracks(task, a.episode, a.object)
    out = Path(a.out or (task / a.episode / "objects" / f"{a.object}_apercu.mp4"))
    T, N, H, W = masks.shape
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
    for fi in range(T):
        im = cv2.imread(str(task / a.episode / doc["data"][fi]["colors"][HEAD_CAM]))
        ov = im.copy()
        ov[masks[fi, 0]] = COLORS_BGR["source"]
        ov[masks[fi, 1]] = COLORS_BGR["cible"]
        im = cv2.addWeighted(ov, 0.5, im, 0.5, 0)
        cv2.putText(im, f"{a.episode} {fi}/{T - 1}  vert = source, rouge = cible", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        vw.write(im)
    vw.release()
    say(f"aperçu : {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("signature")
    s.add_argument("--task", required=True)
    s.add_argument("--object", default="piece")
    d = sub.add_parser("detect")
    d.add_argument("--task", required=True)
    d.add_argument("--object", default="piece")
    d.add_argument("--episodes", nargs="*")
    d.add_argument("--frame", type=int, default=0, help="image où chercher les candidats (main loin des pièces)")
    d.add_argument("--threshold", type=float, default=0.15)
    d.add_argument("--max-instances", type=int, default=12)
    t = sub.add_parser("track")
    t.add_argument("--task", required=True)
    t.add_argument("--object", default="piece")
    t.add_argument("--episodes", nargs="*")
    p = sub.add_parser("preview")
    p.add_argument("--task", required=True)
    p.add_argument("--object", default="piece")
    p.add_argument("--episode", required=True)
    p.add_argument("--out")
    a = ap.parse_args()
    {"signature": cmd_signature, "detect": cmd_detect, "track": cmd_track, "preview": cmd_preview}[a.cmd](a)


if __name__ == "__main__":
    main()
