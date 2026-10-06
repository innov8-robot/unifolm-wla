"""Service de VISUAL PROMPT en direct : détecte les pièces, suit la SOURCE et la CIBLE choisies, dessine le
vert / rouge sur l'image de tête — même rendu qu'à l'entraînement (``dataset_studio.visual_prompt``).

Lancé par l'Inference Studio dans l'env qui a torch + sam2 + transformers (``unitree_lerobot``) :

    python -m inference_studio.vp_tracker --signature <tâche>/objects/piece_signature.npz

* lit la caméra de tête du robot (ZMQ, comme la téléop : ne la gêne pas) ;
* DÉTECTION sans texte : SAM2 « tout segmenter » + signature DINOv3 (comme la fenêtre Objets du studio) ;
* SUIVI image par image (méthode de mpc_any, tracking/sam2plus.py) : SAM2 est relancé à chaque image avec la
  boîte de l'objet à l'image précédente ; si la confiance tombe (main devant), la dernière position est
  gardée et SAM2 est relancé avec elle quand l'objet réapparaît ;
* service REQ/REP sur tcp://127.0.0.1:8610 (JSON + JPEG en 2e partie) :

  ``status`` · ``detect`` · ``select`` {source, target} · ``reset`` ·
  ``view`` -> image d'affichage (candidats numérotés ou suivi) · ``prompted`` -> image du MODÈLE (œil
  gauche, vert / rouge seulement), refusée si le suivi n'est pas prêt.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dataset_studio.objects_worker import CKPT, CropEmbedder, box_of, iou_box, score  # noqa: E402
from dataset_studio.visual_prompt import overlay  # noqa: E402

CONF_MIN = 0.5            # score SAM2 sous lequel la mesure est rejetée (occultation)
GROW = 1.25               # boîte de la frame précédente agrandie pour la relance
AREA_MAX = 1.8            # masque plus grand que AREA_MAX × l'aire de départ : refusé (il « avale » la pince)
AREA_MIN = 0.2            # plus petit que AREA_MIN × l'aire de départ : refusé (pièce presque cachée)
MAX_AGE = 0.6             # image coloriée plus vieille : refusée au modèle (s)


class Tracker:
    def __init__(self, sig_path: Path, cam: str, device: str = "cuda"):
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch
        self.dev = device
        model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(CKPT / "sam2.1_hiera_large.pt"), device=device)
        self.pred = SAM2ImagePredictor(model)
        self.amg = SAM2AutomaticMaskGenerator(model, points_per_side=32, pred_iou_thresh=0.7,
                                              stability_score_thresh=0.85, min_mask_region_area=30)
        self.emb = CropEmbedder(device)
        self.sig = dict(np.load(sig_path))
        self.lock = threading.Lock()
        self.frame = None            # œil gauche BGR courant
        self.frame_t = 0.0
        self.cands = []              # masques candidats (sélection)
        self.objs = {}               # "source" / "target" -> {"box": xyxy, "mask": bool}
        self.prompted = None         # (image coloriée, horodatage)
        self.ctx = zmq.Context.instance()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub.setsockopt(zmq.CONFLATE, 1)
        self.sub.setsockopt(zmq.RCVTIMEO, 1000)
        self.sub.connect(cam)
        self.fps = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------------------------------------------ boucle caméra + suivi
    def _loop(self):
        n, t0 = 0, time.time()
        while True:
            try:
                buf = self.sub.recv()
            except zmq.Again:
                continue
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            left = img[:, : w // 2] if w >= 2 * h else img           # œil gauche = image du modèle
            t = time.time()
            with self.lock:
                self.frame, self.frame_t = left, t
                tracking = dict(self.objs)
            if tracking:
                masks = self._track(left, tracking)
                with self.lock:
                    if self.objs:                                   # pas réinitialisé entre-temps
                        for k, m in masks.items():
                            if m is not None:
                                self.objs[k]["mask"] = m
                                self.objs[k]["box"] = box_of(m)
                        self.prompted = (overlay(left, self.objs.get("source", {}).get("mask"),
                                                 self.objs.get("target", {}).get("mask")), t)
            n += 1
            if time.time() - t0 > 2:
                self.fps, n, t0 = n / (time.time() - t0), 0, time.time()

    def _track(self, img, objs) -> dict:
        out = {}
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self.torch.inference_mode(), self.torch.autocast(self.dev, dtype=self.torch.bfloat16, enabled=self.dev == "cuda"):
            self.pred.set_image(rgb)
            for k, o in objs.items():
                x0, y0, x1, y1 = o["box"]
                cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * GROW, (y1 - y0) * GROW
                box = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], np.float32)
                masks, scores, _ = self.pred.predict(box=box, multimask_output=False)
                m = np.asarray(masks[0]) > 0
                a = float(m.sum())
                ok = float(scores[0]) >= CONF_MIN and AREA_MIN * o["area0"] <= a <= AREA_MAX * o["area0"]
                out[k] = m if ok else None                          # occultation : on garde l'ancien
        return out

    # ------------------------------------------------------------------ commandes
    def detect(self, threshold: float = 0.15, max_inst: int = 12) -> int:
        with self.lock:
            img = None if self.frame is None else self.frame.copy()
        if img is None:
            raise RuntimeError("pas d'image de la caméra de tête")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self.torch.inference_mode(), self.torch.autocast(self.dev, dtype=self.torch.bfloat16, enabled=self.dev == "cuda"):
            props = self.amg.generate(rgb)
        cand = [p["segmentation"] for p in props if 30 < p["segmentation"].sum() < 0.15 * img.shape[0] * img.shape[1]]
        boxes = [box_of(m) for m in cand]
        sc = score(self.emb.embed([self.emb.crop(rgb, b) for b in boxes]), self.sig) if cand else np.zeros(0)
        keep = []
        for i in [i for i in np.argsort(-sc) if sc[i] > threshold]:
            if all(iou_box(boxes[i], boxes[j]) < 0.5 for j in keep):
                keep.append(i)
        keep = sorted(keep[:max_inst], key=lambda i: boxes[i][0])  # numérotés de GAUCHE à DROITE
        with self.lock:
            self.cands = [cand[i] for i in keep]
            self.objs = {}
            self.prompted = None
        return len(keep)

    def select(self, source: int, target: int):
        with self.lock:
            if not (0 <= source < len(self.cands) and 0 <= target < len(self.cands)) or source == target:
                raise ValueError("source / cible invalides")
            self.objs = {k: {"mask": self.cands[i], "box": box_of(self.cands[i]), "area0": float(self.cands[i].sum())}
                         for k, i in (("source", source), ("target", target))}
            self.prompted = None

    def reset(self):
        with self.lock:
            self.cands, self.objs, self.prompted = [], {}, None

    def view(self) -> np.ndarray | None:
        """Image d'affichage : candidats numérotés (sélection) ou suivi en couleurs."""
        with self.lock:
            img = None if self.frame is None else self.frame.copy()
            cands, objs = list(self.cands), dict(self.objs)
            pr = self.prompted
        if img is None:
            return None
        if objs and pr is not None:
            return pr[0]
        out = img.copy()
        for k, m in enumerate(cands):
            fill = out.copy()
            fill[m] = (40, 170, 240)
            out = cv2.addWeighted(fill, 0.4, out, 0.6, 0)
            ys, xs = np.nonzero(m)
            if len(xs):
                org = (int(xs.mean()) - 8, max(20, int(ys.min()) - 6))
                cv2.putText(out, str(k + 1), org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
                cv2.putText(out, str(k + 1), org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return out

    def status(self) -> dict:
        with self.lock:
            age = time.time() - self.prompted[1] if self.prompted else None
            return {"camera": self.frame is not None and time.time() - self.frame_t < 1.0, "fps": round(self.fps, 1),
                    "candidates": len(self.cands), "tracking": bool(self.objs),
                    "prompted_age": None if age is None else round(age, 2),
                    "ready": bool(self.objs) and age is not None and age < MAX_AGE}


def serve(tr: Tracker, port: int):
    rep = zmq.Context.instance().socket(zmq.REP)
    rep.bind(f"tcp://127.0.0.1:{port}")
    print(f"[vp_tracker] prêt : tcp://127.0.0.1:{port}", flush=True)
    while True:
        msg = rep.recv_json()
        cmd, img = msg.get("cmd"), None
        try:
            if cmd == "detect":
                res = {"ok": True, "candidates": tr.detect()}
            elif cmd == "select":
                tr.select(int(msg["source"]), int(msg["target"]))
                res = {"ok": True}
            elif cmd == "reset":
                tr.reset()
                res = {"ok": True}
            elif cmd == "view":
                img = tr.view()
                res = {"ok": img is not None}
            elif cmd == "prompted":
                st = tr.status()
                img = tr.prompted[0] if st["ready"] and tr.prompted else None
                res = {"ok": img is not None, **st}
            else:
                res = {"ok": True}
            res.update(tr.status())
        except Exception as e:
            res = {"ok": False, "error": str(e), **tr.status()}
        if img is not None:
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rep.send_multipart([json.dumps(res).encode(), jpg.tobytes()])
        else:
            rep.send_multipart([json.dumps(res).encode(), b""])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signature", required=True)
    ap.add_argument("--cam", default="tcp://192.168.123.164:55558")
    ap.add_argument("--port", type=int, default=8610)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    tr = Tracker(Path(a.signature), a.cam, a.device)
    serve(tr, a.port)


if __name__ == "__main__":
    main()
