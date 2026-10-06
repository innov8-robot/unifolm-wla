"""Fenêtre « Objets » du Dataset Studio (OPTIONNELLE) : encadrer un objet, détecter toutes ses instances,
choisir la SOURCE (vert) et la CIBLE (rouge) d'un empilement, suivre ces deux masques.

Les calculs (DINOv3, SAM2) tournent dans un autre environnement (``objects_worker``, lancé par
QProcess) : ce module n'importe ni torch ni sam. Interpréteur : ``$STUDIO_OBJECTS_PY``, par défaut
l'env conda ``unitree_lerobot``. S'il manque, le bouton « Objets… » du studio le dit et rien d'autre
ne change.

Déroulé dans la fenêtre (elle suit l'épisode et l'image du studio) :

1. ENCADRER : glisser un rectangle autour de l'objet sur l'image de tête, « Ajouter l'encadré ».
   10 à 20 encadrés sur des images variées (« Image au hasard »), objet posé ET dans la pince.
2. « Construire la signature ».
3. Sur une image où la main est loin des pièces : « Détecter ici » -> masques numérotés.
4. CHOISIR : clic sur un masque = source (vert), clic sur un autre = cible (rouge), re-clic = retirer.
   « Enregistrer le choix ». Pour un second empilement dans le même épisode : aller plus loin dans la
   vidéo, « Détecter ici », choisir, enregistrer (chaque choix vaut jusqu'au suivant).
5. « Suivre » (cet épisode ou tous ceux qui ont un choix) : les masques suivis s'affichent ensuite
   dans la fenêtre, image par image.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QProcess, QProcessEnvironment, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

REPO = Path(__file__).resolve().parents[1]
WORKER_PY = os.environ.get("STUDIO_OBJECTS_PY", str(Path.home() / "miniconda3/envs/unitree_lerobot/bin/python"))
HEAD_CAM = "color_0"
RGB = {"source": (60, 200, 60), "cible": (230, 60, 60), "candidat": (240, 170, 40)}


def worker_available() -> tuple[bool, str]:
    if not Path(WORKER_PY).exists():
        return False, f"interpréteur introuvable : {WORKER_PY} (variable STUDIO_OBJECTS_PY)"
    return True, WORKER_PY


def unpack(d) -> np.ndarray:
    shp = tuple(d["shape"])
    return np.unpackbits(d["masks"], axis=-1)[..., : shp[-1]].astype(bool).reshape(shp)


class BoxView(QLabel):
    """Image de tête à l'échelle : glisser = rectangle (coordonnées image), clic = point (choix de masque)."""
    box_drawn = Signal(list)
    clicked = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#07090c; border:1px solid #1c2430")
        self.rgb = None
        self._p0 = self._p1 = None
        self._scale, self._off = 1.0, QPoint(0, 0)

    def set_rgb(self, rgb: np.ndarray | None):
        self.rgb = rgb
        self._render()

    def _render(self):
        if self.rgb is None:
            self.setPixmap(QPixmap())
            self.setText("(pas d'image)")
            return
        h, w = self.rgb.shape[:2]
        qi = QImage(np.ascontiguousarray(self.rgb).data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pm = QPixmap.fromImage(qi).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scale = pm.width() / w
        self._off = QPoint((self.width() - pm.width()) // 2, (self.height() - pm.height()) // 2)
        if self._p0 is not None and self._p1 is not None:
            p = QPainter(pm)
            p.setPen(QPen(QColor("#35e0c8"), 2))
            p.drawRect(QRect(self._p0 - self._off, self._p1 - self._off).normalized())
            p.end()
        self.setPixmap(pm)

    def to_img(self, pt: QPoint) -> tuple[int, int]:
        q = pt - self._off
        return int(q.x() / self._scale), int(q.y() / self._scale)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()

    def mousePressEvent(self, e):
        self._p0 = self._p1 = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self._p0 is not None:
            self._p1 = e.position().toPoint()
            self._render()

    def mouseReleaseEvent(self, e):
        if self._p0 is None or self.rgb is None:
            return
        p1 = e.position().toPoint()
        (x0, y0), (x1, y1) = self.to_img(self._p0), self.to_img(p1)
        if abs(x1 - x0) < 6 and abs(y1 - y0) < 6:            # simple clic
            self._p0 = self._p1 = None
            self._render()
            self.clicked.emit(x1, y1)
            return
        h, w = self.rgb.shape[:2]
        box = [max(0, min(x0, x1)), max(0, min(y0, y1)), min(w, max(x0, x1)), min(h, max(y0, y1))]
        self.box_drawn.emit(box)


class ObjectsDialog(QDialog):
    def __init__(self, studio):
        super().__init__(studio)
        self.st = studio
        self.setWindowTitle("Objets — encadrer, détecter, choisir, suivre")
        self.resize(1180, 760)
        self.pending_box = None
        self.cand = None            # (masques (N,H,W), scores, image)
        self.choice = {"source": None, "target": None}
        self.tracks = None          # (T, 2, H, W) de l'épisode courant
        self.proc = None
        self._build()

    # ------------------------------------------------------------------ interface
    def _build(self):
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.view = BoxView()
        self.view.box_drawn.connect(self._on_box)
        self.view.clicked.connect(self._on_click)
        left.addWidget(self.view, 1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        root.addLayout(left, 3)

        right = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("Objet"))
        self.name = QLineEdit("piece")
        self.name.editingFinished.connect(self.refresh)
        row.addWidget(self.name, 1)
        right.addLayout(row)
        right.addWidget(QLabel("1 · ENCADRER (glisser sur l'image)"))
        r1 = QHBoxLayout()
        b_add = QPushButton("Ajouter l'encadré")
        b_add.clicked.connect(self._add_example)
        b_rand = QPushButton("Image au hasard")
        b_rand.clicked.connect(self._random_frame)
        r1.addWidget(b_add)
        r1.addWidget(b_rand)
        right.addLayout(r1)
        self.ex_list = QListWidget()
        self.ex_list.setMaximumHeight(130)
        self.ex_list.itemDoubleClicked.connect(self._goto_example)
        right.addWidget(self.ex_list)
        r2 = QHBoxLayout()
        b_del = QPushButton("Retirer l'exemple")
        b_del.clicked.connect(self._del_example)
        b_sig = QPushButton("2 · Construire la signature")
        b_sig.clicked.connect(lambda: self._run(["signature"]))
        r2.addWidget(b_del)
        r2.addWidget(b_sig)
        right.addLayout(r2)
        right.addWidget(QLabel("3 · DÉTECTER (image où la main est loin)"))
        r3 = QHBoxLayout()
        b_det = QPushButton("Détecter ici")
        b_det.clicked.connect(self._detect_here)
        b_det_all = QPushButton("Tous les épisodes (image 0)")
        b_det_all.clicked.connect(lambda: self._run(["detect", "--frame", "0"]))
        r3.addWidget(b_det)
        r3.addWidget(b_det_all)
        right.addLayout(r3)
        right.addWidget(QLabel("4 · CHOISIR : clic = source (vert), autre clic = cible (rouge) — enregistré tout seul"))
        r4 = QHBoxLayout()
        b_save = QPushButton("Enregistrer le choix")
        b_save.clicked.connect(self._save_choice)
        b_clear = QPushButton("Effacer les choix de l'épisode")
        b_clear.clicked.connect(self._clear_choices)
        r4.addWidget(b_save)
        r4.addWidget(b_clear)
        right.addLayout(r4)
        self.choice_lbl = QLabel("")
        self.choice_lbl.setWordWrap(True)
        right.addWidget(self.choice_lbl)
        right.addWidget(QLabel("5 · SUIVRE"))
        r5 = QHBoxLayout()
        b_tr = QPushButton("Suivre cet épisode")
        b_tr.clicked.connect(self._track_here)
        b_tr_all = QPushButton("Tous les épisodes choisis")
        b_tr_all.clicked.connect(lambda: self._run(["track"]))
        r5.addWidget(b_tr)
        r5.addWidget(b_tr_all)
        right.addLayout(r5)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        right.addWidget(self.log, 1)
        w = QWidget()
        w.setLayout(right)
        root.addWidget(w, 2)

    # ------------------------------------------------------------------ fichiers
    def obj(self) -> str:
        return self.name.text().strip() or "piece"

    def objects_path(self) -> Path:
        return self.st.ds.path / "objects" / "objects.json"

    def load_objects(self) -> dict:
        f = self.objects_path()
        return json.loads(f.read_text()) if f.exists() else {"objects": []}

    def save_objects(self, d: dict):
        f = self.objects_path()
        f.parent.mkdir(exist_ok=True)
        f.write_text(json.dumps(d, indent=1))

    def examples(self) -> list:
        for o in self.load_objects()["objects"]:
            if o["name"] == self.obj():
                return o.get("examples", [])
        return []

    def set_examples(self, ex: list):
        d = self.load_objects()
        for o in d["objects"]:
            if o["name"] == self.obj():
                o["examples"] = ex
                break
        else:
            d["objects"].append({"name": self.obj(), "examples": ex})
        self.save_objects(d)

    def ep_dir(self) -> Path:
        return self.st.ds.path / self.st.ep_name

    def selection_path(self) -> Path:
        return self.ep_dir() / "objects" / f"{self.obj()}_selection.json"

    # ------------------------------------------------------------------ rafraîchissement
    def on_episode(self):
        self.cand = None
        self.choice = {"source": None, "target": None}
        self.tracks = None
        if self.st.ds and self.st.ep_name:
            f = self.ep_dir() / "objects" / f"{self.obj()}_tracks.npz"
            if f.exists():
                try:
                    self.tracks = unpack(np.load(f))
                except Exception as e:
                    self.say(f"suivi illisible : {e}")
        self.refresh()

    def _candidates_file(self, frame: int) -> Path:
        return self.ep_dir() / "objects" / f"{self.obj()}_candidates_{frame:05d}.npz"

    def refresh(self):
        if not self.isVisible() or not self.st.ds or not self.st.doc:
            return
        fr = self.st.frame
        st = self.st.doc["data"][fr]
        rgb = cv2.cvtColor(cv2.imread(str(self.ep_dir() / st["colors"][HEAD_CAM])), cv2.COLOR_BGR2RGB)
        # candidats de CETTE image s'ils existent
        cf = self._candidates_file(fr)
        if cf.exists() and (self.cand is None or self.cand[2] != fr):
            d = np.load(cf)
            self.cand = (unpack(d), d["scores"], fr)
            self.choice = self._saved_choice(fr)
        elif not cf.exists() and self.cand is not None and self.cand[2] != fr:
            self.cand = None
        ov = rgb.copy()
        txt = []
        if self.cand is not None and self.cand[2] == fr:
            masks, sc, _ = self.cand
            for k in range(len(masks)):
                role = "source" if k == self.choice["source"] else "cible" if k == self.choice["target"] else "candidat"
                ov[masks[k]] = RGB[role]
                ys, xs = np.nonzero(masks[k])
                if len(xs):
                    cv2.putText(rgb, f"{k}", (int(xs.mean()) - 5, int(ys.min()) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            txt.append(f"{len(masks)} candidat(s) sur cette image — cliquez pour choisir")
        elif self.tracks is not None and fr < len(self.tracks):
            ov[self.tracks[fr, 0]] = RGB["source"]
            ov[self.tracks[fr, 1]] = RGB["cible"]
            txt.append("masques suivis (vert = source, rouge = cible)")
        img = cv2.addWeighted(ov, 0.5, rgb, 0.5, 0)
        for e in self.examples():
            if e["episode"] == self.st.ep_name and e["frame"] == fr:
                x0, y0, x1, y1 = e["box"]
                cv2.rectangle(img, (x0, y0), (x1, y1), (53, 224, 200), 2)
        if self.pending_box:
            x0, y0, x1, y1 = self.pending_box
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)
        self.view.set_rgb(img)
        self.status.setText(f"{self.st.ep_name} · image {fr} · " + (" · ".join(txt) or "aucun candidat ici"))
        self._refresh_lists()

    def _refresh_lists(self):
        self.ex_list.clear()
        for e in self.examples():
            it = QListWidgetItem(f"{e['episode'].replace('episode_', 'ép ')} · image {e['frame']} · {e['box']}")
            it.setData(Qt.UserRole, e)
            self.ex_list.addItem(it)
        sel = json.loads(self.selection_path().read_text())["choices"] if self.st.ep_name and self.selection_path().exists() else []
        cur = f"source {self.choice['source']} · cible {self.choice['target']}"
        self.choice_lbl.setText(f"choix courant : {cur}\nenregistrés : " +
                                (", ".join(f"image {c['frame']} ({c['source']}→{c['target']})" for c in sel) or "aucun"))

    def _saved_choice(self, frame: int) -> dict:
        if self.selection_path().exists():
            for c in json.loads(self.selection_path().read_text())["choices"]:
                if c["frame"] == frame:
                    return {"source": c["source"], "target": c["target"]}
        return {"source": None, "target": None}

    # ------------------------------------------------------------------ actions
    def say(self, msg: str):
        self.log.appendPlainText(msg)

    def _on_box(self, box):
        self.pending_box = box
        self.refresh()

    def _add_example(self):
        if not self.pending_box or not self.st.ep_name:
            self.say("glissez d'abord un rectangle autour de l'objet")
            return
        ex = self.examples() + [{"episode": self.st.ep_name, "frame": int(self.st.frame), "cam": HEAD_CAM,
                                 "box": [int(v) for v in self.pending_box]}]
        self.set_examples(ex)
        self.pending_box = None
        self.say(f"exemple ajouté ({len(ex)} au total)")
        self.refresh()

    def _del_example(self):
        it = self.ex_list.currentItem()
        if not it:
            return
        e = it.data(Qt.UserRole)
        self.set_examples([x for x in self.examples() if x != e])
        self.refresh()

    def _goto_example(self, it):
        e = it.data(Qt.UserRole)
        self.st.select_episode(e["episode"])
        self.st.seek(e["frame"])

    def _random_frame(self):
        eps = self.st.ds.episodes()
        if not eps:
            return
        ep = random.choice(eps)
        self.st.select_episode(ep.name)              # charge cet épisode seulement
        self.st.seek(random.randrange(max(1, self.st._n())))

    def _detect_here(self):
        if self.st.ep_name:
            self._run(["detect", "--episodes", self.st.ep_name, "--frame", str(self.st.frame)])

    def _on_click(self, x, y):
        if self.cand is None or self.cand[2] != self.st.frame:
            return
        masks = self.cand[0]
        hit = [k for k in range(len(masks)) if 0 <= y < masks.shape[1] and 0 <= x < masks.shape[2] and masks[k][y, x]]
        if not hit:
            return
        k = hit[0]
        if self.choice["source"] == k:
            self.choice["source"] = None
        elif self.choice["target"] == k:
            self.choice["target"] = None
        elif self.choice["source"] is None:
            self.choice["source"] = k
        else:
            self.choice["target"] = k
        self._save_choice(quiet=True)               # enregistré à chaque clic : rien à oublier
        self.refresh()

    def _save_choice(self, quiet: bool = False):
        if self.cand is None or self.cand[2] != self.st.frame:
            if not quiet:
                self.say("détectez d'abord sur cette image (« Détecter ici »), puis cliquez sur les masques")
            return
        f = self.selection_path()
        sel = json.loads(f.read_text())["choices"] if f.exists() else []
        sel = [c for c in sel if c["frame"] != self.st.frame]
        if self.choice["source"] is not None:        # plus rien de choisi sur cette image : choix retiré
            sel.append({"frame": int(self.st.frame), "source": self.choice["source"], "target": self.choice["target"]})
        f.parent.mkdir(exist_ok=True)
        f.write_text(json.dumps({"choices": sorted(sel, key=lambda c: c["frame"])}, indent=1))
        if not quiet:
            self.say(f"{self.st.ep_name} : choix enregistré à l'image {self.st.frame}")
        self._refresh_lists()

    def _track_here(self):
        if not self.st.ep_name:
            return
        f = self.selection_path()
        if not f.exists() or not json.loads(f.read_text())["choices"]:
            self.say(f"{self.st.ep_name} : rien à suivre — « Détecter ici » sur une image où la main est loin, "
                     f"puis cliquez sur la pièce à prendre (vert) et sur la cible (rouge)")
            return
        self._run(["track", "--episodes", self.st.ep_name])

    def _clear_choices(self):
        if self.st.ep_name and self.selection_path().exists():
            self.selection_path().unlink()
            self.choice = {"source": None, "target": None}
            self.say(f"{self.st.ep_name} : choix effacés")
            self.refresh()

    def _run(self, args: list):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.say("un calcul est déjà en cours")
            return
        ok, msg = worker_available()
        if not ok:
            self.say(msg)
            return
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(REPO))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(lambda: [
            self.say(l) for l in bytes(self.proc.readAllStandardOutput()).decode(errors="replace").splitlines()
            if l.strip() and "Warning" not in l and "warn" not in l and "it/s]" not in l])
        self.proc.finished.connect(self._done)
        full = ["-m", "dataset_studio.objects_worker", args[0], "--task", str(self.st.ds.path), "--object", self.obj()] + args[1:]
        self.say(f"▶ {args[0]} …")
        self.proc.start(WORKER_PY, full)

    def _done(self, code, _status):
        self.say("✔ terminé" if code == 0 else f"✘ échec (code {code})")
        self.on_episode()
