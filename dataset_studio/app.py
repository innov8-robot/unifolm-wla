"""Dataset Studio : voir, contrôler et modifier les enregistrements xr_teleoperate du G1-D.

    conda activate g1d_teleop && python -m dataset_studio [dossier]      # depuis la racine du dépôt

Le dossier est une tâche (contenant des episode_XXXX/) ou un dossier de tâches (ex. utils/data).

Raccourcis : Espace lecture/pause · ←/→ pas à pas · Maj+←/→ ±1 s · ↑/↓ épisode précédent/suivant ·
I / O début / fin (rognage ou segment) · 1-9 choisir l'étiquette · T étiqueter [début, fin] ·
Suppr supprimer les épisodes sélectionnés.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QColor, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QSlider, QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from .charts import SignalChart
from .model import (CAMERA_LABELS, RL_TAGS, TaskDataset, find_tasks, guess_layout, load_doc, load_segments,
                    signals, summarize)
from .theme import ACCENT, KO, MUTED, OK, QSS, TAG_COLORS, WARN

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO / "teleoperation/Tele_OP/xr_teleoperate/teleop/utils/data"
FPS = 30.0


def panel(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("Panel")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(10, 8, 10, 10)
    lay.setSpacing(6)
    if title:
        h = QLabel(title.upper())
        h.setObjectName("H2")
        lay.addWidget(h)
    return f, lay


def button(text: str, kind: str | None = None, tip: str | None = None) -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if tip:
        b.setToolTip(tip)
    return b


class CamView(QLabel):
    def __init__(self, title: str):
        super().__init__(title)
        self.setObjectName("Cam")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 150)
        self._pix = None
        self.title = title

    def show_image(self, path: Path | None, title: str):
        self.title = title
        self._pix = QPixmap(str(path)) if path is not None and path.exists() else None
        self._render()

    def _render(self):
        if self._pix is None or self._pix.isNull():
            self.setPixmap(QPixmap())
            self.setText(f"{self.title}\n(image absente)")
            return
        self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setToolTip(self.title)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()


class TrashDialog(QDialog):
    def __init__(self, ds: TaskDataset, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Corbeille")
        self.resize(520, 360)
        self.ds = ds
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Épisodes supprimés (le plus récent en haut). « Restaurer » les remet À LA FIN du dataset."))
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for p in ds.trashed():
            self.list.addItem(p.name)
        lay.addWidget(self.list)
        row = QHBoxLayout()
        self.b_restore = button("Restaurer la sélection")
        self.b_empty = button("Vider la corbeille (définitif)", "Danger")
        row.addWidget(self.b_restore)
        row.addStretch()
        row.addWidget(self.b_empty)
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.b_restore.clicked.connect(self._restore)
        self.b_empty.clicked.connect(self._empty)
        self.changed = False

    def _restore(self):
        for it in self.list.selectedItems():
            self.ds.restore(self.ds.trash_dir() / it.text())
            self.changed = True
        self.list.clear()
        for p in self.ds.trashed():
            self.list.addItem(p.name)

    def _empty(self):
        if QMessageBox.warning(self, "Vider la corbeille", "Supprimer DÉFINITIVEMENT tout le contenu de la corbeille ?\n"
                               "(épisodes supprimés et sauvegardes de rognage)",
                               QMessageBox.Yes | QMessageBox.Cancel) == QMessageBox.Yes:
            self.ds.empty_trash()
            self.list.clear()


class TagDialog(QDialog):
    """Nouvelle étiquette (nom + consigne), ou modification du nom et de la consigne d'une étiquette."""

    def __init__(self, parent=None, name: str = "", instruction: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Étiquette" if not name else f"Étiquette « {name} »")
        self.resize(560, 150)
        form = QFormLayout(self)
        self.name = QLineEdit(name)
        self.name.setPlaceholderText("prise_gauche")
        self.instr = QLineEdit(instruction)
        self.instr.setPlaceholderText("pick up the black object with the left hand")
        form.addRow("Nom (sans espace)", self.name)
        form.addRow("Consigne du modèle", self.instr)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)


class Studio(QMainWindow):
    def __init__(self, start: Path):
        super().__init__()
        self.setWindowTitle("Dataset Studio — G1-D")
        self.resize(1600, 980)
        self.ds: TaskDataset | None = None
        self.root: Path | None = None
        self.doc = None
        self.ep_name = None
        self.frame = 0
        self.mark_in = self.mark_out = None
        self.layout_name = "binocular"
        self.sig = {}
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.speed = 1.0
        self.proc = None
        self.obj_dlg = None
        self._build()
        self._shortcuts()
        self.open_root(start)

    # ------------------------------------------------------------------ construction
    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        # en-tête
        head = QFrame()
        head.setObjectName("Header")
        h = QHBoxLayout(head)
        h.setContentsMargins(14, 8, 14, 8)
        t = QLabel("DATASET STUDIO")
        t.setObjectName("Title")
        h.addWidget(t)
        h.addSpacing(18)
        self.task_combo = QComboBox()
        self.task_combo.setMinimumWidth(220)
        self.task_combo.currentIndexChanged.connect(self._task_changed)
        h.addWidget(QLabel("Tâche"))
        h.addWidget(self.task_combo)
        self.path_lbl = QLabel("")
        self.path_lbl.setStyleSheet(f"color:{MUTED}")
        h.addWidget(self.path_lbl, 1)
        b_open = button("Ouvrir…")
        b_open.clicked.connect(self._choose_dir)
        b_obj = button("Objets…", "Quiet", "Encadrer un objet, détecter ses instances, choisir source / cible, suivre (optionnel)")
        b_obj.clicked.connect(self._objects)
        h.addWidget(b_obj)
        b_reload = button("Recharger", "Quiet")
        b_reload.clicked.connect(lambda: self.load_task(self.ds.path) if self.ds else None)
        h.addWidget(b_reload)
        h.addWidget(b_open)
        v.addWidget(head)

        body = QSplitter(Qt.Horizontal)
        body.setContentsMargins(8, 8, 8, 8)
        v.addWidget(body, 1)

        # --- gauche : épisodes
        left, ll = panel("Épisodes")
        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet(f"color:{MUTED}")
        ll.addWidget(self.summary_lbl)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ÉPISODE", "DURÉE", "PAS", "ISSUE", "PINCES G/D", "INDICATEURS"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        ll.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.b_delete = button("🗑 Supprimer", "Danger", "Déplace les épisodes sélectionnés dans la corbeille et renumérote (Suppr)")
        self.b_delete.clicked.connect(self._delete)
        b_trash = button("Corbeille…", "Quiet")
        b_trash.clicked.connect(self._trash)
        b_renum = button("Renuméroter", "Quiet", "episode_0000, 0001… sans trou")
        b_renum.clicked.connect(self._renumber)
        row.addWidget(self.b_delete)
        row.addStretch()
        row.addWidget(b_renum)
        row.addWidget(b_trash)
        ll.addLayout(row)
        self.b_convert = button("Convertir au format WLA", "Run", "g1d_wla.convert_teleop -> playground/Datasets/g1d_<tâche>/")
        self.b_convert.clicked.connect(self._convert)
        ll.addWidget(self.b_convert)
        body.addWidget(left)

        # --- centre : caméras + lecture + courbes
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cams_f, cams_l = panel("Caméras")
        grid = QGridLayout()
        grid.setSpacing(6)
        self.cams = [CamView(f"caméra {i}") for i in range(4)]
        for i, c in enumerate(self.cams):
            grid.addWidget(c, i // 2, i % 2)
        cams_l.addLayout(grid, 1)
        # transport
        tr = QHBoxLayout()
        self.b_first = button("⏮", "Quiet")
        self.b_back = button("◀", "Quiet")
        self.b_play = button("▶  Lecture", "Run")
        self.b_fwd = button("▶", "Quiet")
        self.b_last = button("⏭", "Quiet")
        self.b_first.clicked.connect(lambda: self.seek(0))
        self.b_back.clicked.connect(lambda: self.seek(self.frame - 1))
        self.b_fwd.clicked.connect(lambda: self.seek(self.frame + 1))
        self.b_last.clicked.connect(lambda: self.seek(self._n() - 1))
        self.b_play.clicked.connect(self.toggle_play)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["×0.25", "×0.5", "×1", "×2", "×4"])
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._slider_moved)
        self.frame_lbl = QLabel("—")
        self.frame_lbl.setObjectName("Mono")
        self.frame_lbl.setMinimumWidth(190)
        for w in (self.b_first, self.b_back, self.b_play, self.b_fwd, self.b_last, self.speed_combo):
            tr.addWidget(w)
        tr.addWidget(self.slider, 1)
        tr.addWidget(self.frame_lbl)
        cams_l.addLayout(tr)
        cv.addWidget(cams_f, 3)
        charts_f, charts_l = panel("Signaux")
        self.charts, self.chart_combos = [], []
        for default in ("Bras droit — état", "Pinces (Dex1, 0 fermée → 5,4)"):
            r = QHBoxLayout()
            cb = QComboBox()
            cb.setMinimumWidth(260)
            cb.setProperty("default", default)
            ch = SignalChart()
            ch.seek.connect(self.seek)
            ch.segment_action.connect(self._chart_segment_action)
            cb.currentTextChanged.connect(lambda txt, c=ch: self._chart_group(c, txt))
            r.addWidget(cb)
            r.addStretch()
            charts_l.addLayout(r)
            charts_l.addWidget(ch, 1)
            self.charts.append(ch)
            self.chart_combos.append(cb)
        cv.addWidget(charts_f, 2)
        body.addWidget(center)

        # --- droite : édition
        right, rl = panel("Épisode")
        self.ep_lbl = QLabel("—")
        self.ep_lbl.setObjectName("Title")
        rl.addWidget(self.ep_lbl)
        rl.addWidget(QLabel("Consigne (instruction du modèle)"))
        self.goal_edit = QLineEdit()
        rl.addWidget(self.goal_edit)
        gr = QHBoxLayout()
        b_goal1 = button("Cet épisode")
        b_goal_sel = button("Sélection")
        b_goal_all = button("Tous", "Danger")
        b_goal1.clicked.connect(lambda: self._set_goal("one"))
        b_goal_sel.clicked.connect(lambda: self._set_goal("sel"))
        b_goal_all.clicked.connect(lambda: self._set_goal("all"))
        for b in (b_goal1, b_goal_sel, b_goal_all):
            gr.addWidget(b)
        rl.addLayout(gr)
        rl.addSpacing(8)
        rl.addWidget(QLabel("Issue de l'essai"))
        orow = QHBoxLayout()
        self.outcome_combo = QComboBox()
        self.outcome_combo.addItems(["— (non renseignée)", "success", "failure", "unknown"])
        b_out = button("Appliquer")
        b_out.clicked.connect(self._set_outcome)
        orow.addWidget(self.outcome_combo, 1)
        orow.addWidget(b_out)
        rl.addLayout(orow)
        rl.addSpacing(8)
        h2 = QLabel("ROGNAGE")
        h2.setObjectName("H2")
        rl.addWidget(h2)
        self.trim_lbl = QLabel("début — · fin —")
        self.trim_lbl.setObjectName("Mono")
        rl.addWidget(self.trim_lbl)
        trow = QHBoxLayout()
        b_in = button("Début ici  [I]")
        b_out2 = button("Fin ici  [O]")
        b_clear = button("Effacer", "Quiet")
        b_in.clicked.connect(self.set_in)
        b_out2.clicked.connect(self.set_out)
        b_clear.clicked.connect(self.clear_marks)
        trow.addWidget(b_in)
        trow.addWidget(b_out2)
        trow.addWidget(b_clear)
        rl.addLayout(trow)
        self.b_trim = button("✂ Rogner l'épisode", "Danger", "Garde [début, fin] ; l'original va dans la corbeille")
        self.b_trim.clicked.connect(self._trim)
        rl.addWidget(self.b_trim)
        rl.addSpacing(8)
        h5 = QLabel("DÉCOUPAGE EN SOUS-TÂCHES")
        h5.setObjectName("H2")
        rl.addWidget(h5)
        tgrow = QHBoxLayout()
        self.tag_combo = QComboBox()
        self.tag_combo.setMinimumWidth(170)
        self.tag_combo.currentIndexChanged.connect(self._tag_changed)
        b_tag_new = button("＋", "Quiet", "Nouvelle étiquette (nom + consigne du modèle)")
        b_tag_edit = button("✎", "Quiet", "Renommer l'étiquette ou modifier sa consigne")
        b_tag_del = button("−", "Quiet", "Supprimer l'étiquette (et ses segments, après confirmation)")
        b_tag_new.clicked.connect(self._new_tag)
        b_tag_edit.clicked.connect(self._edit_tag)
        b_tag_del.clicked.connect(self._del_tag)
        tgrow.addWidget(self.tag_combo, 1)
        for b in (b_tag_new, b_tag_edit, b_tag_del):
            b.setFixedWidth(40)
            tgrow.addWidget(b)
        rl.addLayout(tgrow)
        self.tag_instr_lbl = QLabel("")
        self.tag_instr_lbl.setWordWrap(True)
        self.tag_instr_lbl.setStyleSheet(f"color:{MUTED}")
        rl.addWidget(self.tag_instr_lbl)
        keys_lbl = QLabel("1-9 : choisir l'étiquette · I : début · T : poser jusqu'au pas courant · clic droit sur une bande : supprimer")
        keys_lbl.setWordWrap(True)
        keys_lbl.setStyleSheet(f"color:{MUTED}; font-size:11px")
        rl.addWidget(keys_lbl)
        self.b_tag = button("🏷 Étiqueter [début, fin]  [T]", None, "Crée un segment avec l'étiquette choisie ; le début suivant est placé juste après")
        self.b_tag.clicked.connect(self._tag_segment)
        rl.addWidget(self.b_tag)
        self.seg_list = QListWidget()
        self.seg_list.setMaximumHeight(120)
        self.seg_list.itemDoubleClicked.connect(self._goto_segment)
        self.seg_list.setToolTip("Double-clic : aller au segment · Suppr : supprimer le segment sélectionné")
        rl.addWidget(self.seg_list)
        srow = QHBoxLayout()
        b_seg_del = button("Supprimer le segment", "Quiet")
        b_seg_del.clicked.connect(self._del_segment)
        b_rl = button("＋ bon / mauvais (RL)", "Quiet",
                      "Étiquettes d'avantage manuel : remplacent le jugement du modèle de valeur de RECAP")
        b_rl.clicked.connect(self._rl_tags)
        srow.addWidget(b_seg_del)
        srow.addStretch()
        srow.addWidget(b_rl)
        rl.addLayout(srow)
        self.tag_usage_lbl = QLabel("")
        self.tag_usage_lbl.setWordWrap(True)
        self.tag_usage_lbl.setObjectName("Mono")
        rl.addWidget(self.tag_usage_lbl)
        erow = QHBoxLayout()
        b_exp_tag = button("Exporter par étiquette…", None, "Un dossier de tâche par étiquette : <tâche>__<étiquette>")
        b_exp_all = button("Exporter tout…", None, "Un seul dossier <tâche>__segments, une consigne par étiquette")
        b_exp_tag.clicked.connect(lambda: self._export("tag"))
        b_exp_all.clicked.connect(lambda: self._export("all"))
        erow.addWidget(b_exp_tag)
        erow.addWidget(b_exp_all)
        rl.addLayout(erow)
        rl.addSpacing(8)
        h3 = QLabel("EN-TÊTE (info)")
        h3.setObjectName("H2")
        rl.addWidget(h3)
        self.info_text = QPlainTextEdit()
        self.info_text.setReadOnly(True)
        rl.addWidget(self.info_text, 1)
        h4 = QLabel("JOURNAL")
        h4.setObjectName("H2")
        rl.addWidget(h4)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        rl.addWidget(self.log, 1)
        body.addWidget(right)
        body.setSizes([380, 900, 360])

    def _shortcuts(self):
        def sc(key, fn):
            s = QShortcut(QKeySequence(key), self)
            s.activated.connect(fn)
        sc(Qt.Key_Space, self.toggle_play)
        sc(Qt.Key_Right, lambda: self.seek(self.frame + 1))
        sc(Qt.Key_Left, lambda: self.seek(self.frame - 1))
        sc("Shift+Right", lambda: self.seek(self.frame + int(FPS)))
        sc("Shift+Left", lambda: self.seek(self.frame - int(FPS)))
        sc(Qt.Key_Up, lambda: self._step_episode(-1))
        sc(Qt.Key_Down, lambda: self._step_episode(+1))
        sc(Qt.Key_I, self.set_in)
        sc(Qt.Key_O, self.set_out)
        sc(Qt.Key_T, self._tag_segment)
        for k in range(1, 10):
            sc(str(k), lambda k=k: self.tag_combo.setCurrentIndex(k - 1) if k <= self.tag_combo.count() else None)
        sc(Qt.Key_Delete, self._delete_key)

    # ------------------------------------------------------------------ chargement
    def say(self, msg: str, color: str | None = None):
        self.log.appendHtml(f'<span style="color:{color or "#cdd6e0"}">{msg}</span>')

    def _choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Dossier de tâche ou dossier de tâches", str(DEFAULT_DIR))
        if d:
            self.open_root(Path(d))

    def open_root(self, root: Path):
        if not root.exists():
            self.say(f"dossier introuvable : {root}", KO)
            return
        tasks = find_tasks(root)
        self.root = root
        self.task_combo.blockSignals(True)
        self.task_combo.clear()
        for t in tasks:
            self.task_combo.addItem(t.name, str(t))
        self.task_combo.blockSignals(False)
        if tasks:
            self.load_task(tasks[0])
        else:
            self.say(f"aucun episode_* dans {root}", WARN)

    def _task_changed(self, i):
        if i >= 0:
            self.load_task(Path(self.task_combo.itemData(i)))

    def load_task(self, path: Path, select: str | None = None):
        self.timer.stop()
        self.ds = TaskDataset(path)
        self.path_lbl.setText(str(path))
        eps = self.ds.episodes()
        self.table.setRowCount(len(eps))
        tot = 0
        for r, ep in enumerate(eps):
            s = summarize(ep)
            nseg = len(load_segments(ep))
            if nseg:
                s.flags.insert(0, f"{nseg} seg")
            tot += s.n
            out_col = {"success": OK, "failure": KO, "unknown": WARN}.get(s.outcome, MUTED)
            cells = [s.name.replace("episode_", ""), f"{s.seconds:5.1f} s", str(s.n), s.outcome or "—",
                     f"{'●' if s.left_gripper_used else '○'} / {'●' if s.right_gripper_used else '○'}",
                     ", ".join(s.flags) if not s.error else s.error]
            for c, txt in enumerate(cells):
                it = QTableWidgetItem(txt)
                if c == 0:
                    it.setData(Qt.UserRole, s.name)
                if c == 3:
                    it.setForeground(QColor(out_col))
                if c == 5 and (s.error or "manquantes" in txt):
                    it.setForeground(QColor(KO))
                self.table.setItem(r, c, it)
        self._refresh_tags()
        n_tr = len(self.ds.trashed())
        self.summary_lbl.setText(f"{len(eps)} épisodes · {tot} pas · {tot / FPS / 60:.1f} min"
                                 + (f" · corbeille : {n_tr}" if n_tr else ""))
        if eps:
            row = 0
            if select:
                for r in range(self.table.rowCount()):
                    if self.table.item(r, 0).data(Qt.UserRole) == select:
                        row = r
            self.table.selectRow(row)
        else:
            self.doc = None

    def _selected_names(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.table.item(r, 0).data(Qt.UserRole) for r in rows]

    def _selection_changed(self):
        names = self._selected_names()
        if names and names[-1] != self.ep_name:
            self.load_episode(names[-1])

    def _step_episode(self, d):
        r = self.table.currentRow() + d
        if 0 <= r < self.table.rowCount():
            self.table.selectRow(r)

    def load_episode(self, name: str):
        self.timer.stop()
        self.b_play.setText("▶  Lecture")
        try:
            self.doc = load_doc(self.ds.path / name)
        except Exception as e:
            self.doc = None
            self.say(f"{name} : {e}", KO)
            return
        self.ep_name = name
        self.layout_name = guess_layout(self.doc) if self.doc["data"] else "binocular"
        self.mark_in = self.mark_out = None
        self.ep_lbl.setText(name)
        self.goal_edit.setText(self.doc.get("text", {}).get("goal", ""))
        oc = self.doc.get("info", {}).get("outcome")
        self.outcome_combo.setCurrentIndex({"success": 1, "failure": 2, "unknown": 3}.get(oc, 0))
        info = {k: v for k, v in self.doc.get("info", {}).items() if k not in ("joint_names", "tactile_names", "audio", "depth")}
        self.info_text.setPlainText(json.dumps(info, indent=1, ensure_ascii=False))
        self.sig = signals(self.doc) if self.doc["data"] else {}
        for cb, ch in zip(self.chart_combos, self.charts):
            prev = cb.currentText() or cb.property("default")
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(list(self.sig))
            cb.blockSignals(False)
            cb.setCurrentText(prev if prev in self.sig else cb.property("default"))
            self._chart_group(ch, cb.currentText())
        n = self._n()
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, n - 1))
        self.slider.blockSignals(False)
        self._update_marks()
        self._refresh_segments()
        if self.obj_dlg is not None and self.obj_dlg.isVisible():
            self.obj_dlg.on_episode()
        self.seek(0)

    def _chart_group(self, chart: SignalChart, group: str):
        chart.set_series(self.sig.get(group, {}), self._n())
        chart.set_cursor(self.frame)
        chart.set_marks(self.mark_in, self.mark_out)
        chart.set_segments(self._segment_bands())

    def _n(self) -> int:
        return len(self.doc["data"]) if self.doc else 0

    # ------------------------------------------------------------------ lecture
    def seek(self, k: int):
        n = self._n()
        if n == 0:
            return
        self.frame = max(0, min(n - 1, int(k)))
        st = self.doc["data"][self.frame]
        labels = CAMERA_LABELS.get(self.layout_name, {})
        keys = sorted(st["colors"])
        for i, cam in enumerate(self.cams):
            if i < len(keys):
                cam.setVisible(True)
                cam.show_image(self.ds.path / self.ep_name / st["colors"][keys[i]], labels.get(keys[i], keys[i]))
            else:
                cam.setVisible(False)
        self.slider.blockSignals(True)
        self.slider.setValue(self.frame)
        self.slider.blockSignals(False)
        self.frame_lbl.setText(f"pas {self.frame:4d}/{n - 1}   {self.frame / FPS:6.2f} s")
        if self.obj_dlg is not None and self.obj_dlg.isVisible():
            self.obj_dlg.refresh()
        for ch in self.charts:
            ch.set_cursor(self.frame)

    def _slider_moved(self, v):
        self.seek(v)

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.b_play.setText("▶  Lecture")
        elif self._n():
            if self.frame >= self._n() - 1:
                self.seek(0)
            self.timer.start(int(1000 / (FPS * self.speed)))
            self.b_play.setText("⏸  Pause")

    def _speed_changed(self, i):
        self.speed = [0.25, 0.5, 1.0, 2.0, 4.0][i]
        if self.timer.isActive():
            self.timer.start(int(1000 / (FPS * self.speed)))

    def _tick(self):
        if self.frame >= self._n() - 1:
            self.toggle_play()
            return
        self.seek(self.frame + 1)

    # ------------------------------------------------------------------ édition
    def set_in(self):
        self.mark_in = self.frame
        self._update_marks()

    def set_out(self):
        self.mark_out = self.frame
        self._update_marks()

    def clear_marks(self):
        self.mark_in = self.mark_out = None
        self._update_marks()

    def _update_marks(self):
        a = "—" if self.mark_in is None else f"{self.mark_in} ({self.mark_in / FPS:.2f} s)"
        b = "—" if self.mark_out is None else f"{self.mark_out} ({self.mark_out / FPS:.2f} s)"
        self.trim_lbl.setText(f"début {a} · fin {b}")
        for ch in self.charts:
            ch.set_marks(self.mark_in, self.mark_out)

    def _trim(self):
        if not self.doc or (self.mark_in is None and self.mark_out is None):
            self.say("rognage : placer d'abord un début [I] et/ou une fin [O]", WARN)
            return
        a = self.mark_in or 0
        b = self.mark_out if self.mark_out is not None else self._n() - 1
        if b <= a:
            self.say("rognage : la fin doit être après le début", KO)
            return
        if QMessageBox.question(self, "Rogner", f"{self.ep_name} : garder les pas {a} à {b} ({(b - a + 1) / FPS:.1f} s) "
                                f"sur {self._n()} ?\nL'original est gardé dans la corbeille.") != QMessageBox.Yes:
            return
        n = self.ds.trim(self.ep_name, a, b)
        self.say(f"✂ {self.ep_name} rogné : {n} pas", OK)
        self.load_task(self.ds.path, select=self.ep_name)

    def _set_goal(self, scope: str):
        goal = self.goal_edit.text().strip()
        if not goal or not self.ds:
            return
        if scope == "one":
            names = [self.ep_name]
        elif scope == "sel":
            names = self._selected_names()
        else:
            names = [p.name for p in self.ds.episodes()]
            if QMessageBox.question(self, "Consigne", f"Appliquer « {goal} » aux {len(names)} épisodes ?") != QMessageBox.Yes:
                return
        self.ds.set_goal(names, goal)
        self.say(f"consigne « {goal} » appliquée à {len(names)} épisode(s)", OK)

    def _set_outcome(self):
        if not self.ep_name:
            return
        i = self.outcome_combo.currentIndex()
        oc = [None, "success", "failure", "unknown"][i]
        self.ds.set_outcome(self.ep_name, oc)
        self.say(f"{self.ep_name} : issue = {oc or '—'}", OK)
        self.load_task(self.ds.path, select=self.ep_name)

    def _delete(self):
        names = self._selected_names()
        if not names or not self.ds:
            return
        txt = ", ".join(n.replace("episode_", "") for n in names[:12]) + (" …" if len(names) > 12 else "")
        if QMessageBox.warning(self, "Supprimer", f"Supprimer {len(names)} épisode(s) : {txt} ?\n\n"
                               "Ils vont dans la corbeille (restaurables) et les suivants sont RENUMÉROTÉS.",
                               QMessageBox.Yes | QMessageBox.Cancel) != QMessageBox.Yes:
            return
        row = self.table.currentRow()
        self.ds.delete(names)
        self.ep_name = None
        self.say(f"🗑 {len(names)} épisode(s) en corbeille, épisodes renumérotés", OK)
        self.load_task(self.ds.path)
        if self.table.rowCount():
            self.table.selectRow(min(row, self.table.rowCount() - 1))

    def _renumber(self):
        if self.ds:
            ren = self.ds.renumber()
            self.say(f"renumérotation : {len(ren)} dossier(s) renommé(s)", OK)
            self.load_task(self.ds.path)

    def _trash(self):
        if not self.ds:
            return
        d = TrashDialog(self.ds, self)
        d.exec()
        if d.changed:
            self.load_task(self.ds.path)

    # ------------------------------------------------------------------ objets (optionnel)
    def _objects(self):
        if not self.ds:
            return
        try:
            from .objects_ui import ObjectsDialog, worker_available
        except Exception as e:                       # cv2 absent, etc. : le reste du studio n'est pas touché
            self.say(f"fenêtre Objets indisponible : {e}", KO)
            return
        ok, msg = worker_available()
        if not ok:
            self.say(f"Objets : {msg}", WARN)
        if self.obj_dlg is None:
            self.obj_dlg = ObjectsDialog(self)
        self.obj_dlg.show()
        self.obj_dlg.raise_()
        self.obj_dlg.on_episode()

    # ------------------------------------------------------------------ étiquettes et segments
    def _tag_color(self, name: str) -> str:
        rl = dict(RL_TAGS)
        if name in rl:
            return rl[name]
        names = [t["name"] for t in self.ds.tags()] if self.ds else []
        return TAG_COLORS[names.index(name) % len(TAG_COLORS)] if name in names else MUTED

    def _current_tag(self) -> str | None:
        return self.tag_combo.currentData()

    def _refresh_tags(self, select: str | None = None):
        cur = select or self._current_tag()
        tags = self.ds.tags() if self.ds else []
        self.tag_combo.blockSignals(True)
        self.tag_combo.clear()
        for i, t in enumerate(tags):
            self.tag_combo.addItem(f"{i + 1} · {t['name']}" if i < 9 else t["name"], t["name"])
            self.tag_combo.setItemData(i, QColor(self._tag_color(t["name"])), Qt.ForegroundRole)
        names = [t["name"] for t in tags]
        if cur in names:
            self.tag_combo.setCurrentIndex(names.index(cur))
        self.tag_combo.blockSignals(False)
        self._tag_changed()
        usage = self.ds.tag_usage() if self.ds else {}
        self.tag_usage_lbl.setText("\n".join(f"{n:<16} {usage.get(n, (0, 0))[0]:3d} seg · {usage.get(n, (0, 0))[1] / FPS:6.1f} s"
                                              for n in names) or "aucune étiquette : ＋ pour en créer")

    def _tag_changed(self, *_):
        name = self._current_tag()
        instr = next((t.get("instruction", "") for t in (self.ds.tags() if self.ds else []) if t["name"] == name), "")
        self.tag_instr_lbl.setText(f"consigne : {instr}" if instr else ("consigne : (celle de l'épisode)" if name else ""))

    def _segment_bands(self) -> list[tuple]:
        if not self.ds or not self.ep_name:
            return []
        return [(s["start"], s["end"], self._tag_color(s["tag"]), s["tag"]) for s in self.ds.segments(self.ep_name)]

    def _refresh_segments(self):
        self.seg_list.clear()
        if not self.ds or not self.ep_name:
            return
        for s in self.ds.segments(self.ep_name):
            it = QListWidgetItem(f"{s['tag']:<16} {s['start']:4d} → {s['end']:4d}   {(s['end'] - s['start'] + 1) / FPS:5.1f} s")
            it.setData(Qt.UserRole, s)
            it.setForeground(QColor(self._tag_color(s["tag"])))
            self.seg_list.addItem(it)
        bands = self._segment_bands()
        for ch in self.charts:
            ch.set_segments(bands)

    def _new_tag(self):
        if not self.ds:
            return
        d = TagDialog(self)
        if d.exec() != QDialog.Accepted:
            return
        try:
            self.ds.add_tag(d.name.text().strip(), d.instr.text())
        except ValueError as e:
            self.say(str(e), KO)
            return
        self.say(f"étiquette « {d.name.text().strip()} » créée", OK)
        self._refresh_tags(select=d.name.text().strip())

    def _rl_tags(self):
        if not self.ds:
            return
        new = self.ds.ensure_rl_tags()
        self.say(f"étiquettes RL ajoutées : {', '.join(new)}" if new else "étiquettes RL déjà présentes", OK)
        self._refresh_tags(select="bon")

    def _edit_tag(self):
        name = self._current_tag()
        if not name:
            return
        instr = next((t.get("instruction", "") for t in self.ds.tags() if t["name"] == name), "")
        d = TagDialog(self, name, instr)
        if d.exec() != QDialog.Accepted:
            return
        new = d.name.text().strip()
        if new != name:
            try:
                self.ds.rename_tag(name, new)
            except ValueError as e:
                self.say(str(e), KO)
                return
            self.say(f"étiquette « {name} » renommée « {new} » (segments compris)", OK)
            name = new
        self.ds.set_tag_instruction(name, d.instr.text())
        self.say(f"étiquette « {name} » : consigne « {d.instr.text().strip()} »", OK)
        self._refresh_tags(select=name)
        self._refresh_segments()

    def _del_tag(self):
        name = self._current_tag()
        if not name:
            return
        n = self.ds.tag_usage().get(name, (0, 0))[0]
        msg = (f"Supprimer l'étiquette « {name} » et ses {n} segment(s) dans tous les épisodes ?" if n
               else f"Supprimer l'étiquette « {name} » ?")
        if QMessageBox.question(self, "Supprimer l'étiquette", msg) != QMessageBox.Yes:
            return
        n = self.ds.remove_tag(name, with_segments=True)
        self.say(f"étiquette « {name} » supprimée" + (f", avec ses {n} segment(s)" if n else ""), OK)
        self._refresh_tags()
        self._refresh_segments()

    def _tag_segment(self):
        tag = self._current_tag()
        if not self.doc or not tag:
            self.say("étiqueter : créer / choisir d'abord une étiquette", WARN)
            return
        a = self.mark_in if self.mark_in is not None else 0
        b = self.mark_out if self.mark_out is not None else self.frame
        try:
            seg = self.ds.add_segment(self.ep_name, tag, a, b)
        except ValueError as e:
            self.say(f"étiqueter : {e}", KO)
            return
        self.say(f"🏷 {self.ep_name} : {tag} {seg['start']} → {seg['end']}", OK)
        nxt = seg["end"] + 1
        self.mark_in, self.mark_out = (nxt if nxt < self._n() else None), None
        self._update_marks()
        self._refresh_segments()
        self._refresh_tags()

    def _selected_segment(self) -> dict | None:
        it = self.seg_list.currentItem()
        return it.data(Qt.UserRole) if it else None

    def _goto_segment(self, it):
        s = it.data(Qt.UserRole)
        self.mark_in, self.mark_out = s["start"], s["end"]
        self._update_marks()
        self.seek(s["start"])

    def _chart_segment_action(self, i: int, action: str):
        segs = self.ds.segments(self.ep_name) if self.ds and self.ep_name else []
        if not 0 <= i < len(segs):
            return
        s = segs[i]
        if action == "delete":
            self.ds.remove_segment(self.ep_name, s)
            self.say(f"segment {s['tag']} {s['start']} → {s['end']} supprimé", OK)
            self._refresh_segments()
            self._refresh_tags()
        else:
            self.mark_in, self.mark_out = s["start"], s["end"]
            self._update_marks()
            self.seek(s["start"])

    def _delete_key(self):
        """Suppr : le segment sélectionné si la liste des segments a le focus, sinon les épisodes."""
        if self.seg_list.hasFocus():
            self._del_segment()
        else:
            self._delete()

    def _del_segment(self):
        s = self._selected_segment()
        if not s:
            self.say("supprimer un segment : cliquez d'abord dessus dans la liste", WARN)
            return
        self.ds.remove_segment(self.ep_name, s)
        self.say(f"segment {s['tag']} {s['start']} → {s['end']} supprimé", OK)
        self._refresh_segments()
        self._refresh_tags()

    def _export(self, mode: str):
        if not self.ds:
            return
        usage = self.ds.tag_usage()
        used = [t["name"] for t in self.ds.tags() if usage.get(t["name"], (0, 0))[0]]
        if not used:
            self.say("export : aucun segment étiqueté dans ce dataset", WARN)
            return
        task = self.ds.path.name
        jobs = ([([t], self.ds.path.parent / f"{task}__{t}") for t in used] if mode == "tag"
                else [(used, self.ds.path.parent / f"{task}__segments")])
        lines = "\n".join(f"  {out.name} : {sum(usage[t][0] for t in tags)} épisodes" for tags, out in jobs)
        if QMessageBox.question(self, "Exporter", f"Créer les sous-datasets suivants, à côté de « {task} » :\n\n{lines}\n\n"
                                "Un export précédent du même nom est remplacé. Images en liens physiques : "
                                "presque aucune place disque en plus.") != QMessageBox.Yes:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for tags, out in jobs:
                try:
                    k = self.ds.export_segments(tags, out)
                    self.say(f"export {out.name} : {k} épisodes", OK)
                except ValueError as e:
                    self.say(f"export {out.name} : {e}", KO)
        finally:
            QApplication.restoreOverrideCursor()
        cur = self.ds.path
        if self.root and self.root.resolve() != cur.resolve():
            self.task_combo.blockSignals(True)
            self.task_combo.clear()
            for t in find_tasks(self.root):
                self.task_combo.addItem(t.name, str(t))
            self.task_combo.setCurrentIndex(max(0, self.task_combo.findData(str(cur))))
            self.task_combo.blockSignals(False)
        self.say("les sous-datasets apparaissent dans le menu Tâche ; « Convertir au format WLA » marche dessus", ACCENT)

    def _convert(self):
        if not self.ds or (self.proc and self.proc.state() != QProcess.NotRunning):
            return
        task = self.ds.path.name
        out = REPO / "playground" / "Datasets" / f"g1d_{task}" / task
        py = REPO / ".venv" / "bin" / "python"
        if QMessageBox.question(self, "Convertir", f"Convertir « {task} » au format WLA ?\n\n-> {out}\n"
                                "(un dataset existant à cet endroit est remplacé)") != QMessageBox.Yes:
            return
        self.proc = QProcess(self)
        env = QProcessEnvironment()
        env.insert("HOME", str(Path.home()))
        env.insert("PATH", "/usr/bin:/bin")
        env.insert("LANG", "C.UTF-8")
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(REPO))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(
            lambda: [self.say(l) for l in bytes(self.proc.readAllStandardOutput()).decode(errors="replace").splitlines()
                     if l.strip() and "moov atom" not in l])
        self.proc.finished.connect(lambda code, _s: self.say(
            f"conversion terminée (code {code}) -> {out}" if code == 0 else f"conversion ÉCHOUÉE (code {code})",
            OK if code == 0 else KO))
        self.say(f"conversion de {task} -> {out}", ACCENT)
        self.proc.start(str(py), ["-m", "g1d_wla.convert_teleop", "--raw-dir", str(self.ds.path), "--out-dir", str(out),
                                  "--repo-id", f"innov8/g1d_{task}", "--overwrite"])


def main():
    start = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    w = Studio(start)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
