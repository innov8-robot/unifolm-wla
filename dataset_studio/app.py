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

import numpy as np
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QColor, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QSlider, QSplitter, QTableWidget, QTableWidgetItem,
                               QToolButton, QVBoxLayout, QWidget)

from .charts import SignalChart
from .timeline import Timeline
from .model import (CAMERA_LABELS, RL_TAGS, TaskDataset, find_tasks, guess_layout, load_doc, load_segments,
                    signals, summarize)
from .theme import ACCENT, KO, MUTED, OK, QSS, TAG_COLORS, WARN

GRIPPERS_GROUP = "Pinces (Dex1, 0 fermée → 5,4)"
FILTERS = ["Tous les épisodes", "Sans résultat", "Réussis", "Ratés", "Non découpés", "À vérifier (⚠)"]
HELP_TEXT = """LECTURE
  Espace : lecture / pause        ← → : pas à pas        Maj + ← → : ± 1 s
  ↑ ↓ : épisode précédent / suivant
  Frise : clic = aller à ce pas · glisser = sélectionner une plage

SÉLECTION (frise)
  I / O : début / fin au pas courant      Échap : effacer la sélection
  ✂ Rogner : ne garder que la sélection (l'original va à la corbeille)

RÉSULTAT DE L'ESSAI
  R : réussi      E : raté      N : inconnu      (re-cliquer le bouton actif = effacer)

SOUS-TÂCHES
  1 à 9 : choisir l'étiquette      T : étiqueter la sélection (ou du début jusqu'au pas courant)
  Le début suivant se place juste après : I une fois, puis chiffre + T à chaque sous-tâche.
  Clic droit sur un segment (frise ou graphique) : supprimer / reprendre ses bornes

ÉPISODES
  Ctrl / Maj + clic : sélection multiple      Suppr : supprimer (corbeille, renumérotation)
  Filtre en haut de la liste : sans résultat, ratés, non découpés, à vérifier

Les touches ne marchent pas pendant la saisie dans un champ de texte : cliquez sur la vidéo d'abord."""

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


class Section(QWidget):
    """Bloc à en-tête cliquable (▾ / ▸) : replier ce qu'on n'utilise pas."""

    def __init__(self, title: str, collapsed: bool = False):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.head = QToolButton()
        self.head.setObjectName("SectionHead")
        self.head.setText(title.upper())
        self.head.setCheckable(True)
        self.head.setChecked(not collapsed)
        self.head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.head.setArrowType(Qt.DownArrow if not collapsed else Qt.RightArrow)
        self.head.setStyleSheet("QToolButton#SectionHead { border:none; color:#8b949e; font-size:11px; "
                                "letter-spacing:1.5px; font-weight:600; padding:4px 0; }")
        self.body = QFrame()
        self.body.setObjectName("Panel")
        self.inner = QVBoxLayout(self.body)
        self.inner.setContentsMargins(10, 8, 10, 10)
        self.inner.setSpacing(6)
        self.body.setVisible(not collapsed)
        self.head.toggled.connect(self._toggle)
        lay.addWidget(self.head)
        lay.addWidget(self.body)

    def _toggle(self, on: bool):
        self.body.setVisible(on)
        self.head.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

    def __iter__(self):                    # permet « sec, sl = Section(...) »
        return iter((self, self.inner))


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
        # en-tête : tâche, résumé, outils
        head = QFrame()
        head.setObjectName("Header")
        h = QHBoxLayout(head)
        h.setContentsMargins(14, 8, 14, 8)
        t = QLabel("DATASET STUDIO")
        t.setObjectName("Title")
        h.addWidget(t)
        h.addSpacing(18)
        h.addWidget(QLabel("Tâche"))
        self.task_combo = QComboBox()
        self.task_combo.setMinimumWidth(240)
        self.task_combo.currentIndexChanged.connect(self._task_changed)
        h.addWidget(self.task_combo)
        h.addSpacing(12)
        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet(f"color:{MUTED}")
        h.addWidget(self.summary_lbl, 1)
        b_obj = button("Objets…", "Quiet", "Encadrer un objet, détecter ses instances, choisir source / cible, suivre (optionnel)")
        b_obj.clicked.connect(self._objects)
        b_reload = button("Recharger", "Quiet")
        b_reload.clicked.connect(lambda: self.load_task(self.ds.path, select=self.ep_name) if self.ds else None)
        b_open = button("Ouvrir…", "Quiet")
        b_open.clicked.connect(self._choose_dir)
        b_help = button("?  Aide", None, "Raccourcis clavier et mode d'emploi (touche F1 ou ?)")
        b_help.clicked.connect(self._help)
        for w in (b_obj, b_reload, b_open, b_help):
            h.addWidget(w)
        v.addWidget(head)

        body = QSplitter(Qt.Horizontal)
        body.setContentsMargins(8, 8, 8, 8)
        v.addWidget(body, 1)

        # --- gauche : épisodes
        left, ll = panel("Épisodes")
        frow = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("n° …")
        self.search.setMaximumWidth(80)
        self.search.textChanged.connect(self._apply_filter)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(FILTERS)
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)
        frow.addWidget(self.search)
        frow.addWidget(self.filter_combo, 1)
        ll.addLayout(frow)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["N°", "DURÉE", "RÉSULTAT", "ÉTAT"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        for c in range(3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        ll.addWidget(self.table, 1)
        legend = QLabel("✓ réussi · ✗ raté · ? inconnu    ▦ découpé · ◎ objets · ⚠ à vérifier")
        legend.setStyleSheet(f"color:{MUTED}; font-size:11px")
        ll.addWidget(legend)
        row = QHBoxLayout()
        self.b_delete = button("🗑 Supprimer", "Danger", "Met les épisodes sélectionnés à la corbeille et renumérote (Suppr)")
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

        # --- centre : vidéo (tête en grand, poignets dessous), lecture, frise, graphiques
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cams_f, cams_l = panel("Vidéo")
        self.cam_head = CamView("Tête")
        self.cam_wl, self.cam_wr = CamView("Poignet gauche"), CamView("Poignet droit")
        cams_l.addWidget(self.cam_head, 3)
        wr = QHBoxLayout()
        wr.addWidget(self.cam_wl)
        wr.addWidget(self.cam_wr)
        cams_l.addLayout(wr, 2)
        tr = QHBoxLayout()
        self.b_first = button("⏮", "Quiet", "Début de l'épisode")
        self.b_back = button("◀", "Quiet", "Pas précédent (←)")
        self.b_play = button("▶  Lecture", "Run", "Lecture / pause (Espace)")
        self.b_fwd = button("▶", "Quiet", "Pas suivant (→)")
        self.b_last = button("⏭", "Quiet", "Fin de l'épisode")
        self.b_first.clicked.connect(lambda: self.seek(0))
        self.b_back.clicked.connect(lambda: self.seek(self.frame - 1))
        self.b_fwd.clicked.connect(lambda: self.seek(self.frame + 1))
        self.b_last.clicked.connect(lambda: self.seek(self._n() - 1))
        self.b_play.clicked.connect(self.toggle_play)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["×0.25", "×0.5", "×1", "×2", "×4"])
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        self.frame_lbl = QLabel("—")
        self.frame_lbl.setObjectName("Mono")
        for w in (self.b_first, self.b_back, self.b_play, self.b_fwd, self.b_last, self.speed_combo):
            tr.addWidget(w)
        tr.addStretch()
        tr.addWidget(self.frame_lbl)
        cams_l.addLayout(tr)
        self.timeline = Timeline()
        self.timeline.seek.connect(self.seek)
        self.timeline.selection.connect(self._timeline_selection)
        self.timeline.segment_action.connect(self._chart_segment_action)
        cams_l.addWidget(self.timeline)
        cv.addWidget(cams_f, 3)
        charts_f, charts_l = panel("Signaux")
        self.charts, self.chart_combos, self.chart_rows = [], [], []
        for i, default in enumerate((GRIPPERS_GROUP, "Bras droit — état")):
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
            if i == 0:
                self.b_chart2 = button("＋ 2ᵉ graphique", "Quiet", "Afficher un second graphique (angles des bras, base, colonne…)")
                self.b_chart2.clicked.connect(self._toggle_chart2)
                r.addWidget(self.b_chart2)
            holder = QWidget()
            hl = QVBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addLayout(r)
            hl.addWidget(ch, 1)
            charts_l.addWidget(holder, 1)
            self.charts.append(ch)
            self.chart_combos.append(cb)
            self.chart_rows.append(holder)
        self.chart_rows[1].setVisible(False)
        cv.addWidget(charts_f, 2)
        body.addWidget(center)

        # --- droite : sections repliables
        rpanel = QWidget()
        rl = QVBoxLayout(rpanel)
        rl.setContentsMargins(4, 0, 4, 0)
        rl.setSpacing(6)
        # ÉPISODE : résultat + consigne
        sec, sl = Section("Épisode")
        self.ep_lbl = QLabel("—")
        self.ep_lbl.setObjectName("Title")
        sl.addWidget(self.ep_lbl)
        sl.addWidget(QLabel("Résultat de l'essai"))
        orow = QHBoxLayout()
        self.out_btns = {}
        for oc, txt, key, tip in (("success", "✓ Réussi", "R", "Essai réussi (R)"),
                                  ("failure", "✗ Raté", "E", "Essai raté (E)"),
                                  ("unknown", "? Inconnu", "N", "Résultat inconnu (N)")):
            b = button(f"{txt}  [{key}]", None, tip)
            b.setCheckable(True)
            b.clicked.connect(lambda _c=False, o=oc: self._set_outcome(o))
            self.out_btns[oc] = b
            orow.addWidget(b)
        sl.addLayout(orow)
        sl.addSpacing(4)
        sl.addWidget(QLabel("Consigne (instruction du modèle)"))
        self.goal_edit = QLineEdit()
        sl.addWidget(self.goal_edit)
        gr = QHBoxLayout()
        gr.addWidget(QLabel("Appliquer à :"))
        b_goal1 = button("cet épisode", None)
        b_goal_sel = button("la sélection", None, "Les épisodes sélectionnés dans la liste (Ctrl / Maj + clic)")
        b_goal_all = button("tous…", "Danger", "Tous les épisodes de la tâche (confirmation demandée)")
        b_goal1.clicked.connect(lambda: self._set_goal("one"))
        b_goal_sel.clicked.connect(lambda: self._set_goal("sel"))
        b_goal_all.clicked.connect(lambda: self._set_goal("all"))
        for b in (b_goal1, b_goal_sel, b_goal_all):
            gr.addWidget(b)
        sl.addLayout(gr)
        rl.addWidget(sec)
        # SÉLECTION SUR LA FRISE : rogner ou étiqueter
        sec, sl = Section("Sélection sur la frise")
        self.trim_lbl = QLabel("glissez sur la frise, ou I / O")
        self.trim_lbl.setObjectName("Mono")
        sl.addWidget(self.trim_lbl)
        trow = QHBoxLayout()
        b_in = button("Début  [I]", None)
        b_out2 = button("Fin  [O]", None)
        b_clear = button("Effacer [Échap]", None)
        b_in.clicked.connect(self.set_in)
        b_out2.clicked.connect(self.set_out)
        b_clear.clicked.connect(self.clear_marks)
        for b in (b_in, b_out2, b_clear):
            trow.addWidget(b)
        sl.addLayout(trow)
        arow = QHBoxLayout()
        self.b_tag = button("🏷 Étiqueter  [T]", None, "Crée un segment avec l'étiquette choisie ; le début suivant est placé juste après")
        self.b_tag.clicked.connect(self._tag_segment)
        self.b_trim = button("✂ Rogner l'épisode", "Danger", "Ne garder que la sélection ; l'original va dans la corbeille")
        self.b_trim.clicked.connect(self._trim)
        arow.addWidget(self.b_tag, 1)
        arow.addWidget(self.b_trim)
        sl.addLayout(arow)
        rl.addWidget(sec)
        # SOUS-TÂCHES
        sec, sl = Section("Sous-tâches (étiquettes)")
        tgrow = QHBoxLayout()
        self.tag_combo = QComboBox()
        self.tag_combo.setMinimumWidth(170)
        self.tag_combo.currentIndexChanged.connect(self._tag_changed)
        b_tag_new = button("＋", None, "Nouvelle étiquette (nom + consigne du modèle)")
        b_tag_edit = button("✎", None, "Renommer l'étiquette ou modifier sa consigne")
        b_tag_del = button("−", None, "Supprimer l'étiquette (et ses segments, après confirmation)")
        b_tag_new.clicked.connect(self._new_tag)
        b_tag_edit.clicked.connect(self._edit_tag)
        b_tag_del.clicked.connect(self._del_tag)
        tgrow.addWidget(self.tag_combo, 1)
        for b in (b_tag_new, b_tag_edit, b_tag_del):
            b.setFixedWidth(40)
            tgrow.addWidget(b)
        sl.addLayout(tgrow)
        self.tag_instr_lbl = QLabel("")
        self.tag_instr_lbl.setWordWrap(True)
        self.tag_instr_lbl.setStyleSheet(f"color:{MUTED}")
        sl.addWidget(self.tag_instr_lbl)
        self.seg_list = QListWidget()
        self.seg_list.setMaximumHeight(110)
        self.seg_list.itemDoubleClicked.connect(self._goto_segment)
        self.seg_list.setToolTip("Double-clic : aller au segment · Suppr : supprimer le segment sélectionné")
        sl.addWidget(self.seg_list)
        srow = QHBoxLayout()
        b_seg_del = button("Supprimer le segment", None)
        b_seg_del.clicked.connect(self._del_segment)
        b_rl = button("＋ bon / mauvais", None,
                      "Étiquettes d'avantage manuel : remplacent le jugement du modèle de valeur de RECAP")
        b_rl.clicked.connect(self._rl_tags)
        srow.addWidget(b_seg_del)
        srow.addStretch()
        srow.addWidget(b_rl)
        sl.addLayout(srow)
        self.tag_usage_lbl = QLabel("")
        self.tag_usage_lbl.setWordWrap(True)
        self.tag_usage_lbl.setObjectName("Mono")
        sl.addWidget(self.tag_usage_lbl)
        erow = QHBoxLayout()
        b_exp_tag = button("Exporter par étiquette…", None, "Un dossier de tâche par étiquette : <tâche>__<étiquette>")
        b_exp_all = button("Exporter tout…", None, "Un seul dossier <tâche>__segments, une consigne par étiquette")
        b_exp_tag.clicked.connect(lambda: self._export("tag"))
        b_exp_all.clicked.connect(lambda: self._export("all"))
        erow.addWidget(b_exp_tag)
        erow.addWidget(b_exp_all)
        sl.addLayout(erow)
        rl.addWidget(sec)
        # DÉTAILS et JOURNAL : repliés
        sec, sl = Section("Détails du fichier", collapsed=True)
        self.info_text = QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(180)
        sl.addWidget(self.info_text)
        rl.addWidget(sec)
        sec, sl = Section("Journal", collapsed=True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMinimumHeight(160)
        sl.addWidget(self.log)
        rl.addWidget(sec)
        rl.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(rpanel)
        scroll.setMinimumWidth(470)
        body.addWidget(scroll)
        body.setSizes([290, 840, 470])
        self.status = QLabel("")
        self.statusBar().addWidget(self.status, 1)
        self._msg_timer = QTimer(self)
        self._msg_timer.setSingleShot(True)
        self._msg_timer.timeout.connect(lambda: self.status.setText(""))

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
        sc(Qt.Key_Escape, self.clear_marks)
        sc(Qt.Key_T, self._tag_segment)
        sc(Qt.Key_R, lambda: self._set_outcome("success"))
        sc(Qt.Key_E, lambda: self._set_outcome("failure"))
        sc(Qt.Key_N, lambda: self._set_outcome("unknown"))
        sc(Qt.Key_F1, self._help)
        sc("?", self._help)
        for k in range(1, 10):
            sc(str(k), lambda k=k: self.tag_combo.setCurrentIndex(k - 1) if k <= self.tag_combo.count() else None)
        sc(Qt.Key_Delete, self._delete_key)

    def _help(self):
        QMessageBox.information(self, "Aide — Dataset Studio", HELP_TEXT)

    def _toggle_chart2(self):
        vis = not self.chart_rows[1].isVisible()
        self.chart_rows[1].setVisible(vis)
        self.b_chart2.setText("− 2ᵉ graphique" if vis else "＋ 2ᵉ graphique")

    def _timeline_selection(self, a, b):
        self.mark_in, self.mark_out = a, b
        self._update_marks()

    # ------------------------------------------------------------------ chargement
    def say(self, msg: str, color: str | None = None):
        self.log.appendHtml(f'<span style="color:{color or "#cdd6e0"}">{msg}</span>')
        self.status.setText(msg)
        self.status.setStyleSheet(f"color:{color or '#cdd6e0'}; padding:2px 8px;")
        self._msg_timer.start(7000)

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
        self.task_combo.setToolTip(str(path))
        sums = self.ds.summaries()
        self.table.setRowCount(len(sums))
        tot, n_ok, n_ko, n_none = 0, 0, 0, 0
        for r, s_ in enumerate(sums):
            self._fill_row(r, s_)
            tot += s_.n
            n_ok += s_.outcome == "success"
            n_ko += s_.outcome == "failure"
            n_none += not s_.outcome
        self._refresh_tags()
        n_tr = len(self.ds.trashed())
        self.summary_lbl.setText(f"{len(sums)} épisodes · {tot / FPS / 60:.1f} min · ✓ {n_ok} · ✗ {n_ko} · sans résultat {n_none}"
                                 + (f" · corbeille {n_tr}" if n_tr else ""))
        self._apply_filter()
        if sums:
            row = 0
            if select:
                for r in range(self.table.rowCount()):
                    if self.table.item(r, 0).data(Qt.UserRole) == select:
                        row = r
            if self.table.item(row, 0).data(Qt.UserRole) == self.ep_name:
                self.table.blockSignals(True)
                self.table.selectRow(row)
                self.table.blockSignals(False)
            else:
                self.table.selectRow(row)
        else:
            self.doc = None

    def _fill_row(self, r: int, s_):
        ep = self.ds.path / s_.name
        nseg = len(load_segments(ep))
        has_obj = (ep / "objects").is_dir() and any((ep / "objects").glob("*_tracks.npz"))
        warn = bool(s_.error or s_.missing_images or "< 1 s" in s_.flags or "VIDE" in s_.flags)
        res = {"success": "✓", "failure": "✗", "unknown": "?"}.get(s_.outcome, "·")
        state = " ".join(x for x in (f"▦ {nseg}" if nseg else "", "◎" if has_obj else "", "⚠" if warn else "") if x)
        tip = (f"{s_.name} · {s_.n} pas · {s_.seconds:.1f} s · pinces G/D "
               f"{'●' if s_.left_gripper_used else '○'}/{'●' if s_.right_gripper_used else '○'}"
               + (f"\nindicateurs : {', '.join(s_.flags)}" if s_.flags else "")
               + (f"\n{s_.error}" if s_.error else "")
               + (f"\n{nseg} segment(s)" if nseg else ""))
        cells = [s_.name.replace("episode_", ""), f"{s_.seconds:5.1f} s", res, state]
        for c, txt in enumerate(cells):
            it = QTableWidgetItem(txt)
            it.setToolTip(tip)
            if c == 0:
                it.setData(Qt.UserRole, s_.name)
                it.setData(Qt.UserRole + 1, {"outcome": s_.outcome, "nseg": nseg, "warn": warn})
            if c == 2:
                it.setTextAlignment(Qt.AlignCenter)
                it.setForeground(QColor({"✓": OK, "✗": KO, "?": WARN}.get(res, MUTED)))
            if c == 3 and warn:
                it.setForeground(QColor(KO))
            self.table.setItem(r, c, it)

    def _refresh_row(self, name: str):
        """Met à jour une seule ligne (après un changement de résultat, de segments…)."""
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).data(Qt.UserRole) == name:
                self.table.blockSignals(True)
                self._fill_row(r, summarize(self.ds.path / name))
                self.table.blockSignals(False)
                break
        self._apply_filter()

    def _apply_filter(self, *_):
        f = self.filter_combo.currentIndex() if hasattr(self, "filter_combo") else 0
        q = self.search.text().strip().lstrip("0") if hasattr(self, "search") else ""
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            meta = it.data(Qt.UserRole + 1) or {}
            ok = {0: True, 1: not meta.get("outcome"), 2: meta.get("outcome") == "success",
                  3: meta.get("outcome") == "failure", 4: not meta.get("nseg"), 5: meta.get("warn")}[f]
            if q:
                ok = ok and it.text().lstrip("0").startswith(q)
            self.table.setRowHidden(r, not ok)

    def select_episode(self, name: str) -> bool:
        """Sélectionne un épisode dans la liste SANS recharger la tâche (qui relit tous les data.json)."""
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is not None and it.data(Qt.UserRole) == name:
                if name != self.ep_name:
                    self.table.selectRow(r)
                return True
        return False

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
        self._show_outcome(self.doc.get("info", {}).get("outcome"))
        keys = sorted(self.doc["data"][0]["colors"]) if self.doc["data"] else []
        labels = CAMERA_LABELS.get(self.layout_name, {})
        # tête : l'œil GAUCHE seulement (celui que voit le modèle) ; l'œil droit n'est plus affiché
        head = next((k for k in keys if labels.get(k, "").startswith("Tête") and "droit" not in labels.get(k, "")),
                    keys[0] if keys else None)
        wl = next((k for k in keys if labels.get(k) == "Poignet gauche"), None)
        wrr = next((k for k in keys if labels.get(k) == "Poignet droit"), None)
        self.cam_keys = [(self.cam_head, head, "Tête (œil gauche, vue du modèle)"),
                         (self.cam_wl, wl, "Poignet gauche"), (self.cam_wr, wrr, "Poignet droit")]
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
        inter = [st.get("intervention", 0) for st in self.doc["data"]]
        self.timeline.set_episode(self._n(), self._segment_bands(),
                                  np.asarray(inter, float) if any(inter) else None)
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
        for cam, key, title in getattr(self, "cam_keys", []):
            cam.setVisible(key is not None)
            if key is not None:
                cam.show_image(self.ds.path / self.ep_name / st["colors"][key], title)
        self.timeline.set_cursor(self.frame)
        self.frame_lbl.setText(f"pas {self.frame:4d} / {n - 1}    {self.frame / FPS:6.2f} s / {(n - 1) / FPS:.1f} s")
        if self.obj_dlg is not None and self.obj_dlg.isVisible():
            self.obj_dlg.refresh()
        for ch in self.charts:
            ch.set_cursor(self.frame)

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
        if self.mark_in is None and self.mark_out is None:
            self.trim_lbl.setText("aucune — glissez sur la frise, ou I / O")
        else:
            a = "début" if self.mark_in is None else f"{self.mark_in}"
            b = "pas courant" if self.mark_out is None else f"{self.mark_out}"
            dur = (f"  ({(self.mark_out - self.mark_in + 1) / FPS:.1f} s)"
                   if self.mark_in is not None and self.mark_out is not None else "")
            self.trim_lbl.setText(f"{a} → {b}{dur}")
        for ch in self.charts:
            ch.set_marks(self.mark_in, self.mark_out)
        self.timeline.set_marks(self.mark_in, self.mark_out)

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

    def _show_outcome(self, oc):
        for k, b in self.out_btns.items():
            b.setChecked(k == oc)

    def _set_outcome(self, oc: str):
        """Clic ou touche R / E / N : appliqué tout de suite ; re-cliquer le résultat actif l'efface."""
        if not self.ep_name or not self.doc:
            return
        cur = self.doc.get("info", {}).get("outcome")
        new = None if cur == oc else oc
        self.ds.set_outcome(self.ep_name, new)
        self.doc.setdefault("info", {})["outcome"] = new
        self._show_outcome(new)
        self._refresh_row(self.ep_name)
        lab = {"success": "réussi", "failure": "raté", "unknown": "inconnu"}
        self.say(f"{self.ep_name} : résultat {lab[new] if new else 'effacé'}", OK)

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
        self.timeline.set_segments(bands)

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
        if n:
            self.load_task(self.ds.path, select=self.ep_name)    # pastilles ▦ de tous les épisodes
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
        self._refresh_row(self.ep_name)
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
            self._refresh_row(self.ep_name)
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
        self._refresh_row(self.ep_name)

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
