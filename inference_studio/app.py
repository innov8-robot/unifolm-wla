"""Inference Studio : choisir un modèle, lancer le serveur et la téléop en mode politique, piloter les essais.

    ~/Documents/project/manip/unifolm-wla/inference.sh        (active l'env g1d_teleop et lance l'appli)

Ce que fait l'application :

* SERVEUR : lance / arrête ``model_server.action_server_wbc_msgpack_unitree`` (env ``.venv`` du dépôt) sur le
  modèle choisi ; prêt quand il affiche « server listening on ».
* TÉLÉOP : lance / arrête ``teleop_hand_and_arm.py`` en mode politique avec ``--ipc`` (env g1d_teleop), la
  consigne et les options choisies ; ``adb reverse`` pour le casque (optionnel : sans casque, pas de
  correction au grip, tout le reste marche).
* PILOTAGE par IPC (mêmes effets que la manette) : activer le robot, lancer l'essai, réussi, raté, annuler,
  gravité zéro, garde, quitter. L'état (essai en cours, pas, gravité zéro, caméras, netteté, réussis / ratés)
  vient du heartbeat de la téléop.

L'arrêt d'urgence reste le bouton physique du robot.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import zmq
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTabWidget, QVBoxLayout,
                               QWidget)

from dataset_studio.theme import ACCENT, KO, MUTED, OK, QSS, WARN

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "playground" / "Checkpoints"
TELEOP_DIR = REPO / "teleoperation/Tele_OP/xr_teleoperate/teleop"
RAW = TELEOP_DIR / "utils/data"
SETTINGS = Path.home() / ".config" / "g1d_inference_studio.json"
PORT = 8600
IPC_DATA = "ipc://@xr_teleoperate_data.ipc"
IPC_HB = "ipc://@xr_teleoperate_hb.ipc"
DEFAULTS = {"network_interface": "enx0c3796e0bc5b", "img_server_ip": "192.168.123.164", "torso_pitch": 0.166,
            "torso_yaw": True, "base": True, "column": True, "record": True, "max_speed": 1.0}


EXTRA_QSS = """
QPushButton:disabled, QPushButton#Run:disabled, QPushButton#Danger:disabled {
    background: #10141b; color: #3c4553; border: 1px solid #1c2430; }
"""


def panel(title: str) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("Panel")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(12, 10, 12, 12)
    lay.setSpacing(8)
    h = QLabel(title.upper())
    h.setObjectName("H2")
    lay.addWidget(h)
    return f, lay


def big(text: str, kind: str | None = None, tip: str = "") -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    b.setMinimumHeight(54)
    b.setStyleSheet("font-size:15px; font-weight:600;")
    b.setToolTip(tip)
    return b


def model_instructions(run_dir: Path) -> list[str]:
    """Consignes des datasets qui ont servi à entraîner le modèle (config du run -> config de données ->
    tasks.parquet). Liste vide si introuvable."""
    try:
        import pyarrow.parquet as pq
        import yaml
        cfg = run_dir / "config.yaml" if (run_dir / "config.yaml").exists() else run_dir / "config.full.yaml"
        c = yaml.safe_load(cfg.read_text())
        dcfg = REPO / c["datasets"]["vla_data"]["data_config_path"]
        d = yaml.safe_load(dcfg.read_text())
        out = []
        for ds in d.get("datasets", []):
            root = REPO / d.get("data_base", "playground/Datasets") / ds["data_path"]
            for t in sorted(root.glob("*/meta/tasks.parquet")):
                tab = pq.read_table(t)                    # sans pandas (absent de l'env g1d_teleop)
                if "task" in tab.column_names:
                    out += [str(x) for x in tab.column("task").to_pylist() if str(x) not in out]
        return out
    except Exception:
        return []


class Studio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Inference Studio — G1-D")
        self.resize(1500, 900)
        self.cfg = dict(DEFAULTS)
        try:
            self.cfg.update(json.loads(SETTINGS.read_text()))
        except Exception:
            pass
        self.server = None
        self.server_ready = False
        self.teleop = None
        self.hb = {}
        self.hb_t = 0.0
        self.ctx = zmq.Context.instance()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub.setsockopt(zmq.RCVHWM, 10)
        self.sub.connect(IPC_HB)
        self._build()
        self._shortcuts()
        self._load_models()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(100)

    # ------------------------------------------------------------------ interface
    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        head = QFrame()
        head.setObjectName("Header")
        h = QHBoxLayout(head)
        h.setContentsMargins(14, 8, 14, 8)
        t = QLabel("INFERENCE STUDIO")
        t.setObjectName("Title")
        h.addWidget(t)
        h.addSpacing(20)
        self.pills = {}
        for key, txt in (("server", "Serveur"), ("teleop", "Téléop"), ("robot", "Robot"), ("trial", "Essai"),
                         ("zero_g", "Gravité zéro"), ("cams", "Caméras"), ("score", "Session")):
            l = QLabel(f"{txt} —")
            l.setStyleSheet(f"color:{MUTED}; padding:3px 10px; border:1px solid #263041; border-radius:10px")
            self.pills[key] = l
            h.addWidget(l)
        h.addStretch()
        v.addWidget(head)

        body = QSplitter(Qt.Horizontal)
        body.setContentsMargins(8, 8, 8, 8)
        v.addWidget(body, 1)

        # --- gauche : modèle, consigne, options, lancement
        left, ll = panel("1 · Modèle et lancement")
        self.models = QListWidget()
        self.models.setStyleSheet("QListWidget { font-size:14px; } QListWidget::item { padding:6px 4px; }")
        self.models.currentItemChanged.connect(self._model_changed)
        ll.addWidget(self.models, 1)
        ll.addWidget(QLabel("Consigne (instruction du modèle)"))
        self.instr = QComboBox()
        self.instr.setEditable(True)
        self.instr.setInsertPolicy(QComboBox.NoInsert)
        ll.addWidget(self.instr)
        orow = QGridLayout()
        self.cb_record = QCheckBox("Enregistrer les essais")
        self.cb_record.setChecked(self.cfg["record"])
        self.task_name = QLineEdit()
        self.task_name.setPlaceholderText("nom du dossier d'enregistrement")
        self.cb_yaw = QCheckBox("Rotation du buste")
        self.cb_yaw.setChecked(self.cfg["torso_yaw"])
        self.cb_base = QCheckBox("Base roulante")
        self.cb_base.setChecked(self.cfg["base"])
        self.cb_col = QCheckBox("Colonne")
        self.cb_col.setChecked(self.cfg["column"])
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.05, 2.0)
        self.speed.setSingleStep(0.05)
        self.speed.setValue(float(self.cfg["max_speed"]))
        self.speed.setPrefix("vitesse max ")
        self.speed.setSuffix(" m/s")
        orow.addWidget(self.cb_record, 0, 0)
        orow.addWidget(self.task_name, 0, 1)
        orow.addWidget(self.cb_yaw, 1, 0)
        orow.addWidget(self.cb_base, 1, 1)
        orow.addWidget(self.cb_col, 2, 0)
        orow.addWidget(self.speed, 2, 1)
        ll.addLayout(orow)
        self.b_server = big("Démarrer le serveur", "Run", "Charge le modèle sur le GPU (≈ 1 min)")
        self.b_server.clicked.connect(self._toggle_server)
        self.b_teleop = big("Lancer la téléop", None, "Téléop en mode politique, pilotée par cette application")
        self.b_teleop.clicked.connect(self._toggle_teleop)
        ll.addWidget(self.b_server)
        ll.addWidget(self.b_teleop)
        body.addWidget(left)

        # --- centre : pilotage
        center, cl = panel("2 · Pilotage")
        self.big_state = QLabel("Choisissez un modèle, démarrez le serveur, puis lancez la téléop.")
        self.big_state.setWordWrap(True)
        self.big_state.setAlignment(Qt.AlignCenter)
        self.big_state.setStyleSheet("font-size:20px; font-weight:600; padding:18px; background:#0a0d12; border-radius:6px")
        cl.addWidget(self.big_state)
        g = QGridLayout()
        g.setSpacing(10)
        self.b_enable = big("⏻  Activer le robot  [Entrée]", "Run", "Équivaut à « r » : les bras se mettent sous contrôle")
        self.b_trial = big("▶  Lancer l'essai  [Espace]", None, "Le modèle prend la main (A droit)")
        self.b_ok = big("✓  Réussi  [R]", None, "X gauche : essai réussi, enregistré")
        self.b_ko = big("✗  Raté  [E]", None, "Y gauche : essai raté, enregistré")
        self.b_cancel = big("⨯  Annuler  [Échap]", "Danger", "B droit : essai annulé, non enregistré")
        self.b_zero = big("🪶  Gravité zéro  [Z]", None, "Bras souples pour les placer à la main (hors essai)")
        self.b_guard = big("🛡  Garde  [G]", None, "Bras en position de garde (hors essai)")
        self.b_quit = big("⏹  Quitter la téléop  [Q]", "Danger", "Arrête la téléop : bras ramenés au repos lentement")
        self.b_enable.clicked.connect(lambda: self._cmd("CMD_START"))
        self.b_trial.clicked.connect(lambda: self._cmd("CMD_TRIAL"))
        self.b_ok.clicked.connect(lambda: self._cmd("CMD_SUCCESS"))
        self.b_ko.clicked.connect(lambda: self._cmd("CMD_FAILURE"))
        self.b_cancel.clicked.connect(lambda: self._cmd("CMD_CANCEL"))
        self.b_zero.clicked.connect(lambda: self._cmd("CMD_ZERO_G"))
        self.b_guard.clicked.connect(lambda: self._cmd("CMD_GUARD"))
        self.b_quit.clicked.connect(lambda: self._cmd("CMD_STOP"))
        g.addWidget(self.b_enable, 0, 0, 1, 2)
        g.addWidget(self.b_trial, 1, 0, 1, 2)
        g.addWidget(self.b_ok, 2, 0)
        g.addWidget(self.b_ko, 2, 1)
        g.addWidget(self.b_cancel, 3, 0, 1, 2)
        g.addWidget(self.b_zero, 4, 0)
        g.addWidget(self.b_guard, 4, 1)
        g.addWidget(self.b_quit, 5, 0, 1, 2)
        cl.addLayout(g)
        note = QLabel("Arrêt d'urgence : le bouton physique du robot. La manette et le casque restent utilisables "
                      "en même temps (grip = corriger).")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{MUTED}")
        cl.addWidget(note)
        cl.addStretch()
        body.addWidget(center)

        # --- droite : journaux
        right, rl = panel("3 · Journaux")
        self.tabs = QTabWidget()
        self.log_server = QPlainTextEdit()
        self.log_teleop = QPlainTextEdit()
        for w in (self.log_server, self.log_teleop):
            w.setReadOnly(True)
            w.setMaximumBlockCount(3000)
        self.tabs.addTab(self.log_teleop, "Téléop")
        self.tabs.addTab(self.log_server, "Serveur")
        rl.addWidget(self.tabs, 1)
        body.addWidget(right)
        body.setSizes([430, 560, 510])
        self.status = QLabel("")
        self.statusBar().addWidget(self.status, 1)

    def _shortcuts(self):
        for key, cmd in ((Qt.Key_Return, "CMD_START"), (Qt.Key_Space, "CMD_TRIAL"), (Qt.Key_R, "CMD_SUCCESS"),
                         (Qt.Key_E, "CMD_FAILURE"), (Qt.Key_Escape, "CMD_CANCEL"), (Qt.Key_Z, "CMD_ZERO_G"),
                         (Qt.Key_G, "CMD_GUARD"), (Qt.Key_Q, "CMD_STOP")):
            s = QShortcut(QKeySequence(key), self)
            s.activated.connect(lambda c=cmd: self._cmd(c))

    def say(self, msg: str, color: str | None = None):
        self.status.setText(msg)
        self.status.setStyleSheet(f"color:{color or '#cdd6e0'}; padding:2px 8px")

    # ------------------------------------------------------------------ modèles
    def _load_models(self):
        self.models.clear()
        runs = sorted((p for p in CKPT.glob("*") if (p / "final_model" / "model.safetensors").exists()),
                      key=lambda p: (p / "final_model" / "model.safetensors").stat().st_mtime, reverse=True)
        for r in runs:
            t = time.strftime("%d/%m %H:%M", time.localtime((r / "final_model" / "model.safetensors").stat().st_mtime))
            it = QListWidgetItem(f"{r.name}    ·  {t}")
            it.setData(Qt.UserRole, str(r))
            self.models.addItem(it)
        last = self.cfg.get("model")
        for i in range(self.models.count()):
            if self.models.item(i).data(Qt.UserRole) == last:
                self.models.setCurrentRow(i)
                return
        if self.models.count():
            self.models.setCurrentRow(0)

    def _model_changed(self, cur, _prev):
        if cur is None:
            return
        run = Path(cur.data(Qt.UserRole))
        tasks = model_instructions(run)
        self.instr.clear()
        self.instr.addItems(tasks)
        saved = self.cfg.get("instructions", {}).get(run.name)
        if saved:
            self.instr.setCurrentText(saved)
        self.task_name.setText(f"{run.name.removeprefix('g1d_')}_policy")
        self.say(f"{run.name} : {len(tasks)} consigne(s) d'entraînement trouvée(s)" if tasks else
                 f"{run.name} : consignes d'entraînement introuvables, tapez la consigne", None if tasks else WARN)

    def _run_dir(self) -> Path | None:
        it = self.models.currentItem()
        return Path(it.data(Qt.UserRole)) if it else None

    def _save_settings(self):
        run = self._run_dir()
        self.cfg.update(record=self.cb_record.isChecked(), torso_yaw=self.cb_yaw.isChecked(),
                        base=self.cb_base.isChecked(), column=self.cb_col.isChecked(),
                        max_speed=self.speed.value(), model=str(run) if run else None)
        if run:
            self.cfg.setdefault("instructions", {})[run.name] = self.instr.currentText()
        try:
            SETTINGS.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS.write_text(json.dumps(self.cfg, indent=1, ensure_ascii=False))
        except OSError:
            pass

    # ------------------------------------------------------------------ serveur
    def _toggle_server(self):
        if self.server and self.server.state() != QProcess.NotRunning:
            self.server.terminate()
            if not self.server.waitForFinished(5000):
                self.server.kill()
            return
        run = self._run_dir()
        if run is None:
            return
        busy = REPO / "playground" / ".gpu_busy"
        if busy.exists() and QMessageBox.question(
                self, "GPU occupé", f"Le GPU est réservé ({busy.read_text().strip()}) : un entraînement tourne sans "
                "doute, la mémoire risque de manquer.\nDémarrer quand même ?") != QMessageBox.Yes:
            return
        self._save_settings()
        self.server_ready = False
        self.server = QProcess(self)
        env = QProcessEnvironment()
        for k, v in (("HOME", str(Path.home())), ("PATH", "/usr/bin:/bin"), ("LANG", "C.UTF-8")):
            env.insert(k, v)
        self.server.setProcessEnvironment(env)
        self.server.setWorkingDirectory(str(REPO))
        self.server.setProcessChannelMode(QProcess.MergedChannels)
        self.server.readyReadStandardOutput.connect(self._server_out)
        self.server.finished.connect(lambda c, _s: self._proc_end("serveur", c))
        self.log_server.appendPlainText(f"▶ serveur : {run.name}")
        self.server.start(str(REPO / ".venv" / "bin" / "python"),
                          ["-m", "model_server.action_server_wbc_msgpack_unitree", "--ckpt_path",
                           str(run / "final_model" / "model.safetensors"), "--unnorm_key", "UnifoLM_G1_Dex1",
                           "--port", str(PORT)])
        self.tabs.setCurrentWidget(self.log_server)

    def _server_out(self):
        for line in bytes(self.server.readAllStandardOutput()).decode(errors="replace").splitlines():
            if not line.strip():
                continue
            self.log_server.appendPlainText(line)
            if "server listening on" in line:
                self.server_ready = True
                self.say("Serveur prêt", OK)

    # ------------------------------------------------------------------ téléop
    def _toggle_teleop(self):
        if self.teleop and self.teleop.state() != QProcess.NotRunning:
            self._cmd("CMD_STOP")
            if not self.teleop.waitForFinished(15000):
                self.teleop.terminate()
            return
        if not self.server_ready and QMessageBox.question(
                self, "Serveur", "Le serveur du modèle n'est pas prêt. Lancer la téléop quand même ?") != QMessageBox.Yes:
            return
        instr = self.instr.currentText().strip()
        if not instr:
            self.say("indiquez la consigne du modèle", KO)
            return
        self._save_settings()
        if shutil.which("adb"):
            r = QProcess()
            r.start("adb", ["reverse", "tcp:8012", "tcp:8012"])
            r.waitForFinished(4000)
            msg = bytes(r.readAllStandardOutput() + r.readAllStandardError()).decode(errors="replace").strip()
            self.log_teleop.appendPlainText(f"adb reverse : {'ok (casque branché)' if r.exitCode() == 0 else 'casque absent — ' + msg}")
        c = self.cfg
        args = ["teleop_hand_and_arm.py", f"--network-interface={c['network_interface']}",
                f"--img-server-ip={c['img_server_ip']}", "--input-mode=controller", "--arm=G1_29", "--ee=dex1",
                "--torso-pitch", str(c["torso_pitch"]), "--ipc", "--policy-uri", f"ws://127.0.0.1:{PORT}",
                "--policy-instruction", instr, "--policy-max-speed", f"{self.speed.value():.2f}"]
        if self.cb_yaw.isChecked():
            args += ["--torso-yaw-index", "12", "--torso-yaw-max", "1.0", "--torso-yaw-rate", "0.5"]
        if self.cb_base.isChecked():
            args.append("--base")
        if self.cb_col.isChecked():
            args.append("--column")
        if self.cb_record.isChecked():
            args += ["--record", f"--task-name={self.task_name.text().strip() or 'policy'}", f"--task-goal={instr}"]
        self.teleop = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("COLUMNS", "200")                     # journaux de la téléop sur des lignes larges
        self.teleop.setProcessEnvironment(env)
        self.teleop.setWorkingDirectory(str(TELEOP_DIR))
        self.teleop.setProcessChannelMode(QProcess.MergedChannels)
        self.teleop.readyReadStandardOutput.connect(self._teleop_out)
        self.teleop.finished.connect(lambda c, _s: self._proc_end("téléop", c))
        self.log_teleop.appendPlainText("▶ python " + " ".join(args))
        self.teleop.start(sys.executable, args)
        self.tabs.setCurrentWidget(self.log_teleop)

    def _teleop_out(self):
        txt = bytes(self.teleop.readAllStandardOutput()).decode(errors="replace")
        for line in txt.splitlines():
            if line.strip():
                self.log_teleop.appendPlainText(line.rstrip())
            low = line.lower()
            if "channel factory init error" in low or "does not match an available interface" in low:
                self.say("Robot introuvable sur le réseau : câble Ethernet du robot branché ? robot allumé ?", KO)
            elif "traceback" in low:
                self.say("La téléop a rencontré une erreur : voir le journal Téléop", KO)
            elif "caméra" in low and ("absente" in low or "floue" in low):
                self.say(line.split("📷")[-1].strip()[:120], WARN)

    def _proc_end(self, name: str, code: int):
        self.say(f"{name} arrêté (code {code})", OK if code == 0 else WARN)
        if name == "serveur":
            self.server_ready = False
        else:
            self.hb = {}

    # ------------------------------------------------------------------ IPC
    def _cmd(self, cmd: str):
        if not self.teleop or self.teleop.state() == QProcess.NotRunning:
            self.say("la téléop n'est pas lancée", WARN)
            return
        s = self.ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, 1500)
        s.setsockopt(zmq.SNDTIMEO, 1500)
        try:
            s.connect(IPC_DATA)
            s.send_json({"reqid": uuid.uuid4().hex[:8], "cmd": cmd})
            rep = s.recv_json()
            ok = rep.get("status") == "ok"
            labels = {"CMD_START": "robot activé", "CMD_TRIAL": "essai lancé / arrêté", "CMD_SUCCESS": "essai réussi",
                      "CMD_FAILURE": "essai raté", "CMD_CANCEL": "essai annulé", "CMD_ZERO_G": "gravité zéro",
                      "CMD_GUARD": "garde", "CMD_STOP": "arrêt de la téléop"}
            self.say(labels.get(cmd, cmd) if ok else f"{cmd} refusé : {rep.get('msg')}", OK if ok else KO)
        except zmq.Again:
            self.say("la téléop ne répond pas (encore en démarrage ?)", WARN)
        finally:
            s.close()

    def _poll(self):
        try:
            while True:
                self.hb = self.sub.recv_json(flags=zmq.NOBLOCK)
                self.hb_t = time.time()
        except zmq.Again:
            pass
        teleop_on = bool(self.teleop and self.teleop.state() != QProcess.NotRunning)
        alive = teleop_on and time.time() - self.hb_t < 2.0
        hb = self.hb if alive else {}
        server_on = bool(self.server and self.server.state() != QProcess.NotRunning)
        self._pill("server", "Serveur prêt" if self.server_ready else ("Serveur : chargement…" if server_on else "Serveur arrêté"),
                   OK if self.server_ready else WARN if server_on else MUTED)
        self._pill("teleop", "Téléop connectée" if alive else ("Téléop : démarrage…" if teleop_on else "Téléop arrêtée"),
                   OK if alive else WARN if teleop_on else MUTED)
        self._pill("robot", "Robot actif" if hb.get("START") else "Robot inactif", OK if hb.get("START") else MUTED)
        trial = hb.get("RECORD_RUNNING")
        self._pill("trial", f"Essai en cours · {hb.get('EPISODE_STEPS', 0)} pas" if trial else "Pas d'essai",
                   ACCENT if trial else MUTED)
        self._pill("zero_g", "Gravité zéro" if hb.get("ZERO_G") else "Bras tenus", WARN if hb.get("ZERO_G") else MUTED)
        miss = hb.get("CAM_MISSING") or []
        sharp = hb.get("SHARP")
        blurry = bool(sharp and sharp[1] >= 60 and sharp[0] < 0.5 * sharp[1])
        cam_txt = (f"Caméra absente : {', '.join(miss)}" if miss else "Caméra de tête floue" if blurry
                   else "Caméras OK" if alive else "Caméras —")
        self._pill("cams", cam_txt, KO if miss or blurry else OK if alive else MUTED)
        self._pill("score", f"✓ {hb.get('N_OK', 0)} · ✗ {hb.get('N_KO', 0)}", MUTED)
        # gros état + boutons utiles
        if not teleop_on:
            msg = ("Serveur prêt : lancez la téléop." if self.server_ready else
                   "Démarrez le serveur, puis lancez la téléop." if not server_on else "Chargement du modèle…")
        elif not alive:
            msg = "Démarrage de la téléop…"
        elif not hb.get("START"):
            msg = "Téléop prête : activez le robot (Entrée)."
        elif trial:
            msg = f"ESSAI EN COURS — {hb.get('EPISODE_STEPS', 0)} pas\nRéussi (R) · Raté (E) · Annuler (Échap)"
        elif hb.get("ZERO_G"):
            msg = "GRAVITÉ ZÉRO — placez les bras à la main, puis Z pour les tenir."
        else:
            last = {"success": "dernier essai : réussi", "failure": "dernier essai : raté"}.get(hb.get("LAST_OUTCOME"), "")
            msg = "Prêt — Espace pour lancer l'essai" + (f"\n{last}" if last else "")
        self.big_state.setText(msg)
        self.b_server.setText("Arrêter le serveur" if server_on else "Démarrer le serveur")
        self.b_teleop.setText("Arrêter la téléop" if teleop_on else "Lancer la téléop")
        self.b_enable.setEnabled(alive and not hb.get("START"))
        for b in (self.b_trial, self.b_quit):
            b.setEnabled(alive)
        for b in (self.b_ok, self.b_ko, self.b_cancel):
            b.setEnabled(bool(alive and trial))
        for b in (self.b_zero, self.b_guard):
            b.setEnabled(bool(alive and hb.get("START") and not trial))
        self.b_trial.setText("⏹  Arrêter l'essai (sans résultat)  [Espace]" if trial else "▶  Lancer l'essai  [Espace]")
        self.b_zero.setText("🪶  Tenir les bras  [Z]" if hb.get("ZERO_G") else "🪶  Gravité zéro  [Z]")

    def _pill(self, key: str, text: str, color: str):
        l = self.pills[key]
        l.setText(text)
        l.setMinimumWidth(l.fontMetrics().horizontalAdvance(text) + 26)
        l.setStyleSheet(f"color:{color}; padding:3px 10px; border:1px solid {color if color != MUTED else '#263041'}; "
                        f"border-radius:10px")

    def closeEvent(self, e):
        running = [n for n, p in (("la téléop", self.teleop), ("le serveur", self.server))
                   if p and p.state() != QProcess.NotRunning]
        if running and QMessageBox.question(self, "Quitter", f"Arrêter {' et '.join(running)} ?") != QMessageBox.Yes:
            e.ignore()
            return
        if self.teleop and self.teleop.state() != QProcess.NotRunning:
            self._cmd("CMD_STOP")
            if not self.teleop.waitForFinished(15000):
                self.teleop.terminate()
        if self.server and self.server.state() != QProcess.NotRunning:
            self.server.terminate()
            self.server.waitForFinished(5000)
        self._save_settings()
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS + EXTRA_QSS)
    w = Studio()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
