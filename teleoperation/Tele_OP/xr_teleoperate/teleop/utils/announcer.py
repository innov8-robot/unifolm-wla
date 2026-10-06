"""Annonces vocales de la téléop (enregistrement lancé, sauvegardé, annulé, pause...).

* ``pc``    : synthèse du PC, voix NEURONALE Piper si installée (``pip install piper-tts`` + un modèle
  dans ``~/.local/share/piper-voices/``, voix choisie par ``$TELEOP_VOICE``, défaut fr_FR-siwis-medium),
  sinon ``spd-say`` (repli ``espeak-ng``) ; non bloquante, phrases mises en cache ;
* ``robot`` : haut-parleur du robot, service « voice » d'Unitree (TtsMaker), en anglais ;
* ``off``   : rien.
Une annonce qui échoue ne fait jamais planter la téléop.
"""
import io
import os
import queue
import shutil
import subprocess
import threading
import wave
from pathlib import Path

import logging_mp

logger_mp = logging_mp.getLogger(__name__)


class Announcer:
    def __init__(self, mode: str = "pc", robot_speaker_id: int = 1):
        self.mode = mode
        self.client = None
        self.speaker_id = robot_speaker_id
        if mode == "robot":
            try:
                from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
                self.client = AudioClient()
                self.client.SetTimeout(2.0)
                self.client.Init()
            except Exception as e:
                logger_mp.error(f"[voix] haut-parleur du robot indisponible ({e}) : annonces sur le PC")
                self.mode = "pc"
        self.pc_cmd = None
        self.piper = None
        if self.mode == "pc":
            self._init_piper()
        if self.mode == "pc" and self.piper is None:
            if shutil.which("spd-say"):
                self.pc_cmd = ["spd-say", "-l", "fr", "-r", "10"]
            elif shutil.which("espeak-ng"):
                self.pc_cmd = ["espeak-ng", "-v", "fr"]
            else:
                logger_mp.warning("[voix] ni spd-say ni espeak-ng : annonces désactivées")
                self.mode = "off"

    def _init_piper(self):
        """Voix neuronale Piper (naturelle) : modèle chargé une fois, phrases synthétisées dans un fil dédié
        et gardées en cache (les annonces se répètent), lecture par aplay / pw-play."""
        name = os.environ.get("TELEOP_VOICE", "fr_FR-siwis-medium")
        model = Path(name) if name.endswith(".onnx") else Path.home() / ".local/share/piper-voices" / f"{name}.onnx"
        player = shutil.which("aplay") or shutil.which("pw-play") or shutil.which("paplay")
        if not model.exists() or not player:
            return
        try:
            from piper import PiperVoice
            self.piper = PiperVoice.load(str(model))
        except Exception as e:
            logger_mp.warning(f"[voix] Piper indisponible ({e}) : repli sur spd-say")
            self.piper = None
            return
        self.player = [player, "-q"] if player.endswith("aplay") else [player]
        self.cache = {}
        self.q = queue.Queue()
        threading.Thread(target=self._piper_loop, daemon=True).start()
        logger_mp.info(f"[voix] Piper : {model.stem}")

    def _piper_loop(self):
        while True:
            text = self.q.get()
            try:
                wav = self.cache.get(text)
                if wav is None:
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as w:
                        self.piper.synthesize_wav(text, w)
                    wav = self.cache[text] = buf.getvalue()
                subprocess.run(self.player + ["-"], input=wav, timeout=15,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                logger_mp.debug(f"[voix] annonce en échec : {e}")

    def say(self, fr: str, en: str = None):
        if self.mode == "off":
            return
        if self.piper is not None:
            self.q.put(fr)                 # dans l'ordre, sans chevauchement
            return
        threading.Thread(target=self._say, args=(fr, en or fr), daemon=True).start()

    def _say(self, fr: str, en: str):
        try:
            if self.mode == "robot":
                self.client.TtsMaker(en, self.speaker_id)
            else:
                subprocess.run(self.pc_cmd + [fr], timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logger_mp.debug(f"[voix] annonce en échec : {e}")
