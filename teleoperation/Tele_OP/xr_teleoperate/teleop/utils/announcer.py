"""Annonces vocales de la téléop (enregistrement lancé, sauvegardé, annulé, pause...).

* ``pc``    : synthèse du PC (``spd-say`` en français, repli ``espeak-ng``), non bloquante ;
* ``robot`` : haut-parleur du robot, service « voice » d'Unitree (TtsMaker), en anglais ;
* ``off``   : rien.
Une annonce qui échoue ne fait jamais planter la téléop.
"""
import shutil
import subprocess
import threading

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
        if self.mode == "pc":
            if shutil.which("spd-say"):
                self.pc_cmd = ["spd-say", "-l", "fr", "-r", "10"]
            elif shutil.which("espeak-ng"):
                self.pc_cmd = ["espeak-ng", "-v", "fr"]
            else:
                logger_mp.warning("[voix] ni spd-say ni espeak-ng : annonces désactivées")
                self.mode = "off"

    def say(self, fr: str, en: str = None):
        if self.mode == "off":
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
