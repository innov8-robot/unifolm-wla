"""Surveillance des caméras pendant la téléop (casque en pass-through : on ne voit plus les images).

Une caméra est « absente » si elle n'a pas fourni de NOUVELLE image depuis ``timeout`` s (image
nulle, flux coupé ou figé). Annonces vocales : au démarrage (« trois caméras OK » ou laquelle
manque), à chaque disparition (répétée toutes les ``repeat`` s tant qu'elle manque) et au retour.
"""
import time

import logging_mp

logger_mp = logging_mp.getLogger(__name__)


class CameraWatch:
    def __init__(self, names, voice, timeout: float = 1.0, repeat: float = 10.0):
        now = time.time()
        self.voice, self.timeout, self.repeat = voice, timeout, repeat
        self.state = {n: {"sig": None, "t": now, "missing": False, "said": 0.0} for n in names}
        self.startup_done = False
        self.t0 = now

    @staticmethod
    def _sig(frame):
        if frame is None or getattr(frame, "bgr", None) is None:
            return None
        b = frame.bgr
        return b[::max(1, b.shape[0] // 6), ::max(1, b.shape[1] // 8)].tobytes()

    def update(self, frames: dict, recording: bool = False) -> list:
        """``frames`` = {nom: trame ou None}. Rend la liste des caméras absentes."""
        now = time.time()
        missing = []
        for n, f in frames.items():
            st = self.state[n]
            sig = self._sig(f)
            if sig is not None and sig != st["sig"]:
                st["sig"], st["t"] = sig, now
            gone = now - st["t"] > self.timeout
            if gone:
                missing.append(n)
                if not st["missing"] or now - st["said"] > self.repeat:
                    msg = f"Attention, caméra {n} absente" + (", épisode à refaire" if recording else "")
                    logger_mp.warning(f"📷  {msg} (aucune nouvelle image depuis {now - st['t']:.1f} s)")
                    if self.startup_done or now - self.t0 > 3.0:
                        self.voice.say(msg, f"Warning, {n} camera missing")
                    st["said"] = now
                st["missing"] = True
            elif st["missing"]:
                st["missing"] = False
                logger_mp.info(f"📷  caméra {n} revenue")
                self.voice.say(f"Caméra {n} revenue", f"{n} camera back")
        if not self.startup_done and (not missing or now - self.t0 > 3.0):
            self.startup_done = True
            if not missing:
                self.voice.say("Trois caméras OK" if len(frames) == 3 else f"{len(frames)} caméras OK", "Cameras OK")
                logger_mp.info(f"📷  caméras OK : {', '.join(frames)}")
        return missing
