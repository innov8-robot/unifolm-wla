"""Surveillance des caméras pendant la téléop (casque en pass-through : on ne voit plus les images).

Une caméra est « absente » si elle n'a pas fourni de NOUVELLE image depuis ``timeout`` s (image
nulle, flux coupé ou figé). Annonces vocales : au démarrage (« trois caméras OK » ou laquelle
manque), à chaque disparition (répétée toutes les ``repeat`` s tant qu'elle manque) et au retour.

NETTETÉ de la caméra de tête (6/10 : l'œil gauche, celui que voit le modèle, a été flou pendant 145 épisodes
sans que personne ne le voie) : toutes les ``sharp_every`` s, variance du laplacien au centre de chaque
œil ; si l'œil GAUCHE est nettement moins net que l'œil droit (référence, même scène), annonce vocale
« caméra de tête floue », répétée toutes les ``sharp_repeat`` s. Régler avec teleoperation/focus_camera.py.
"""
import time

import cv2
import logging_mp
import numpy as np

logger_mp = logging_mp.getLogger(__name__)


class CameraWatch:
    #: œil gauche flou si sa netteté < SHARP_RATIO × celle de l'œil droit (mesuré : net ≈ 0.8-0.9, flou ≈ 0.15)
    SHARP_RATIO = 0.5
    #: en dessous, l'œil droit voit une scène trop uniforme pour servir de référence (mur, cache)
    SHARP_MIN_REF = 60.0

    def __init__(self, names, voice, timeout: float = 1.0, repeat: float = 10.0,
                 sharp_every: float = 5.0, sharp_repeat: float = 60.0):
        now = time.time()
        self.voice, self.timeout, self.repeat = voice, timeout, repeat
        self.sharp_every, self.sharp_repeat = sharp_every, sharp_repeat
        self.sharp = None                 # (gauche, droit) lissés
        self.sharp_t = 0.0
        self.sharp_said = 0.0
        self.sharp_logged = False
        self.state = {n: {"sig": None, "t": now, "missing": False, "said": 0.0} for n in names}
        self.startup_done = False
        self.t0 = now

    @staticmethod
    def _sig(frame):
        if frame is None or getattr(frame, "bgr", None) is None:
            return None
        b = frame.bgr
        return b[::max(1, b.shape[0] // 6), ::max(1, b.shape[1] // 8)].tobytes()

    @staticmethod
    def _sharpness(img: np.ndarray) -> float:
        h, w = img.shape[:2]
        c = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        return float(cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

    def _check_focus(self, frame, now: float, recording: bool):
        b = getattr(frame, "bgr", None) if frame is not None else None
        if b is None or b.shape[1] < 2 * b.shape[0]:          # tête stéréo seulement (deux yeux côte à côte)
            return
        w = b.shape[1] // 2
        sl, sr = self._sharpness(b[:, :w]), self._sharpness(b[:, w:])
        self.sharp = (sl, sr) if self.sharp is None else (0.6 * self.sharp[0] + 0.4 * sl, 0.6 * self.sharp[1] + 0.4 * sr)
        sl, sr = self.sharp
        if not self.sharp_logged:
            logger_mp.info(f"📷  netteté caméra de tête : œil gauche {sl:.0f}, œil droit {sr:.0f}")
            self.sharp_logged = True
        if sr >= self.SHARP_MIN_REF and sl < self.SHARP_RATIO * sr:
            if now - self.sharp_said > self.sharp_repeat:
                logger_mp.warning(f"📷  CAMÉRA DE TÊTE FLOUE : œil gauche (vue du modèle) {sl:.0f} contre {sr:.0f} pour l'œil "
                                  f"droit — régler la mise au point (teleoperation/focus_camera.py)"
                                  + (" ; épisode à refaire" if recording else ""))
                self.voice.say("Attention, caméra de tête floue", "Warning, head camera blurry")
                self.sharp_said = now

    def update(self, frames: dict, recording: bool = False) -> list:
        """``frames`` = {nom: trame ou None}. Rend la liste des caméras absentes."""
        now = time.time()
        if now - self.sharp_t > self.sharp_every and "tête" in frames:
            self.sharp_t = now
            try:
                self._check_focus(frames["tête"], now, recording)
            except Exception as e:                            # une mesure ratée ne doit jamais gêner la téléop
                logger_mp.debug(f"netteté : {e}")
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
