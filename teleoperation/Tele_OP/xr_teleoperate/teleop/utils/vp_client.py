"""Client du service de visual prompt (inference_studio/vp_tracker.py) : donne l'image de tête COLORIÉE
(source verte, cible rouge) à envoyer au modèle, ou None si le suivi n'est pas prêt."""
import json
import time

import cv2
import numpy as np
import zmq


class VPClient:
    def __init__(self, url: str, timeout_ms: int = 300):
        self.url, self.timeout_ms = url, timeout_ms
        self.ctx = zmq.Context.instance()
        self.sock = None
        self.last_status = {}
        self._cache = (None, 0.0)                     # 10 requêtes / s au plus (la boucle tourne à 60 Hz)

    def _socket(self):
        if self.sock is None:
            self.sock = self.ctx.socket(zmq.REQ)
            self.sock.setsockopt(zmq.LINGER, 0)
            self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self.sock.connect(self.url)
        return self.sock

    def prompted_head(self, max_age: float = 0.1):
        """Image BGR de l'œil gauche avec les couleurs, ou None (suivi pas prêt / service muet)."""
        img, t = self._cache
        if time.time() - t < max_age:
            return img
        img = self._fetch()
        self._cache = (img, time.time())
        return img

    def _fetch(self):
        try:
            s = self._socket()
            s.send_json({"cmd": "prompted"})
            meta, jpg = s.recv_multipart()
        except zmq.ZMQError:
            if self.sock is not None:                 # REQ bloqué après un délai : on le recrée
                self.sock.close()
                self.sock = None
            self.last_status = {"ok": False, "error": "service de visual prompt muet"}
            return None
        self.last_status = json.loads(meta)
        if not self.last_status.get("ok") or not jpg:
            return None
        return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
