"""Base roulante du G1-D en vitesse, par le service RPC « agv » du robot (repris de mpc_any,
SLAM/app_lidar.py) : ``Move(vx, vy, vyaw)`` = api 1001, base différentielle (vy ignoré).

⚠ Le service relaie sur rt/cmd_vel_no_limit : les garde-fous de vitesse du châssis sont
contournés, les bornes sont ICI. La base continue ~1,5 s après la dernière commande : un arrêt
envoie toujours des vitesses nulles (plusieurs fois), jamais un simple silence.

Un fil d'envoi à 20 Hz transmet la dernière consigne ; si la boucle de téléop ne l'a pas
rafraîchie depuis ``STALE_S`` (boucle bloquée, plantage), il envoie 0.
"""
import json
import threading
import time

import logging_mp
from unitree_sdk2py.rpc.client import Client

logger_mp = logging_mp.getLogger(__name__)

AGV_SERVICE_NAME = "agv"
AGV_API_VERSION = "1.0.0.1"
AGV_API_MOVE = 1001
HARD_MAX_VX = 1.0        # m/s, maximum du châssis (lu en REST dans mpc_any)
HARD_MAX_VYAW = 0.6      # rad/s
SEND_HZ = 20.0
STALE_S = 0.3
STOP_REPEATS = 4


class _AgvClient(Client):
    def __init__(self):
        super().__init__(AGV_SERVICE_NAME, False)

    def Init(self):
        self._SetApiVerson(AGV_API_VERSION)
        self._RegistApi(AGV_API_MOVE, 0)

    def move(self, vx: float, vyaw: float) -> int:
        code, _ = self._Call(AGV_API_MOVE, json.dumps({"vx": vx, "vy": 0.0, "vyaw": vyaw}))
        return code


class G1DBase:
    def __init__(self, max_vx: float = 0.3, max_vyaw: float = 0.4):
        self.max_vx = min(abs(float(max_vx)), HARD_MAX_VX)
        self.max_vyaw = min(abs(float(max_vyaw)), HARD_MAX_VYAW)
        self.client = _AgvClient()
        self.client.SetTimeout(0.2)
        self.client.Init()
        self._lock = threading.Lock()
        self._cmd = (0.0, 0.0)
        self._t = 0.0
        self._sent = (0.0, 0.0)
        self._err_logged = 0.0
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger_mp.info(f"[G1DBase] base roulante pilotée (service agv) : ±{self.max_vx} m/s, ±{self.max_vyaw} rad/s")

    def set(self, vx: float, vyaw: float) -> tuple:
        """Consigne (bornée) ; à rafraîchir à chaque tour de boucle. Rend la consigne retenue."""
        vx = max(-self.max_vx, min(self.max_vx, float(vx)))
        vyaw = max(-self.max_vyaw, min(self.max_vyaw, float(vyaw)))
        with self._lock:
            self._cmd, self._t = (vx, vyaw), time.time()
        return vx, vyaw

    def _loop(self):
        while self.running:
            with self._lock:
                cmd = self._cmd if time.time() - self._t < STALE_S else (0.0, 0.0)
            try:
                # à l'arrêt, ne pas inonder le service : un zéro tous les 5 envois suffit
                if cmd != (0.0, 0.0) or self._sent != (0.0, 0.0) or int(time.time() * SEND_HZ) % 5 == 0:
                    self.client.move(*cmd)
                    self._sent = cmd
            except Exception as e:
                if time.time() - self._err_logged > 2.0:
                    self._err_logged = time.time()
                    logger_mp.error(f"[G1DBase] appel agv en échec : {e}")
            time.sleep(1.0 / SEND_HZ)

    def stop(self):
        """Arrêt : vitesses nulles répétées, puis fin du fil d'envoi."""
        self.set(0.0, 0.0)
        self.running = False
        for _ in range(STOP_REPEATS):
            try:
                self.client.move(0.0, 0.0)
            except Exception:
                pass
            time.sleep(0.02)
        logger_mp.info("[G1DBase] base arrêtée")
