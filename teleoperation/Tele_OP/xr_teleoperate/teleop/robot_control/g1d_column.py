"""Colonne télescopique du G1-D (montée / descente du buste), d'après mpc_any (runtime/column.py).

    état      rt/hispeed_state  (geometry_msgs Point32)  hauteur capteur = .y (m)
    commande  rt/cmd_hispeed    (geometry_msgs Point32)  VITESSE = .z ∈ [-1, 1], à publier EN CONTINU

Le zéro du capteur DÉRIVE : la seule référence fiable est la BUTÉE BASSE. ``home()`` descend
jusqu'à l'arrêt du mouvement et mémorise la lecture ; toutes les hauteurs sont ensuite
« mètres au-dessus de la butée basse ».

Un fil streame la dernière consigne à 50 Hz ; consigne non rafraîchie depuis ``STALE_S`` -> 0.
Bornes logicielles : jamais sous la butée (descente coupée à 5 mm), jamais au-dessus de
``max_height`` ; ralentissement dans les 3 derniers cm.
"""
import threading
import time

import logging_mp
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_

logger_mp = logging_mp.getLogger(__name__)

TOPIC_STATE = "rt/hispeed_state"
TOPIC_CMD = "rt/cmd_hispeed"
HARD_VEL_MAX = 0.30        # bride de mpc_any (l'API accepte 1.0)
SEND_HZ = 50.0
STALE_S = 0.3
HOME_VEL = -0.15           # descente franche vers la butée (mpc_any)
STUCK_WINDOW_S = 1.0
STUCK_MOVE_M = 0.0006      # < 0,6 mm sur la fenêtre : pas de progrès
STUCK_WINDOWS = 2
SLOW_ZONE_M = 0.03


class ColumnController:
    def __init__(self, vel_max: float = 0.3, max_height: float = 0.40):
        self.vel_max = min(abs(float(vel_max)), HARD_VEL_MAX)
        self.max_height = float(max_height)
        self.pub = ChannelPublisher(TOPIC_CMD, Point32_)
        self.pub.Init()
        self.sub = ChannelSubscriber(TOPIC_STATE, Point32_)
        self.sub.Init()
        self.ref = None                       # lecture capteur à la butée basse
        self._raw = None
        self._lock = threading.Lock()
        self._cmd, self._t = 0.0, 0.0
        self.running = True
        self._homing = False
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._send_loop, daemon=True).start()

    # ------------------------------------------------------------------ E/S
    def _read_loop(self):
        while self.running:
            m = self.sub.Read()
            if m is not None:
                with self._lock:
                    self._raw = float(m.y)
            time.sleep(0.005)

    def raw_height(self):
        with self._lock:
            return self._raw

    def height(self):
        """Mètres au-dessus de la butée basse, ou None (pas d'état, ou pas de home)."""
        raw = self.raw_height()
        return None if raw is None or self.ref is None else raw - self.ref

    def _send_loop(self):
        while self.running:
            with self._lock:
                v = self._cmd if time.time() - self._t < STALE_S else 0.0
            try:
                self.pub.Write(Point32_(0.0, 0.0, float(v)))
            except Exception:
                pass
            time.sleep(1.0 / SEND_HZ)

    def _write(self, v: float):
        with self._lock:
            self._cmd, self._t = float(v), time.time()

    # ------------------------------------------------------------------ commande
    def set_velocity(self, v: float) -> float:
        """Consigne de vitesse (+ = monte), bornée et coupée aux limites ; rend la consigne retenue.
        Sans home, la colonne ne bouge pas."""
        h = self.height()
        if h is None or self._homing:
            self._write(0.0)
            return 0.0
        v = max(-self.vel_max, min(self.vel_max, float(v)))
        if v < 0:
            v = 0.0 if h <= 0.005 else v * min(1.0, h / SLOW_ZONE_M)
        elif v > 0:
            room = self.max_height - h
            v = 0.0 if room <= 0.0 else v * min(1.0, room / SLOW_ZONE_M)
        self._write(v)
        return v

    def home(self, timeout: float = 30.0) -> bool:
        """Descend jusqu'à la butée basse (plus de progrès) et la mémorise. Bloquant."""
        t0 = time.time()
        while self.raw_height() is None and time.time() - t0 < 3.0:
            self._write(0.0)
            time.sleep(0.05)
        cur = self.raw_height()
        if cur is None:
            logger_mp.error("[Colonne] aucun état sur rt/hispeed_state : HOME refusé, colonne non pilotée")
            return False
        self._homing = True
        logger_mp.info(f"[Colonne] descente en butée basse (home), capteur {cur:.4f} m ...")
        t0 = t_win = time.time()
        h_win, stuck = cur, 0
        while time.time() - t0 < timeout:
            self._write(HOME_VEL)
            time.sleep(0.02)
            cur = self.raw_height()
            now = time.time()
            if now - t_win >= STUCK_WINDOW_S:
                stuck = stuck + 1 if abs(cur - h_win) < STUCK_MOVE_M else 0
                if stuck >= STUCK_WINDOWS:
                    break
                t_win, h_win = now, cur
        self._write(0.0)
        time.sleep(0.2)
        self.ref = self.raw_height()
        self._homing = False
        logger_mp.info(f"[Colonne] butée basse = référence 0 (capteur {self.ref:.4f} m)")
        return True

    def stop(self):
        """Zéros streamés puis arrêt des fils."""
        self._write(0.0)
        for _ in range(10):
            try:
                self.pub.Write(Point32_(0.0, 0.0, 0.0))
            except Exception:
                pass
            time.sleep(0.02)
        self.running = False
