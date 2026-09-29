#!/usr/bin/env python3
"""
Lecture / commande de la colonne telescopique (lift) du G1 via DDS, depuis le PC.

Topics (confirmes sur le robot) :
  - etat    : rt/hispeed_state  (Point32)  -> hauteur = champ .y  en metres
  - commande: rt/cmd_hispeed    (Point32)  -> VITESSE = champ .z  dans [-1, 1]
              (positif = monte, negatif = descend, 0 = maintient ; a publier en continu)

Exemples :
  # lecture unique
  python height_tool.py read
  # lecture continue (Ctrl-C pour stopper)
  python height_tool.py watch
  # commande de VITESSE brute (comme le joystick) : monte a 0.12 pendant 1 s
  python height_tool.py vel 0.12 --duration 1.0
  # aller a une hauteur cible (boucle fermee) : 0.10 m
  python height_tool.py goto 0.10

Interface reseau par defaut : enx0c3796e0bc5b (cable vers le robot). Override avec --iface.
"""
import argparse
import os
import sys
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_

TOPIC_STATE = "rt/hispeed_state"
TOPIC_CMD = "rt/cmd_hispeed"

# --- garde-fous ---
VEL_MAX = 0.30      # vitesse max autorisee par cet outil (l'API accepte jusqu'a 1.0)
HEIGHT_MIN = -0.30  # borne basse capteur (le zero derive, peut etre negatif)
HEIGHT_MAX = 0.70   # butee haute SUPPOSEE (a ajuster ! override avec --max)
PUB_HZ = 50.0

# reference de la butee basse (ecrite par 'home'), pour travailler en relatif
REF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".height_ref")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def save_ref(v):
    try:
        with open(REF_FILE, "w") as f:
            f.write("%.6f" % v)
    except OSError as e:
        print("!! impossible d'ecrire la reference: %s" % e)


def load_ref():
    try:
        with open(REF_FILE) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return None


def make_io(iface):
    ChannelFactoryInitialize(0, iface)
    pub = ChannelPublisher(TOPIC_CMD, Point32_)
    pub.Init()
    sub = ChannelSubscriber(TOPIC_STATE, Point32_)
    sub.Init()
    return pub, sub


def read_height(sub, tries=40, wait=0.05):
    """Renvoie la hauteur (m) ou None si aucun message."""
    for _ in range(tries):
        m = sub.Read()
        if m is not None:
            return m.y
        time.sleep(wait)
    return None


def send_vel(pub, v):
    """Publie une commande de vitesse (champ .z)."""
    pub.Write(Point32_(0.0, 0.0, float(v)))


def hold_stop(pub, cycles=25, dt=0.02):
    """Stream de vitesse nulle pour arreter/maintenir proprement."""
    for _ in range(cycles):
        send_vel(pub, 0.0)
        time.sleep(dt)


def warmup(pub, sub, timeout=3.0):
    """Attend l'appariement DDS (inter-machines) en streamant une vitesse nulle
    jusqu'a recevoir le premier etat. Sinon une commande courte ne bouge rien."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        send_vel(pub, 0.0)
        if sub.Read() is not None:
            return True
        time.sleep(0.05)
    return False


def cmd_read(pub, sub, args):
    h = read_height(sub)
    if h is None:
        print("NO_DATA (aucun message sur %s)" % TOPIC_STATE)
        return 1
    ref = load_ref()
    if ref is not None:
        print("hauteur = %.4f m  (soit %+.4f m au-dessus du fond home)" % (h, h - ref))
    else:
        print("hauteur = %.4f m  (pas de reference : lance 'home' pour du relatif)" % h)
    return 0


def cmd_watch(pub, sub, args):
    print("Lecture continue (Ctrl-C pour arreter)...")
    try:
        while True:
            m = sub.Read()
            if m is not None:
                print("\rhauteur = %.4f m   " % m.y, end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    return 0


def cmd_vel(pub, sub, args):
    v = clamp(args.value, -VEL_MAX, VEL_MAX)
    if v != args.value:
        print("!! vitesse bridee a %.3f (|max|=%.2f)" % (v, VEL_MAX))
    h0 = read_height(sub)
    print("AVANT  hauteur = %s" % ("%.4f m" % h0 if h0 is not None else "??"))
    print(">> stream vitesse %.3f pendant %.2f s" % (v, args.duration))
    dt = 1.0 / PUB_HZ
    t0 = time.time()
    while time.time() - t0 < args.duration:
        send_vel(pub, v)
        time.sleep(dt)
    hold_stop(pub)
    time.sleep(0.2)
    h1 = read_height(sub)
    print("APRES  hauteur = %s" % ("%.4f m" % h1 if h1 is not None else "??"))
    if h0 is not None and h1 is not None:
        print("DELTA  %+.4f m" % (h1 - h0))
    return 0


def cmd_goto(pub, sub, args):
    if args.rel:
        ref = load_ref()
        if ref is None:
            print("!! pas de reference : lance d'abord 'python height_tool.py home'")
            return 1
        physical = clamp(args.target, 0.0, args.max)
        if physical != args.target:
            print("!! hauteur relative bridee a %.4f m (plage 0..%.2f)" % (physical, args.max))
        target = ref + physical
        print("(relatif) %.3f m au-dessus du fond (ref %.4f) -> cible capteur %.4f m"
              % (physical, ref, target))
    else:
        target = clamp(args.target, HEIGHT_MIN, args.max)
        if target != args.target:
            print("!! cible bridee a %.4f m (plage %.2f..%.2f)" % (target, HEIGHT_MIN, args.max))
    h0 = read_height(sub)
    if h0 is None:
        print("NO_DATA : impossible de lire l'etat, j'annule.")
        return 1
    print("AVANT  hauteur = %.4f m  ->  cible %.4f m" % (h0, target))

    tol = 0.002            # 2 mm de tolerance
    kp = 6.0               # gain proportionnel vitesse/erreur
    vmin = 0.05            # plancher de vitesse : creep fin pres de la cible
    dt = 1.0 / PUB_HZ
    win = 1.0              # fenetre de mesure du progres (s) - longue = pas de faux blocage
    win_move = 0.0006      # < 0.6 mm net sur 1 s => pas de progres
    t0 = time.time()
    hcur = h0
    t_win = time.time()
    h_win = h0
    stuck_windows = 0
    while time.time() - t0 < args.timeout:
        m = sub.Read()
        if m is not None:
            hcur = m.y
        err = target - hcur
        if abs(err) <= tol:
            break
        v = clamp(kp * err, -VEL_MAX, VEL_MAX)
        if 0 < abs(v) < vmin:          # applique le plancher de vitesse
            v = vmin if v > 0 else -vmin
        send_vel(pub, v)
        # watchdog anti-butee sur fenetre de temps (et non par cycle) :
        # butee reelle = on commande une vitesse mais rien ne bouge sur ~1.2 s
        now = time.time()
        if now - t_win >= win:
            if abs(hcur - h_win) < win_move:
                stuck_windows += 1
                if stuck_windows >= 2:
                    print("!! plus de progres (butee reelle ?), arret a %.4f m" % hcur)
                    break
            else:
                stuck_windows = 0
            t_win = now
            h_win = hcur
        time.sleep(dt)

    hold_stop(pub)
    time.sleep(0.2)
    h1 = read_height(sub)
    print("APRES  hauteur = %s" % ("%.4f m" % h1 if h1 is not None else "??"))
    if h1 is not None:
        print("erreur finale : %+.4f m" % (target - h1))
    return 0


def cmd_home(pub, sub, args):
    """Descend jusqu'a la butee basse mecanique = seule reference physique fiable.
    (le zero capteur n'est pas absolu et derive apres un debranchement/coupure)"""
    h0 = read_height(sub)
    print("Descente vers la butee basse... (capteur depart = %s)"
          % ("%.4f m" % h0 if h0 is not None else "??"))
    dt = 1.0 / PUB_HZ
    win = 1.0
    win_move = 0.0006
    t0 = time.time()
    t_win = time.time()
    hcur = h0 if h0 is not None else 0.0
    h_win = hcur
    stuck = 0
    while time.time() - t0 < args.timeout:
        m = sub.Read()
        if m is not None:
            hcur = m.y
        send_vel(pub, -0.15)          # descente franche
        now = time.time()
        if now - t_win >= win:
            if abs(hcur - h_win) < win_move:
                stuck += 1
                if stuck >= 2:
                    break             # plus de mouvement => butee atteinte
            else:
                stuck = 0
            t_win = now
            h_win = hcur
        time.sleep(dt)
    hold_stop(pub)
    time.sleep(0.3)
    hb = read_height(sub)
    if hb is not None:
        save_ref(hb)
        print("BUTEE BASSE atteinte. Capteur au fond = %.4f m  (memorise comme reference 0)" % hb)
        print(">> Utilise maintenant :  goto --rel <metres au-dessus du fond>")
    else:
        print("BUTEE BASSE atteinte mais lecture capteur impossible (reference non memorisee)")
    return 0


def main():
    p = argparse.ArgumentParser(description="Lire/commander la colonne telescopique du G1 via DDS")
    p.add_argument("--iface", default="enx0c3796e0bc5b", help="interface reseau DDS vers le robot")
    p.add_argument("--max", type=float, default=HEIGHT_MAX,
                   help="hauteur max autorisee pour goto (m) - AJUSTE selon ta colonne")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("read", help="lire la hauteur une fois")
    sub.add_parser("watch", help="lire la hauteur en continu")

    ph = sub.add_parser("home", help="descendre a la butee basse (reference physique fiable)")
    ph.add_argument("--timeout", type=float, default=30.0, help="timeout de securite (s)")

    pv = sub.add_parser("vel", help="commander une VITESSE brute (comme le joystick)")
    pv.add_argument("value", type=float, help="vitesse dans [-%.2f, %.2f] (+ = monte)" % (VEL_MAX, VEL_MAX))
    pv.add_argument("--duration", type=float, default=0.8, help="duree du stream (s)")

    pg = sub.add_parser("goto", help="aller a une hauteur cible en metres (boucle fermee)")
    pg.add_argument("target", type=float, help="hauteur cible en metres (absolue, ou relative au fond avec --rel)")
    pg.add_argument("--rel", action="store_true",
                    help="cible = metres AU-DESSUS de la butee basse memorisee par 'home' (repetable)")
    pg.add_argument("--timeout", type=float, default=15.0, help="timeout de securite (s)")

    args = p.parse_args()
    pub, io_sub = make_io(args.iface)
    # attend l'appariement DDS avant toute commande (crucial en inter-machines)
    if not warmup(pub, io_sub):
        print("!! pas d'etat DDS recu (verifie --iface / le lien vers le robot)")

    dispatch = {
        "read": cmd_read,
        "watch": cmd_watch,
        "home": cmd_home,
        "vel": cmd_vel,
        "goto": cmd_goto,
    }
    return dispatch[args.cmd](pub, io_sub, args)


if __name__ == "__main__":
    sys.exit(main())
