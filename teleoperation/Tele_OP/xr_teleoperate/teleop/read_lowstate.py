"""Lecture SEULE de l'état des 35 moteurs (rt/lowstate) : n'envoie AUCUNE commande au robot.

Sert à identifier sur le G1-D les indices du tangage et de la rotation du buste avant d'utiliser
--torso-pitch-index / --torso-yaw-index. Affiche les angles qui ne sont pas des bras (15–28),
et signale ceux qui bougent pendant la lecture.

    python read_lowstate.py --network-interface enx0c3796e0bc5b [--seconds 20]
"""
import argparse
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

ARMS = set(range(15, 29))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network-interface", default=None)
    ap.add_argument("--seconds", type=float, default=20.0)
    a = ap.parse_args()
    ChannelFactoryInitialize(0, networkInterface=a.network_interface)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()
    q0 = None
    lo = hi = None
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        msg = sub.Read()
        if msg is None:
            time.sleep(0.01)
            continue
        q = np.array([msg.motor_state[i].q for i in range(35)])
        if q0 is None:
            q0, lo, hi = q.copy(), q.copy(), q.copy()
            print("indice : angle (rad) au départ — hors bras (15–28)")
            for i in range(35):
                if i not in ARMS:
                    print(f"  {i:2d} : {q[i]:+.3f}")
        lo, hi = np.minimum(lo, q), np.maximum(hi, q)
        moving = [i for i in range(35) if i not in ARMS and hi[i] - lo[i] > 0.02]
        print(f"\r  hors bras en mouvement (>0,02 rad) : {moving}   ", end="", flush=True)
        time.sleep(0.05)
    print()
    if q0 is None:
        print("aucun message rt/lowstate reçu : vérifier --network-interface et le câble")


if __name__ == "__main__":
    main()
