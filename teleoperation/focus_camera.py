"""Mise au point des caméras du G1-D : flux en direct + mesure de NETTETÉ, pour régler la bague à la main.

    conda activate g1d_teleop
    python ~/Documents/project/manip/unifolm-wla/teleoperation/focus_camera.py            # tête (stéréo)
    python ~/Documents/project/manip/unifolm-wla/teleoperation/focus_camera.py --cam poignet-g
    python ~/Documents/project/manip/unifolm-wla/teleoperation/focus_camera.py --cam poignet-d

Lit les images publiées par le service caméra du robot (teleimager, ZMQ) : ne gêne ni la téléop ni
l'enregistrement, on peut le lancer en même temps.

Affichage : l'image (les deux yeux pour la tête), et pour chaque vue un ZOOM ×2 du centre, la netteté
courante (variance du laplacien, lissée) et la MEILLEURE valeur vue depuis le lancement. Tournez la
bague doucement : la netteté monte, passe par un maximum, puis redescend — revenez au maximum.
Visez au moins la valeur de l'autre œil (référence) ; la zone centrale doit contenir des détails
(pièces, marqueur ArUco), pas un mur uni.

Touches : q / Échap = quitter · r = remettre les maxima à zéro.
"""
import argparse
import time

import cv2
import numpy as np
import zmq

PORTS = {"tete": 55558, "poignet-g": 55557, "poignet-d": 55556}


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def center(img: np.ndarray, frac: float = 0.5) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    return img[(h - ch) // 2:(h + ch) // 2, (w - cw) // 2:(w + cw) // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", choices=list(PORTS), default="tete")
    ap.add_argument("--ip", default="192.168.123.164")
    a = ap.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.CONFLATE, 1)                 # toujours la DERNIÈRE image, pas de retard qui s'accumule
    sock.setsockopt(zmq.RCVTIMEO, 3000)
    sock.connect(f"tcp://{a.ip}:{PORTS[a.cam]}")
    print(f"caméra « {a.cam} » : tcp://{a.ip}:{PORTS[a.cam]} — q pour quitter, r pour remettre les maxima à zéro")

    ema, best = {}, {}
    win = f"Mise au point — {a.cam}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last = time.time()
    while True:
        try:
            buf = sock.recv()
        except zmq.Again:
            print("aucune image depuis 3 s : service caméra du robot arrêté, ou câble réseau ?")
            continue
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        views = {"oeil gauche (modele)": img[:, : w // 2], "oeil droit": img[:, w // 2:]} \
            if a.cam == "tete" and w >= 2 * h else {a.cam: img}
        tiles = []
        for name, v in views.items():
            s = sharpness(cv2.cvtColor(center(v), cv2.COLOR_BGR2GRAY))
            ema[name] = s if name not in ema else 0.8 * ema[name] + 0.2 * s
            best[name] = max(best.get(name, 0.0), ema[name])
            view = v.copy()
            cv2.rectangle(view, (v.shape[1] // 4, v.shape[0] // 4), (3 * v.shape[1] // 4, 3 * v.shape[0] // 4),
                          (53, 224, 200), 1)
            zoom = cv2.resize(center(v), (v.shape[1], v.shape[0]), interpolation=cv2.INTER_NEAREST)
            col = np.vstack([view, zoom])
            ratio = ema[name] / best[name] if best[name] > 0 else 0
            color = (60, 200, 60) if ratio > 0.95 else (40, 170, 240) if ratio > 0.8 else (60, 60, 230)
            cv2.rectangle(col, (0, 0), (col.shape[1], 64), (20, 20, 20), -1)
            cv2.putText(col, f"{name}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(col, f"nettete {ema[name]:6.0f}   max {best[name]:6.0f}", (10, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            bw = int((col.shape[1] - 20) * min(1.0, ratio))
            cv2.rectangle(col, (10, 60), (10 + bw, 63), color, -1)
            tiles.append(col)
        cv2.imshow(win, np.hstack(tiles))
        if time.time() - last > 1.0:
            print("   ".join(f"{n}: {ema[n]:6.0f} (max {best[n]:6.0f})" for n in views), flush=True)
            last = time.time()
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("r"):
            best.clear()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
