"""Tâche de validation en sim : saisir un cube rouge de 4 cm avec le bras droit et le soulever.

Expert SCRIPTÉ (pas de VR) : garde l'orientation de pince de départ du G1 (doigts vers l'intérieur,
inclinés vers le bas, comme dans les démos d'entraînement), approche le cube le long des doigts,
ferme la pince, soulève. Toutes les cibles sont exprimées dans la base WLA.

Chaque pas suit l'ordre de xr_teleoperate : mesure -> commande (IK) -> ``on_step`` -> exécution.
``on_step`` reçoit donc l'état MESURÉ et la commande du pas, ce qu'enregistre la téléop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from g1d_sim import G1DSim

INSTRUCTION = "pick up the red cube"
CUBE_HALF = 0.02
TABLE_TOP_WORLD = 0.87
#: zone de tirage du cube dans la base WLA (bras droit), lacet en rad
CUBE_X = (0.27, 0.37)
CUBE_Y = (-0.24, -0.10)
CUBE_YAW = (-0.35, 0.35)
LIFT_SUCCESS = 0.05


@dataclass
class ExpertParams:
    #: recul du point effecteur WLA derrière le centre du cube, le long des doigts (m)
    grasp_back: float = 0.035
    #: recul et hauteur supplémentaires du point de pré-saisie
    pre_back: float = 0.07
    pre_up: float = 0.05
    lift: float = 0.10
    #: durées des phases, en pas de 30 Hz
    t_hold0: int = 8
    t_pre: int = 45
    t_approach: int = 30
    t_close: int = 20
    t_lift: int = 40
    t_hold1: int = 15


def _minjerk(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n + 1)[1:]
    s = 10 * s**3 - 15 * s**4 + 6 * s**5
    return a[None] + (b - a)[None] * s[:, None]


def sample_cube(sim: G1DSim, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Tire la pose du cube (base WLA -> monde) et le pose sur la table. Rend (xyz monde, lacet)."""
    B = sim.base_pose_wla()
    xb, yb = rng.uniform(*CUBE_X), rng.uniform(*CUBE_Y)
    p = B @ np.array([xb, yb, 0.0, 1.0])
    xyz = np.array([p[0], p[1], TABLE_TOP_WORLD + CUBE_HALF + 0.0005])
    yaw = float(rng.uniform(*CUBE_YAW))
    sim.place_object("cube", xyz, yaw)
    sim.step(5)
    return sim.object_pose("cube")[:3, 3].copy(), yaw


def cube_in_base(sim: G1DSim) -> np.ndarray:
    return (np.linalg.inv(sim.base_pose_wla()) @ sim.object_pose("cube"))[:3, 3]


def run_expert(sim: G1DSim, rng: np.random.Generator, params: ExpertParams = ExpertParams(),
               on_step=None) -> dict:
    """Un épisode expert à partir de la pose de départ courante. Rend un résumé (succès, etc.)."""
    z0 = sim.object_pose("cube")[2, 3]
    T0 = sim.ee_pose_wla("right")
    R = T0[:3, :3]
    fingers = R[:, 0]                                   # axe des doigts, dans la base WLA
    c = cube_in_base(sim)
    grasp = c - params.grasp_back * fingers
    pre = grasp - params.pre_back * fingers + np.array([0.0, 0.0, params.pre_up])
    lift = grasp + np.array([0.0, 0.0, params.lift])
    p0 = T0[:3, 3]
    hold_left = sim.ee_pose_wla("left")

    plan = []                                            # (position droite, fermeture pince droite)
    plan += [(p0, 0.0)] * params.t_hold0
    plan += [(p, 0.0) for p in _minjerk(p0, pre, params.t_pre)]
    plan += [(p, 0.0) for p in _minjerk(pre, grasp, params.t_approach)]
    plan += [(grasp, c_) for c_ in np.linspace(0.0, 1.0, params.t_close + 1)[1:]]
    plan += [(p, 1.0) for p in _minjerk(grasp, lift, params.t_lift)]
    plan += [(lift, 1.0)] * params.t_hold1

    refused = 0
    for t, (pos, closure) in enumerate(plan):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, pos
        refused += not sim.track_ee_wla("right", T)
        sim.track_ee_wla("left", hold_left)
        sim.set_gripper("right", closure)
        sim.set_gripper("left", 0.0)
        if on_step is not None:
            on_step(t, {"right": closure, "left": 0.0})
        sim.step()
    lifted = float(sim.object_pose("cube")[2, 3] - z0)
    return {"success": lifted > LIFT_SUCCESS, "lifted": lifted, "steps": len(plan), "ik_refused": refused,
            "cube_base": np.round(c, 4).tolist()}


if __name__ == "__main__":
    import argparse
    import time

    from g1d_sim import SCENE_CUBE_XML

    ap = argparse.ArgumentParser(description="Taux de réussite de l'expert scripté, sans rendu.")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grasp-back", type=float, default=ExpertParams.grasp_back)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    sim = G1DSim(scene_xml=SCENE_CUBE_XML)
    ok, t = 0, time.time()
    for i in range(a.n):
        sim.reset()
        sim.go_ready()
        sample_cube(sim, rng)
        r = run_expert(sim, rng, ExpertParams(grasp_back=a.grasp_back))
        ok += r["success"]
        print(f"{i:3d} {'OK ' if r['success'] else 'RATÉ'} soulevé {r['lifted']*100:5.1f} cm  cube {r['cube_base']}  IK refusées {r['ik_refused']}", flush=True)
    print(f"réussite {ok}/{a.n}  ({time.time()-t:.0f} s)")
