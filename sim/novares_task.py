"""Tâche Novares en sim : saisir la pièce Novares par sa prise PEINTE et la soulever.

La prise vient des zones peintes par l'opérateur dans mpc_any (``<objet>.zones.json``, indices de
faces du STL par mors), avec la même logique que ``mpc_any/src/oc_svfp/perception/zones_prise.py`` :

* axe des mors = droite entre les centroïdes (pondérés par l'aire) des deux zones ; centre = milieu ;
* direction d'approche libre autour de cet axe, échantillonnée (24 angles) ;
* rejet des approches dont la paume traverse la pièce (rayon sur le CAD, 60 mm) ;
* la plus VERTICALE d'abord ; ici on ajoute le filtre « atteignable par l'IK du bras droit ».

Conventions de pince identiques (mpc_any ↔ effecteur WLA) : x = approche (le long des doigts),
y = axe des mors. Le point effecteur WLA est à ``EE_BACK`` derrière le centre de prise.

Variance de placement : la pièce garde sa pose STABLE sur la table (celle de la scène) et reçoit un
décalage (x, y) et un lacet tirés au hasard dans la base WLA.

⚠ Fichiers non versionnés (pièce client) : ``sim/assets/meshes/_Novares_Piece1_centered.stl`` et
``.zones.json`` (copié de ``mpc_any/configs/projects/usine/novares.zones.json``, même STL, md5 égal).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from cube_task import _minjerk
from g1d_sim import SCENE_XML, G1DSim  # noqa: F401  (SCENE_XML : scène avec la pièce Novares)
from g1d_sim.robot import ASSETS

INSTRUCTION = "pick up the black part"
MESH = ASSETS / "meshes" / "_Novares_Piece1_centered.stl"
ZONES = ASSETS / "meshes" / "_Novares_Piece1_centered.zones.json"
#: décalage du point effecteur WLA derrière le centre de prise, le long de l'approche (m)
EE_BACK = 0.035
#: variance de placement autour de la pose de la scène, dans la base WLA
DX = (-0.03, 0.03)
DY = (-0.03, 0.03)
DYAW = (-0.35, 0.35)
LIFT_SUCCESS = 0.05
N_ANGLES = 24
LONG_DOIGT = 0.060


@dataclass
class NovaresParams:
    pre_back: float = 0.08      # pré-prise : recul le long de l'approche
    lift: float = 0.10
    raise_z: float = 0.08       # montée verticale de la main avant le transfert
    t_hold0: int = 8
    t_raise: int = 20
    t_pre: int = 45             # transfert ARTICULAIRE vers la pré-prise (au-dessus de la pièce)
    t_approach: int = 30
    t_close: int = 20
    t_lift: int = 40
    t_hold1: int = 15


class NovaresGrasp:
    """Géométrie de la prise peinte, dans le repère du STL (mètres)."""

    def __init__(self) -> None:
        import trimesh
        self.mesh = trimesh.load(str(MESH), process=False)
        self.mesh.apply_scale(0.001)
        z = json.loads(ZONES.read_text())
        pr = z["prises"][0]
        cA, cB = self._centroid(pr["mors_a"]), self._centroid(pr["mors_b"])
        ax = cB - cA
        self.width = float(np.linalg.norm(ax))
        self.axis = ax / self.width
        self.center = 0.5 * (cA + cB)
        k = int(np.argmin(np.abs(self.axis)))
        e = np.zeros(3)
        e[k] = 1.0
        u = np.cross(self.axis, e)
        self.u = u / np.linalg.norm(u)
        self.v = np.cross(self.axis, self.u)
        inside = bool(self.mesh.contains(self.center[None])[0])
        # approches libres de la pièce (indépendantes de sa pose : calculées une fois dans le STL)
        self.free = []
        for k in range(N_ANGLES):
            th = 2 * np.pi * k / N_ANGLES
            f = np.cos(th) * self.u + np.sin(th) * self.v
            locs, _, _ = self.mesh.ray.intersects_location(self.center[None], (-f)[None], multiple_hits=True)
            dist = np.sort(np.linalg.norm(np.asarray(locs).reshape(-1, 3) - self.center, axis=1))
            dist = dist[dist > 1e-6]
            if inside:
                dist = dist[1:]
            if not (len(dist) and dist[0] < LONG_DOIGT):
                self.free.append(f)

    def _centroid(self, faces) -> np.ndarray:
        f = np.asarray(faces, int)
        c, w = self.mesh.triangles_center[f], self.mesh.area_faces[f]
        return (c * w[:, None]).sum(0) / w.sum()


def _stl_to_world(sim: G1DSim):
    """(point, direction) STL -> monde, pour la pose courante de la pièce."""
    m, d = sim.m, sim.d
    gid = m.geom("piece_visu").id
    mid = m.geom_dataid[gid]
    q = m.mesh_quat[mid]
    Rq = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    p = m.mesh_pos[mid]
    Rg, pg = d.geom_xmat[gid].reshape(3, 3), d.geom_xpos[gid]
    return (lambda v: Rg @ (Rq.T @ (v - p)) + pg), (lambda v: Rg @ (Rq.T @ v))


def grasp_candidates(sim: G1DSim, g: NovaresGrasp, side: str = "right") -> list:
    """Prises (4×4 effecteur WLA, base WLA) triées : la plus verticale atteignable d'abord."""
    pt, dr = _stl_to_world(sim)
    Bi = np.linalg.inv(sim.base_pose_wla())
    B = sim.base_pose_wla()
    X = np.linalg.inv(sim._tcp_to_ee(side))
    c = (Bi @ np.r_[pt(g.center), 1.0])[:3]
    ax = Bi[:3, :3] @ dr(g.axis)
    out = []
    for f_stl in g.free:
        f = Bi[:3, :3] @ dr(f_stl)
        for sgn in (1.0, -1.0):
            y = sgn * ax
            R = np.c_[f, y, np.cross(f, y)]
            T, Tpre = np.eye(4), np.eye(4)
            T[:3, :3] = Tpre[:3, :3] = R
            T[:3, 3] = c - EE_BACK * f
            Tpre[:3, 3] = T[:3, 3] - NovaresParams.pre_back * f
            ok = all(sim.solve_ik(side, B @ M @ X, q0=sim.arm_q(side), follow=False)[1] for M in (Tpre, T))
            if ok:
                out.append((float(f @ np.array([0.0, 0.0, -1.0])), T))
    out.sort(key=lambda s: -s[0])
    return [T for _, T in out]


def sample_piece(sim: G1DSim, rng: np.random.Generator, pose0: np.ndarray) -> None:
    """Replace la pièce : pose stable ``pose0`` (monde) + décalage (x, y) et lacet dans la base."""
    B = sim.base_pose_wla()
    dx, dy, dyaw = rng.uniform(*DX), rng.uniform(*DY), rng.uniform(*DYAW)
    Rz = Rotation.from_euler("z", dyaw).as_matrix()
    T = pose0.copy()
    T[:3, :3] = Rz @ pose0[:3, :3]
    T[:3, 3] = pose0[:3, 3] + B[:3, :3] @ np.array([dx, dy, 0.0])
    m, d = sim.m, sim.d
    j = m.body("piece").jntadr[0]
    qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    d.qpos[qa:qa + 3] = T[:3, 3]
    d.qpos[qa + 3:qa + 7] = [q[3], q[0], q[1], q[2]]
    d.qvel[va:va + 6] = 0.0
    import mujoco
    mujoco.mj_forward(m, d)
    sim.step(10)


def run_expert(sim: G1DSim, g: NovaresGrasp, params: NovaresParams = NovaresParams(),
               on_step=None) -> dict:
    """Un épisode expert depuis la pose de départ ; rend un résumé (succès, etc.)."""
    z0 = sim.object_pose("piece")[2, 3]
    cands = grasp_candidates(sim, g)
    if not cands:
        return {"success": False, "lifted": 0.0, "steps": 0, "ik_refused": -1, "reason": "aucune prise atteignable"}
    Tg = cands[0]
    f = Tg[:3, 0]
    T0 = sim.ee_pose_wla("right")
    hold_left = sim.ee_pose_wla("left")
    p0, R0 = T0[:3, 3], T0[:3, :3]
    grasp = Tg[:3, 3]
    pre = grasp - params.pre_back * f
    lift = grasp + np.array([0.0, 0.0, params.lift])
    up = p0 + np.array([0.0, 0.0, params.raise_z])

    # Trajet en trois temps. Une rotation cartésienne près de la table balaie la pièce avec
    # l'avant-bras (mesuré : pièce poussée, voire tombée) : on MONTE, puis transfert ARTICULAIRE
    # vers la pré-prise (au-dessus de la pièce), puis descente le long de l'approche.
    B, X = sim.base_pose_wla(), np.linalg.inv(sim._tcp_to_ee("right"))
    Tup, Tpre = np.eye(4), np.eye(4)
    Tup[:3, :3], Tup[:3, 3] = R0, up
    Tpre[:3, :3], Tpre[:3, 3] = Tg[:3, :3], pre
    q_up, ok1 = sim.solve_ik("right", B @ Tup @ X, q0=sim.arm_q("right"), follow=True)
    q_pre, ok2 = sim.solve_ik("right", B @ Tpre @ X, q0=q_up, follow=False)
    if not (ok1 and ok2):
        return {"success": False, "lifted": 0.0, "steps": 0, "ik_refused": -1, "reason": "transfert inatteignable"}

    plan = []                                   # ("cart", position, rotation, fermeture) | ("joint", q, None, fermeture)
    plan += [("cart", p0, R0, 0.0)] * params.t_hold0
    plan += [("cart", p, R0, 0.0) for p in _minjerk(p0, up, params.t_raise)]
    plan += [("joint", q, None, 0.0) for q in _minjerk(q_up, q_pre, params.t_pre)]
    plan += [("cart", p, Tg[:3, :3], 0.0) for p in _minjerk(pre, grasp, params.t_approach)]
    plan += [("cart", grasp, Tg[:3, :3], c) for c in np.linspace(0.0, 1.0, params.t_close + 1)[1:]]
    plan += [("cart", p, Tg[:3, :3], 1.0) for p in _minjerk(grasp, lift, params.t_lift)]
    plan += [("cart", lift, Tg[:3, :3], 1.0)] * params.t_hold1

    refused = 0
    for t, (kind, p, R, closure) in enumerate(plan):
        if kind == "joint":
            sim.set_arm_target("right", p)
        else:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, p
            refused += not sim.track_ee_wla("right", T)
        sim.track_ee_wla("left", hold_left)
        sim.set_gripper("right", closure)
        sim.set_gripper("left", 0.0)
        if on_step is not None:
            on_step(t, {"right": closure, "left": 0.0})
        sim.step()
    lifted = float(sim.object_pose("piece")[2, 3] - z0)
    return {"success": lifted > LIFT_SUCCESS, "lifted": lifted, "steps": len(plan), "ik_refused": refused,
            "verticality": float(f @ np.array([0.0, 0.0, -1.0])), "n_candidates": len(cands)}


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Taux de réussite de l'expert Novares, sans rendu.")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    g = NovaresGrasp()
    print(f"prise peinte : largeur {g.width*1000:.1f} mm, {len(g.free)}/{N_ANGLES} approches libres de la pièce")
    sim = G1DSim()
    pose0 = sim.object_pose("piece")
    ok, t = 0, time.time()
    for i in range(a.n):
        sim.reset()
        sim.go_ready()
        sample_piece(sim, rng, pose0)
        r = run_expert(sim, g)
        ok += r["success"]
        print(f"{i:3d} {'OK ' if r['success'] else 'RATÉ'} soulevé {r['lifted']*100:5.1f} cm  "
              f"verticalité {r.get('verticality', 0):+.2f}  candidats {r.get('n_candidates', 0)}  "
              f"IK refusées {r['ik_refused']} {r.get('reason', '')}", flush=True)
    print(f"réussite {ok}/{a.n}  ({time.time()-t:.0f} s)")
