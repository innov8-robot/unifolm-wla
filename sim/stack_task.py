"""Tâche d'empilement en sim : saisir une pièce Novares (prise peinte) et l'EMBOÎTER sur une seconde
pièce identique, dans le même sens.

* Scène ``scene_g1d_stack.xml`` : deux pièces, ``piece`` (à saisir) et ``piece2`` (support).
* Emboîtement : décalage fixe de la pièce du dessus dans le repère de la pièce support, trouvé par
  recherche géométrique (hauteur minimale sans interpénétration) puis vérifié par lâchers :
  ``_Novares_Piece1_centered.stack.json`` (non versionné, à côté de la pièce). En sim : 17 mm vers la
  gauche et 16 mm plus haut ; un lâcher avec ±3 mm d'erreur retombe à moins de 3 mm de cette pose.
* Pose : pièces à plat, plaque sur la table et paroi courbe en haut (comme en réel) ; les zones
  peintes tombent sur les flancs de la paroi. L'ancienne scène les posait sur le dos.
* Variance : les deux pièces dans leur pose stable, positions et lacets tirés au hasard ; l'expert
  réaligne la pièce saisie sur le support pendant le transport (« même sens »).
* Expert : prise peinte (``novares_task``), levée, transport au-dessus de la pose emboîtée, descente,
  ouverture, retrait. Réussite : pièce emboîtée à moins de ``STACK_TOL`` et support resté en place.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import novares_task as N
from cube_task import _minjerk
from g1d_sim import G1DSim
from g1d_sim.robot import ASSETS

INSTRUCTION = "stack the black part on the other one"
SCENE_STACK_XML = ASSETS / "scene_g1d_stack.xml"
NEST_FILE = ASSETS / "meshes" / "_Novares_Piece1_centered.stack.json"
#: variance, dans la base WLA, autour de la pose de la pièce dans la scène
PICK_DX, PICK_DY, PICK_DYAW = (-0.03, 0.03), (-0.02, 0.02), (-0.35, 0.35)
SUP_DX, SUP_DY, SUP_DYAW = (0.0, 0.04), (0.12, 0.15), (-0.35, 0.35)
STACK_TOL = 0.008           # écart de position max à la pose emboîtée (m)
STACK_TOL_DEG = 10.0
SUPPORT_MOVE_TOL = 0.01
FULL_ORIENTATION = True


@dataclass
class StackParams(N.NovaresParams):
    carry_up: float = 0.08      # hauteur de transport au-dessus de la pose emboîtée
    place_clear: float = 0.005  # lâcher 5 mm au-dessus de la pose emboîtée : pièces à plat, les
                                # doigts sur les flancs de la paroi restent hors de la pièce du
                                # dessous ; lâchée de 2 cm, elle bascule d'environ 25° (mesuré).
                                # Balayage sur 20 essais : 3 mm 5, 5 mm 8, 8 mm 7, 12 mm 3
    t_carry: int = 50
    t_place: int = 30
    t_open: int = 15
    t_retreat: int = 25


def nest_offset() -> np.ndarray:
    """Pose emboîtée de la pièce du dessus dans le repère du support. Le fichier a été calculé avec
    les pièces sur le dos (paroi courbe en bas) ; retournées, plaque à plat comme en réel, c'est le
    support qui est « dessus » dans cette relation : on prend l'inverse (translation pure)."""
    T = np.eye(4)
    T[:3, 3] = -np.asarray(json.loads(NEST_FILE.read_text())["offset_in_support_frame_m"])
    return T


def _set_pose(sim: G1DSim, body: str, T: np.ndarray) -> None:
    m, d = sim.m, sim.d
    j = m.body(body).jntadr[0]
    qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    d.qpos[qa:qa + 3] = T[:3, 3]
    d.qpos[qa + 3:qa + 7] = [q[3], q[0], q[1], q[2]]
    d.qvel[va:va + 6] = 0.0


def place_pieces(sim: G1DSim, rng: np.random.Generator, pose0: np.ndarray) -> None:
    """Pose stable ``pose0`` (monde) + décalages et lacets aléatoires pour les deux pièces."""
    Bm = sim.base_pose_wla()[:3, :3]
    for body, (rx, ry, ryaw) in (("piece", (PICK_DX, PICK_DY, PICK_DYAW)), ("piece2", (SUP_DX, SUP_DY, SUP_DYAW))):
        T = pose0.copy()
        T[:3, :3] = Rotation.from_euler("z", rng.uniform(*ryaw)).as_matrix() @ pose0[:3, :3]
        T[:3, 3] = pose0[:3, 3] + Bm @ np.array([rng.uniform(*rx), rng.uniform(*ry), 0.0])
        _set_pose(sim, body, T)
    mujoco.mj_forward(sim.m, sim.d)
    sim.step(10)


def stack_error(sim: G1DSim) -> tuple[float, float]:
    """(écart de position en m, écart d'angle en degrés) de la pièce à sa pose emboîtée."""
    rel = np.linalg.inv(sim.object_pose("piece2") @ nest_offset()) @ sim.object_pose("piece")
    return float(np.linalg.norm(rel[:3, 3])), float(np.degrees(np.linalg.norm(Rotation.from_matrix(rel[:3, :3]).as_rotvec())))


def _execute(sim: G1DSim, plan: list, t0: int, hold_left: np.ndarray, on_step) -> int:
    """Exécute un morceau de plan ; rend le nombre de refus d'IK du bras droit."""
    refused = 0
    for k, (kind, p, R, closure) in enumerate(plan):
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
            on_step(t0 + k, {"right": closure, "left": 0.0})
        sim.step()
    return refused


def _place_pose(sim: G1DSim, clear: float) -> np.ndarray:
    """Effecteur (base WLA) qui amène la pièce TENUE à sa pose emboîtée + ``clear`` en z, calculé
    avec la pose RÉELLE de la pièce dans la pince (elle glisse de ~1 cm et ~10° à la levée).

    Seuls la position et le LACET sont corrigés : imposer aussi l'inclinaison de la pièce rendait
    la pose de poignet souvent inatteignable, et l'emboîtement tolère quelques degrés."""
    Bi = np.linalg.inv(sim.base_pose_wla())
    P = Bi @ sim.object_pose("piece")
    E = sim.ee_pose_wla("right")
    target = (Bi @ sim.object_pose("piece2")) @ nest_offset()
    # lacet qui aligne l'axe x de la pièce tenue sur celui de la cible (projetés à l'horizontale)
    a = np.arctan2(target[1, 0], target[0, 0]) - np.arctan2(P[1, 0], P[0, 0])
    a = (a + np.pi) % (2 * np.pi) - np.pi
    Rz = Rotation.from_euler("z", a).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rz @ E[:3, :3]
    # position : la pièce, tournée du même lacet autour de l'effecteur, doit tomber sur la cible
    p_rel = Rz @ (P[:3, 3] - E[:3, 3])
    T[:3, 3] = target[:3, 3] - p_rel
    T[2, 3] += clear
    if FULL_ORIENTATION:
        # pièces à plat : elle pivote dans la pince autour de l'axe des mors (~20°) ; on corrige
        # l'orientation complète si le poignet l'atteint, sinon on garde le lacet seul
        Tf = target @ np.linalg.inv(P) @ E
        Tf[2, 3] += clear
        B, X = sim.base_pose_wla(), np.linalg.inv(sim._tcp_to_ee("right"))
        if sim.solve_ik("right", B @ Tf @ X, q0=sim.arm_q("right"), follow=False)[1]:
            return Tf
    return T


def _pick_grasp(sim: G1DSim, cands: list, params: StackParams):
    """Première prise (la plus verticale) dont le DÉPÔT est aussi atteignable : pièce supposée fixe
    dans la pince, amenée sur sa pose emboîtée (et au-dessus) par un simple lacet."""
    B, X = sim.base_pose_wla(), np.linalg.inv(sim._tcp_to_ee("right"))
    Bi = np.linalg.inv(B)
    P = Bi @ sim.object_pose("piece")
    target = (Bi @ sim.object_pose("piece2")) @ nest_offset()
    a = np.arctan2(target[1, 0], target[0, 0]) - np.arctan2(P[1, 0], P[0, 0])
    Rz = np.eye(4)
    Rz[:3, :3] = Rotation.from_euler("z", a).as_matrix()
    for Tg in cands:
        Tp = Rz @ Tg
        Tp[:3, 3] = target[:3, 3] + Rz[:3, :3] @ (Tg[:3, 3] - P[:3, 3])
        Tp[2, 3] += params.place_clear
        Tab = Tp.copy()
        Tab[2, 3] += params.carry_up
        if all(sim.solve_ik("right", B @ M @ X, q0=sim.arm_q("right"), follow=False)[1] for M in (Tp, Tab)):
            return Tg
    return None


def run_expert(sim: G1DSim, g: N.NovaresGrasp, params: StackParams = StackParams(), on_step=None) -> dict:
    """Expert en phases, REPLANIFIÉ avec l'état réel (sim) : saisie -> levée -> [mesure] ->
    transport au-dessus -> [mesure] -> descente -> ouverture -> retrait."""
    cands = N.grasp_candidates(sim, g)
    if not cands:
        return {"success": False, "lifted": 0.0, "steps": 0, "ik_refused": -1, "reason": "aucune prise atteignable"}
    B, X = sim.base_pose_wla(), np.linalg.inv(sim._tcp_to_ee("right"))
    sup0 = sim.object_pose("piece2")[:3, 3].copy()
    Tg = _pick_grasp(sim, cands, params)
    if Tg is None:
        return {"success": False, "lifted": 0.0, "steps": 0, "ik_refused": -1, "reason": "aucune prise avec dépôt atteignable"}
    f = Tg[:3, 0]
    T0 = sim.ee_pose_wla("right")
    hold_left = sim.ee_pose_wla("left")
    p0, R0 = T0[:3, 3], T0[:3, :3]
    grasp = Tg[:3, 3]
    pre = grasp - params.pre_back * f
    lift = grasp + np.array([0.0, 0.0, params.lift])
    up = p0 + np.array([0.0, 0.0, params.raise_z])
    Tup, Tpre = np.eye(4), np.eye(4)
    Tup[:3, :3], Tup[:3, 3] = R0, up
    Tpre[:3, :3], Tpre[:3, 3] = Tg[:3, :3], pre
    q_up, ok1 = sim.solve_ik("right", B @ Tup @ X, q0=sim.arm_q("right"), follow=True)
    q_pre, ok2 = sim.solve_ik("right", B @ Tpre @ X, q0=q_up, follow=False)
    if not (ok1 and ok2):
        return {"success": False, "lifted": 0.0, "steps": 0, "ik_refused": -1, "reason": "trajet inatteignable"}

    # 1. saisie + levée
    plan = [("cart", p0, R0, 0.0)] * params.t_hold0
    plan += [("cart", p, R0, 0.0) for p in _minjerk(p0, up, params.t_raise)]
    plan += [("joint", q, None, 0.0) for q in _minjerk(q_up, q_pre, params.t_pre)]
    plan += [("cart", p, Tg[:3, :3], 0.0) for p in _minjerk(pre, grasp, params.t_approach)]
    plan += [("cart", grasp, Tg[:3, :3], c) for c in np.linspace(0.0, 1.0, params.t_close + 1)[1:]]
    plan += [("cart", p, Tg[:3, :3], 1.0) for p in _minjerk(grasp, lift, params.t_lift)]
    t = 0
    refused = _execute(sim, plan, t, hold_left, on_step)
    t += len(plan)

    # 2. transport au-dessus de la pose emboîtée (calculée avec la pièce réellement tenue)
    Tp = _place_pose(sim, params.place_clear)
    above = Tp[:3, 3] + np.array([0.0, 0.0, params.carry_up])
    Tab = Tp.copy()
    Tab[:3, 3] = above
    if not all(sim.solve_ik("right", B @ M @ X, q0=sim.arm_q("right"), follow=False)[1] for M in (Tab, Tp)):
        return {"success": False, "lifted": 0.0, "steps": t, "ik_refused": -1, "reason": "dépôt inatteignable"}
    Tl = sim.ee_pose_wla("right")
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Tl[:3, :3], Tp[:3, :3]])))
    s = np.linspace(0.0, 1.0, params.t_carry + 1)[1:]
    s = 10 * s**3 - 15 * s**4 + 6 * s**5
    plan = [("cart", p, slerp(si).as_matrix(), 1.0) for si, p in zip(s, _minjerk(Tl[:3, 3], above, params.t_carry))]
    plan += [("cart", above, Tp[:3, :3], 1.0)] * 10                 # stabiliser avant de remesurer
    refused += _execute(sim, plan, t, hold_left, on_step)
    t += len(plan)

    # 3. descente vers la pose emboîtée REMESURÉE, ouverture, retrait
    Tp = _place_pose(sim, params.place_clear)
    Tn = sim.ee_pose_wla("right")
    retreat = Tp[:3, 3] + np.array([0.0, 0.0, params.carry_up])
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Tn[:3, :3], Tp[:3, :3]])))
    s = np.linspace(0.0, 1.0, params.t_place + 1)[1:]
    plan = [("cart", p, slerp(si).as_matrix(), 1.0) for si, p in zip(s, _minjerk(Tn[:3, 3], Tp[:3, 3], params.t_place))]
    plan += [("cart", Tp[:3, 3], Tp[:3, :3], c) for c in np.linspace(1.0, 0.0, params.t_open + 1)[1:]]
    plan += [("cart", p, Tp[:3, :3], 0.0) for p in _minjerk(Tp[:3, 3], retreat, params.t_retreat)]
    plan += [("cart", retreat, Tp[:3, :3], 0.0)] * params.t_hold1
    refused += _execute(sim, plan, t, hold_left, on_step)
    t += len(plan)

    sim.step(15)                                          # laisser la pièce se poser
    err, ang = stack_error(sim)
    moved = float(np.linalg.norm(sim.object_pose("piece2")[:3, 3] - sup0))
    ok = err < STACK_TOL and ang < STACK_TOL_DEG and moved < SUPPORT_MOVE_TOL
    return {"success": ok, "stack_err_mm": round(err * 1000, 1), "stack_err_deg": round(ang, 1),
            "support_moved_mm": round(moved * 1000, 1), "lifted": 0.0, "steps": t, "ik_refused": refused}


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Taux de réussite de l'expert d'empilement, sans rendu.")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    g = N.NovaresGrasp()
    sim = G1DSim(scene_xml=SCENE_STACK_XML)
    pose0 = sim.object_pose("piece").copy()
    ok, t = 0, time.time()
    for i in range(a.n):
        sim.reset()
        sim.go_ready()
        place_pieces(sim, rng, pose0)
        r = run_expert(sim, g)
        ok += r["success"]
        print(f"{i:3d} {'OK ' if r['success'] else 'RATÉ'} écart {r.get('stack_err_mm', '-')} mm / "
              f"{r.get('stack_err_deg', '-')}°  support déplacé {r.get('support_moved_mm', '-')} mm  "
              f"IK refusées {r['ik_refused']} {r.get('reason', '')}", flush=True)
    print(f"réussite {ok}/{a.n}  ({time.time()-t:.0f} s)")
