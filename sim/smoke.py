"""Vérifie la sim de bout en bout et écrit les 3 vues caméra en PNG.

    MUJOCO_GL=egl python sim/smoke.py            # depuis tencent/
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1d_sim import CAMERAS, SIDES, G1DSim  # noqa: E402
from g1d_sim.robot import READY_TCP, TORSO_PITCH  # noqa: E402

OUT = Path(__file__).resolve().parent / "smoke_out"


def check(ok: bool, msg: str) -> bool:
    print(("  OK   " if ok else "  ÉCHEC ") + msg, flush=True)
    return ok


def main() -> int:
    sim = G1DSim()
    results = []
    print(f"scène chargée : {sim.m.nq} qpos, {sim.m.nu} actionneurs, "
          f"contrôle {sim.control_hz:.0f} Hz ({sim.sous_pas} sous-pas)")

    # 1. repos : les bras tiennent le tuck sous gravité (feedforward branché)
    q0 = {s: sim.arm_q(s) for s in SIDES}
    sim.step(30)
    for s in SIDES:
        drift = np.abs(sim.arm_q(s) - q0[s]).max()
        results.append(check(drift < 0.02, f"{s}: dérive au repos 1 s = {drift:.4f} rad"))

    # 2. FK -> IK aller-retour sur la pose courante
    for s in SIDES:
        T = sim.tcp_pose(s)
        q, ok = sim.solve_ik(s, T, follow=False)
        err = np.linalg.norm(sim.tcp_pose(s, q)[:3, 3] - T[:3, 3]) * 1000
        results.append(check(ok and err < 1.0, f"{s}: IK aller-retour {err:.3f} mm"))

    # 3. pose de travail, puis suivi cartésien à 30 Hz via track_tcp
    piece0 = sim.object_pose()[:3, 3].copy()
    sim.go_ready()
    moved = np.linalg.norm(sim.object_pose()[:3, 3] - piece0) * 1000
    results.append(check(moved < 5.0, f"go_ready ne touche pas la pièce (déplacée de {moved:.1f} mm)"))
    for s in SIDES:
        err = np.linalg.norm(sim.tcp_pose(s)[:3, 3] - READY_TCP[s]) * 1000
        results.append(check(err < 10.0, f"{s}: pose de travail atteinte à {err:.1f} mm"))
    # 4. pinces : fermeture à vide puis réouverture, en pose de travail AU-DESSUS de la pièce.
    #    (Testées après l'approche, le doigt droit se referme à 1 cm du centre de la pièce et
    #    l'accroche : le test passait ou échouait au hasard.)
    for s in SIDES:
        sim.set_gripper(s, 1.0)
    sim.step(30)
    closed = {s: sim.gripper(s) for s in SIDES}
    for s in SIDES:
        sim.set_gripper(s, 0.0)
    sim.step(30)
    for s in SIDES:
        results.append(check(closed[s] > 0.9 and sim.gripper(s) < 0.1,
                             f"{s}: pince fermée {closed[s]:.2f} / rouverte {sim.gripper(s):.2f}"))

    # approche de prise : 8 cm vers la table, 3 cm vers le robot, les deux bras ensemble
    # (pince verticale + colonne basse : atteignable pour x ∈ [0.25, 0.35], z ∈ [0.78, 0.90])
    T0 = {s: sim.tcp_pose(s) for s in SIDES}
    T1 = {s: T0[s].copy() for s in SIDES}
    for s in SIDES:
        T1[s][:3, 3] += [-0.03, 0.0, -0.08]
    n_ok = {s: 0 for s in SIDES}
    ramp = np.linspace(0, 1, 45)[1:]                      # 1,5 s de rampe cartésienne
    for a in ramp:
        for s in SIDES:
            Ta = T0[s].copy()
            Ta[:3, 3] = (1 - a) * T0[s][:3, 3] + a * T1[s][:3, 3]
            n_ok[s] += sim.track_tcp(s, Ta)
        sim.step()
    sim.step(30)
    for s in SIDES:
        err = np.linalg.norm(sim.tcp_pose(s)[:3, 3] - T1[s][:3, 3]) * 1000
        results.append(check(err < 5.0 and n_ok[s] == len(ramp),
                             f"{s}: suivi cartésien {n_ok[s]}/{len(ramp)} IK, erreur finale {err:.1f} mm"))

    # hors d'atteinte : la consigne doit être TENUE, pas transformée en saut
    q_before = sim.arm_q("left")
    T_far = sim.tcp_pose("left")
    T_far[:3, 3] += [0.60, 0.0, 0.40]
    refused = not sim.track_tcp("left", T_far)
    sim.step(10)
    jump = np.abs(sim.arm_q("left") - q_before).max()
    results.append(check(refused and jump < 0.01,
                         f"cible hors d'atteinte refusée, bras tenu (dérive {jump:.4f} rad)"))

    # 5. rollback : sauvegarder, bouger, restaurer -> état identique
    snap = sim.save_state()
    qa = sim.arm_q("right")
    sim.set_arm_target("right", qa + 0.3)
    sim.step(15)
    sim.restore_state(snap)
    results.append(check(np.allclose(sim.arm_q("right"), qa), "rollback save/restore_state"))

    # 6. les trois caméras du VLA
    try:
        import imageio.v3 as iio
    except ImportError:
        iio = None
    OUT.mkdir(exist_ok=True)
    for key, img in sim.render_all().items():
        ok = img.ndim == 3 and img.shape[2] == 3 and img.std() > 1.0
        results.append(check(ok, f"caméra {key} ({CAMERAS[key]}) {img.shape}, écart-type {img.std():.1f}"))
        if iio is not None:
            iio.imwrite(OUT / f"{key}.png", img)
    if iio is not None:
        print(f"vues écrites dans {OUT}/")

    # 7. aucune divergence : MuJoCo remet l'état à zéro EN SILENCE sur un NaN (vu : le buste
    #    retombait à 0 et les tests passaient quand même)
    import mujoco
    n_bad = int(sim.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    results.append(check(n_bad == 0, f"physique stable (divergences détectées : {n_bad})"))
    pitch_ok = abs(sim.torso_pitch() - TORSO_PITCH) < 0.01
    results.append(check(pitch_ok, f"buste incliné à {sim.torso_pitch():.3f} rad (cible {TORSO_PITCH})"))

    print(f"objet 'piece' (GT) en {np.round(sim.object_pose()[:3, 3], 3)}")
    sim.close()
    print(f"{sum(results)}/{len(results)} vérifications passées")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
