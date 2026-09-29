"""Tests hors robot du pont téléop <-> WLA (policy_bridge.py) :

1. conversion base WLA -> repère IK de la téléop, comparée à la FK de l'IK elle-même (G1_29_ArmIK) :
   identique (le modèle de la téléop, g1_body29_hand14, a le poignet du G1-D : 0.046 m) ;
2. aller-retour wla_to_ik / ik_to_wla ;
3. correspondance gâchette <-> angle Dex1.

    cd teleoperation/Tele_OP/xr_teleoperate/teleop && python test_policy_bridge.py
"""
import os
import sys

import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.abspath(".."))
from policy_bridge import DEX1_OPEN, TRIGGER_MAX, TRIGGER_MIN, FrameConverter, dex1_to_trigger  # noqa: E402
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK  # noqa: E402

ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
model, data = ik.reduced_robot.model, ik.reduced_robot.data
conv = FrameConverter()
rng = np.random.default_rng(0)
worst_p, worst_r, worst_rt = 0.0, 0.0, 0.0
for _ in range(50):
    q = np.clip(rng.uniform(-1.0, 1.0, 14), model.lowerPositionLimit, model.upperPositionLimit)
    pin.framesForwardKinematics(model, data, q)
    for i, s in enumerate(("left", "right")):
        T_ik = data.oMf[model.getFrameId("L_ee" if s == "left" else "R_ee")].homogeneous
        qs = q[:7] if s == "left" else q[7:]
        pitch = rng.uniform(-0.2, 0.3)
        T_conv = conv.wla_to_ik(s, conv.ee_wla(s, qs, pitch), pitch)
        worst_p = max(worst_p, np.linalg.norm(T_conv[:3, 3] - T_ik[:3, 3]))
        worst_r = max(worst_r, np.abs(T_conv[:3, :3] - T_ik[:3, :3]).max())
        back = conv.ik_to_wla(s, T_conv, pitch)
        worst_rt = max(worst_rt, np.abs(back - conv.ee_wla(s, qs, pitch)).max())
print(f"1. écart position conversion / IK téléop : max {worst_p * 1000:.2f} mm (le modèle IK de la téléop a le même poignet que le G1-D)")
print(f"   écart rotation : max {worst_r:.2e}")
print(f"2. aller-retour wla_to_ik / ik_to_wla : max {worst_rt:.2e}")
g = np.linspace(0, DEX1_OPEN, 5)
tr = [dex1_to_trigger(x) for x in g]
back = np.interp(tr, [TRIGGER_MIN, TRIGGER_MAX], [0.0, DEX1_OPEN])
print(f"3. gâchette : Dex1 {np.round(g, 2)} -> gâchette {np.round(tr, 2)} -> Dex1 {np.round(back, 2)}")
ok = worst_p < 1e-6 and worst_r < 1e-6 and worst_rt < 1e-9 and np.allclose(back, g)
print("RÉSULTAT 1-3 :", "OK" if ok else "ÉCHEC")


# ---------------------------------------------------------------- 4. logique de correction (faux serveur)
import threading  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from scipy.spatial.transform import Rotation  # noqa: E402
from websockets.sync.server import serve  # noqa: E402

import policy_bridge as PB  # noqa: E402

N_QUERIES = [0]
MODE = {"m": "hold", "offset": 0.0}      # hold | ahead (vise 20 cm devant) | error | nan | mute


def handler(ws):
    p = PB.msgpack_numpy.Packer()
    ws.send(p.pack({"env": "faux"}))
    for raw in ws:
        m = PB.msgpack_numpy.unpackb(raw)
        if m["type"] != "get_action":
            ws.send(p.pack({}))
            continue
        N_QUERIES[0] += 1
        if MODE["m"] == "error":
            ws.send("Traceback: erreur simulée")
            continue
        if MODE["m"] == "mute":
            import time as _t
            _t.sleep(PB.RECV_TIMEOUT + 1.0)
            continue
        out = {}
        for s in ("left", "right"):          # « politique » immobile : garde la pose mesurée, pince ouverte
            e = np.asarray(m["obs"][f"observation.state.{s}_ee_6d"], float)
            R = np.stack([e[3:6], e[6:9], np.cross(e[3:6], e[6:9])], 1)
            x = np.r_[e[:3], Rotation.from_matrix(R).as_euler("xyz")]
            if MODE["m"] == "ahead":
                x[0] += 0.20
            if MODE["m"] == "nan":
                x[0] = np.nan
            out[f"action.{s}_ee_rpy"] = np.tile(x, (1, 30, 1)).astype(np.float32)
            out[f"action.{s}_gripper"] = np.full((1, 30, 1), 5.4, np.float32)
        ws.send(p.pack(out))


srv = serve(handler, "127.0.0.1", 8612, max_size=None)
threading.Thread(target=srv.serve_forever, daemon=True).start()
br = PB.PolicyBridge("ws://127.0.0.1:8612", "test", max_speed=0.15, torso_pitch=0.166)
q = np.array([-0.235, 0.848, 0.25, 0.115, -0.55, 0.201, -1.32, -0.164, -0.972, -0.262, -0.371, 0.619, 0.625, 1.135])
imgs = {k: np.zeros((480, 640, 3), np.uint8) for k in ("head", "left_wrist", "right_wrist")}
grip = np.array([5.4, 5.4])
P0 = np.eye(4)
td = SimpleNamespace(left_ctrl_squeezeValue=0.0, right_ctrl_squeezeValue=0.0, left_ctrl_triggerValue=10.0,
                     right_ctrl_triggerValue=10.0, left_wrist_pose=P0.copy(), right_wrist_pose=P0.copy())
o = br.step(imgs, q, grip, None, td)
base = o["right"].copy()
print(f"4a. sans correction : intervention={o['intervention']}, gâchette droite {o['trigger']['right']:.1f} (7 = ouverte)")
td.right_ctrl_squeezeValue = 1.0
br.step(imgs, q, grip, None, td)                          # appui : référence de la manette
td.right_wrist_pose = P0.copy()
td.right_wrist_pose[0, 3] += 0.05                           # la manette avance de 5 cm
moves = []
for _ in range(20):
    o = br.step(imgs, q, grip, None, td)
    moves.append(o["right"][0, 3] - base[0, 3])
print(f"4b. correction +5 cm : intervention={o['intervention']}, cible avancée de {moves[0]*1000:.1f} mm au 1er pas "
      f"(borne {0.15/30*1000:.1f} mm), {moves[-1]*1000:.1f} mm après 20 pas")
td.right_ctrl_triggerValue = 5.0
o = br.step(imgs, q, grip, None, td)
print(f"4c. gâchette pressée pendant la correction : gâchette droite {o['trigger']['right']:.1f} (5 = fermée)")
nq = N_QUERIES[0]
td.right_ctrl_squeezeValue, td.right_ctrl_triggerValue = 0.0, 10.0
o = br.step(imgs, q, grip, None, td)
print(f"4d. relâchement : intervention={o['intervention']}, nouvelle requête au modèle : {N_QUERIES[0] > nq}")
ok2 = (not o["intervention"]) and N_QUERIES[0] > nq and abs(moves[0] - 0.005) < 1e-6 and abs(moves[-1] - 0.05) < 1e-3
print("RÉSULTAT 4 :", "OK" if ok2 else "ÉCHEC")

# ---------------------------------------------------------------- 5. protections (audit du 29/09)
res5 = {}


def fresh(**kw):
    td_ = SimpleNamespace(left_ctrl_squeezeValue=0.0, right_ctrl_squeezeValue=0.0, left_ctrl_triggerValue=10.0,
                          right_ctrl_triggerValue=10.0, left_wrist_pose=np.eye(4), right_wrist_pose=np.eye(4),
                          head_pose=np.eye(4))
    for k_, v_ in kw.items():
        setattr(td_, k_, v_)
    return td_


# 5a. correction tenue sur plusieurs chunks, manette immobile à +5 cm, suivi parfait : pas de dérive
MODE["m"] = "hold"
br.reset()
td = fresh()
qm = q.copy()
meas0 = br.measured_ik(qm, None)["right"]
br.step(imgs, qm, grip, None, td)
td.right_ctrl_squeezeValue = 1.0
br.step(imgs, qm, grip, None, td)
td.right_wrist_pose = np.eye(4)
td.right_wrist_pose[0, 3] += 0.05
ik5 = G1_29_ArmIK(Unit_Test=False, Visualization=False)
for _ in range(150):                                   # suivi parfait : les angles suivent la cible IK
    o = br.step(imgs, qm, grip, None, td)
    sol, _ = ik5.solve_ik(o["left"], o["right"], qm, np.zeros(14))
    qm = sol
drift = br.measured_ik(qm, None)["right"][0, 3] - meas0[0, 3]
res5["a"] = abs(drift - 0.05) < 0.01
print(f"5a. correction +5 cm tenue 150 pas (5 chunks) : bras avancé de {drift*1000:.0f} mm (attendu ~50, sans dérive)")

# 5b. premier pas d'un essai : borne de vitesse depuis la pose MESURÉE
MODE["m"] = "ahead"
br.reset()
m0 = br.measured_ik(q, None)["right"]
o = br.step(imgs, q, grip, None, fresh())
jump = float(np.linalg.norm(o["right"][:3, 3] - m0[:3, 3]))
res5["b"] = jump <= br.max_step + 1e-9
print(f"5b. politique qui vise 20 cm plus loin : saut au 1er pas {jump*1000:.2f} mm (borne {br.max_step*1000:.2f})")

# 5c. serveur : erreur texte, actions NaN, serveur muet -> PolicyError
for mode in ("error", "nan", "mute"):
    MODE["m"] = mode
    br.pause()
    try:
        br.step(imgs, q, grip, None, fresh())
        res5["c_" + mode] = False
    except PB.PolicyError as e:
        res5["c_" + mode] = True
        print(f"5c. serveur « {mode} » -> PolicyError : {str(e).splitlines()[0][:70]}")
MODE["m"] = "hold"
br.reset()                                             # reconnexion après le serveur muet
print("    reconnexion après le serveur muet : OK")

# 5d. l'opérateur se penche (tête +10 cm) pendant la correction : le bras ne bouge pas
br.reset()
td = fresh()
br.step(imgs, q, grip, None, td)
td.right_ctrl_squeezeValue = 1.0
o1 = br.step(imgs, q, grip, None, td)
td.right_wrist_pose = np.eye(4)
td.right_wrist_pose[0, 3] -= 0.10                      # la téléop rend le poignet relatif à la tête
td.head_pose = np.eye(4)
td.head_pose[0, 3] += 0.10
for _ in range(10):
    o2 = br.step(imgs, q, grip, None, td)
res5["d"] = np.linalg.norm(o2["right"][:3, 3] - o1["right"][:3, 3]) < 1e-6
print(f"5d. tête +10 cm, manette immobile dans le monde : bras déplacé de "
      f"{np.linalg.norm(o2['right'][:3, 3] - o1['right'][:3, 3])*1000:.2f} mm")

# 5e. perte de suivi (saut de 40 cm de la manette) : décalage figé
td.right_wrist_pose = np.eye(4)
td.right_wrist_pose[0, 3] += 0.40 - 0.10
for _ in range(10):
    o3 = br.step(imgs, q, grip, None, td)
res5["e"] = np.linalg.norm(o3["right"][:3, 3] - o2["right"][:3, 3]) < 1e-6
print(f"5e. saut de manette de 40 cm : bras déplacé de {np.linalg.norm(o3['right'][:3, 3] - o2['right'][:3, 3])*1000:.2f} mm")

# 5f. gâchette effleurée (8/10) pendant la correction : la pince reste à la politique (ouverte, 7)
td = fresh(right_ctrl_squeezeValue=1.0, right_ctrl_triggerValue=8.0)
br.reset()
o4 = br.step(imgs, q, grip, None, td)
res5["f"] = abs(o4["trigger"]["right"] - 7.0) < 1e-9
print(f"5f. gâchette à 8/10 pendant la correction : gâchette envoyée {o4['trigger']['right']:.1f} (7 = politique, ouverte)")
ok3 = all(res5.values())
print("RÉSULTAT 5 :", "OK" if ok3 else f"ÉCHEC {[k for k, v in res5.items() if not v]}")
br.close()
srv.shutdown()
sys.exit(0 if (ok and ok2 and ok3) else 1)
