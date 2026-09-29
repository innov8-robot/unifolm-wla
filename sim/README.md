# sim — G1-D + Dex1-1 dans MuJoCo (extrait de mpc_any)

Sim minimale pour brancher Hy-VLA : la scène, les servos, la gravité, les pinces, les
3 caméras et la FK/IK des deux bras. Rien d'autre de mpc_any (collision, planif,
perception, BT, grasping) n'est repris.

```
sim/
├── assets/
│   ├── scene_g1d.xml      # copie de mpc_any/sim/assets/g1_d_description/scene_g1d.xml
│   ├── g1_d_dex1.urdf     # le robot (Pinocchio : FK/IK + gravité)
│   └── meshes/            # les 197 meshes RÉFÉRENCÉS (20 Mo au lieu de 66)
├── g1d_sim/
│   ├── robot.py           # G1DSim : physique, bras, pinces, caméras, save/restore
│   ├── kinematics.py      # ArmKinematics (Pinocchio, DLS) — mêmes constantes que mpc_any
│   └── camera.py          # SimCamera (RGB + depth mm)
└── smoke.py               # 15 vérifications + les 3 vues en PNG dans smoke_out/
```

## Lancer

```bash
conda activate unitree_lerobot          # mujoco 3.x + pinocchio 3.x + imageio
MUJOCO_GL=egl python sim/smoke.py       # depuis tencent/
```

## Utiliser

```python
from g1d_sim import G1DSim

sim = G1DSim(control_hz=30, image_size=(640, 480))   # reset = tuck + settle
sim.go_ready()                                       # pose de travail, pinces vers le bas

imgs = sim.render_all()        # {"top_head", "hand_left", "hand_right"} -> HxWx3 uint8
T = sim.tcp_pose("left")       # 4x4, repère monde MuJoCo
g = sim.gripper("left")        # fermeture mesurée ∈ [0, 1]

sim.track_tcp("left", T_cible) # 1 pas cartésien sûr (IK follow, tenue si hors d'atteinte)
sim.set_gripper("left", 1.0)   # 0 = ouverte, 1 = fermée
sim.step()                     # 1 pas de contrôle (17 pas de physique à 0,002 s)

snap = sim.save_state()        # rollback FlowPRO : restore_state(snap)
```

## Conventions (identiques à mpc_any)

| | |
|---|---|
| Caméras | `top_head` = `torso_rgbd` (fovy 65°), `hand_left/right` = `*_wrist_cam` (fovy 110°) |
| Bras | 7 DoF, ordre `shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw` |
| TCP | `*_gripper_base_link` + (0, 0.105, 0) — ⚠ 16 mm du vrai centre des patins (`TCP_DOIGTS_M`) |
| Pince | consigne Joint1_1 de -0.018 (ouverte) à 0.0245 (fermée) ; Joint2_1 couplé par `<equality>` |
| Colonne / buste | tenus à 0 (butée basse) |
| Contrôle | 30 Hz, consigne tenue entre deux pas ; physique à 500 Hz |

## Ce qu'on a mesuré

* **Espace de travail, pince verticale, colonne basse** : x ∈ [0.25, 0.35], z ∈ [0.78, 0.90]
  (à y = ±0.15). Au-delà, `wrist_pitch` arrive en butée (-1.61 rad). La pièce est à x = 0.34.
* **Ne pas suivre de trajectoire cartésienne depuis le tuck** : la pose est quasi singulière,
  l'IK `follow` décroche (6 sous-cibles convergées sur 44). Toujours passer par `go_ready()`.
* **Ne jamais envoyer à la main un résultat d'IK non convergé** : la DLS rend son meilleur
  effort, jusqu'à 4.5 rad en un pas. Les servos (kp = 300) l'exécutent, le bras percute et
  la physique diverge (NaN). `track_tcp` tient la consigne dans ce cas et borne le pas à
  3 rad/s.

## Ce qui reste hypothétique (repris de mpc_any)

* Le montage de la Dex1 sur le poignet (`ATTACH` / `ATTACH_RPY` dans `make_g1d_dex1.py`)
  n'a pas été vérifié sur le vrai robot.
* Les poses des caméras et de la table sont des valeurs plausibles, pas des mesures.
