# sim — G1-D + Dex1-1 dans MuJoCo (extrait de mpc_any)

Sim minimale pour tester UnifoLM-WLA : la scène, les servos, la gravité, les pinces, les
3 caméras RGB attendues par WLA et la FK/IK des deux bras. Caméras et posture du buste
sont calées sur les datasets G1 Dex1 d'entraînement (voir `docs/G1D_Constats.md`). Rien d'autre de mpc_any (collision, planif,
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
│   └── camera.py          # SimCamera (RGB seulement)
└── smoke.py               # 18 vérifications + les 3 vues en PNG dans smoke_out/
```

## Lancer

```bash
conda activate unitree_lerobot          # mujoco 3.x + pinocchio 3.x + imageio
MUJOCO_GL=egl python sim/smoke.py       # depuis la racine du dépôt
```

## Utiliser

```python
from g1d_sim import G1DSim

sim = G1DSim(control_hz=30, image_size=(640, 480))   # reset = tuck + settle
sim.go_ready()                                       # pose de travail, pinces vers le bas,
                                                     # buste incliné à 0.136 rad comme le G1

imgs = sim.render_all()        # {"head_left", "cam_wrist_left", "cam_wrist_right"} -> HxWx3 uint8
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
| Caméras | rôles WLA : `head_left` = `head_left_cam`, œil gauche rectifié de la stéréo de tête (fovy 75°) ; `cam_wrist_left/right` = `*_wrist_cam` (fovy 110°). 640×480, RGB seulement |
| Bras | 7 DoF, ordre `shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw` |
| TCP | `*_gripper_base_link` + (0, 0.105, 0) — ⚠ 16 mm du vrai centre des patins (`TCP_DOIGTS_M`) |
| Pince | consigne Joint1_1 de -0.018 (ouverte) à 0.0245 (fermée) ; Joint2_1 couplé par `<equality>` |
| Colonne | tenue à 0 (butée basse) |
| Buste | tangage (`Yaw_Joint`, axe y) droit au reset, incliné à `TORSO_PITCH` = 0.136 rad par `go_ready`, compensé en gravité |
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

## Caméra de tête : d'où viennent les valeurs

* **Pose** : celle de la caméra `d435` du G1 dans `torso_link`, lue dans les datasets
  (`observation.state.state_d435` et `state_torso`) : xyz (0.0576, 0.0175, 0.4299), tangage
  0.8308 rad, constante sur tout un épisode.
* **Œil gauche et champ** : décalage de 0.074 m le long de y et fovy 75°, **estimés** en
  projetant les poses effecteur enregistrées sur 7 images `head_stereo_left_rec` de
  `G1_Dex1_Stack_Block`. Erreur de reprojection de 12 à 18 px.
* **Buste** : le G1 manipule buste penché de 0.136 rad (médiane `state_torso`). Avec ce
  tangage, la caméra plonge de 55°, comme à l'entraînement.
* **Hypothèse** : le `torso_link` du G1-D est celui du G1. Le montage de la tête dans la
  scène (`head_link`, `logo_link`) est identique à celui de l'URDF G1.

## Ce qu'on a corrigé

* L'ancien point de passage de `go_ready` (« monter sur place » depuis le tuck) était
  toujours hors d'atteinte : l'IK échouait et seul le trajet direct s'exécutait.
* Le buste fléchissait de 0.03 rad sous son servo alors que la FK le supposait à 0.
  Il est désormais compensé en gravité, et la FK utilise les angles réels.
* MuJoCo remet l'état à zéro en silence sur une divergence. Le smoke test le détecte.

## Ce qui reste hypothétique (repris de mpc_any)

* Le montage de la Dex1 sur le poignet (`ATTACH` / `ATTACH_RPY` dans `make_g1d_dex1.py`)
  n'a pas été vérifié sur le vrai robot.
* Les poses des caméras et de la table sont des valeurs plausibles, pas des mesures.
