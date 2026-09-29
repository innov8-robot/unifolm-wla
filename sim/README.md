# sim — G1-D + Dex1-1 dans MuJoCo (extrait de mpc_any)

Sim minimale pour tester UnifoLM-WLA : la scène, les servos, la gravité, les pinces, les
3 caméras RGB attendues par WLA et la FK/IK des deux bras. Caméras, buste, table, pose de
départ et repères base/effecteur sont calés sur les datasets G1 Dex1 d'entraînement (voir `docs/G1D_Constats.md`). Rien d'autre de mpc_any (collision, planif,
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
└── smoke.py               # 21 vérifications + les 3 vues en PNG dans smoke_out/
```

## Lancer

```bash
conda activate unitree_lerobot          # mujoco 3.x + pinocchio 3.x + imageio
MUJOCO_GL=egl python sim/smoke.py       # depuis la racine du dépôt
```

## Utiliser

```python
from g1d_sim import G1DSim

sim = G1DSim(control_hz=30, image_size=(640, 480))   # reset = tuck + settle, buste droit
sim.go_ready()                 # buste penché à 0.166 rad, bras aux angles de départ du G1

imgs = sim.render_all()        # {"head_left", "cam_wrist_left", "cam_wrist_right"} -> HxWx3 RGB
                               # ⚠ le serveur WLA attend du BGR : convertir avant envoi

# Ce que le modèle WLA attend, dans SES repères
T_ee = sim.ee_pose_wla("left") # 4x4 effecteur WLA dans la base WLA (bassin virtuel du G1)
lb = sim.lower_body_wla()      # (15) jambes G1 debout + taille [0, 0, tangage du buste]
sim.track_ee_wla("left", T)    # applique une pose effecteur WLA absolue (sortie du serveur)

# Bas niveau, repère monde MuJoCo
T = sim.tcp_pose("left")       # 4x4 TCP sim (≠ effecteur WLA : 4 cm plus loin, tourné de 90°)
sim.track_tcp("left", T_cible) # 1 pas cartésien sûr (IK follow, tenue si hors d'atteinte)
g = sim.gripper("left")        # fermeture mesurée ∈ [0, 1] (≠ unité Dex1 de WLA, voir docs)
sim.set_gripper("left", 1.0)   # 0 = ouverte, 1 = fermée
sim.step()                     # 1 pas de contrôle (17 pas de physique à 0,002 s)

snap = sim.save_state()        # restore_state(snap)
```

## Conventions

| | |
|---|---|
| Caméras | rôles WLA : `head_left` = `head_left_cam`, œil gauche rectifié de la stéréo de tête (fovy 75°) ; `cam_wrist_left/right` = `*_wrist_cam` (fovy 110°, non calé). 640×480, RGB seulement |
| Bras | 7 DoF, ordre `shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw` (= ordre des datasets G1, malgré les noms de leurs métadonnées) |
| Base WLA | bassin virtuel du G1 : torse = base · translation (-0.004, 0, 0.044) · tangage du buste |
| Effecteur WLA | `*_wrist_yaw_link` + (0.105, ±0.003, 0), orientation du poignet |
| TCP sim | `*_gripper_base_link` + (0, 0.105, 0) — ⚠ 16 mm du vrai centre des patins (`TCP_DOIGTS_M`) |
| Pince | consigne Joint1_1 de -0.018 (ouverte) à 0.0245 (fermée) ; Joint2_1 couplé par `<equality>` |
| Colonne | tenue à 0 (butée basse) |
| Buste | tangage (`Yaw_Joint`, axe y) droit au reset, penché à `TORSO_PITCH` = 0.166 rad par `go_ready`, compensé en gravité |
| Table | dessus à 0.87 m, ~0.07 m au-dessus de la base WLA comme chez le G1 (remontée de 0.13 m) |
| Contrôle | 30 Hz, consigne tenue entre deux pas ; physique à 500 Hz |

## Ce qu'on a mesuré

* **Ne pas suivre de trajectoire cartésienne depuis le tuck** : la pose est quasi singulière,
  l'IK `follow` décroche (6 sous-cibles convergées sur 44). Toujours passer par `go_ready()`.
* **Ne jamais envoyer à la main un résultat d'IK non convergé** : la DLS rend son meilleur
  effort, jusqu'à 4.5 rad en un pas. Les servos (kp = 300) l'exécutent, le bras percute et
  la physique diverge (NaN). `track_tcp` tient la consigne dans ce cas et borne le pas à
  3 rad/s.
* **Rampe directe tuck → départ** : les poignets tapent le bord de la table et le carton.
  `go_ready` passe par `VIA_Q_RIGHT`, trouvée par recherche de collisions.
* **La pièce est sur le chemin du doigt droit** quand le bras droit descend depuis la pose de
  départ : prévoir la scène en conséquence.

## Caméra de tête : d'où viennent les valeurs

* **Pose** : celle de la caméra `d435` du G1, identique dans l'URDF officiel et dans les
  datasets : xyz (0.0576, 0.0175, 0.4299) dans le `torso_link` du G1, tangage 0.8308 rad.
  La tête du G1-D est montée 1 cm plus bas sur `torso_link`, d'où z = 0.4199 dans la sim.
* **Œil gauche et champ** : décalage de 0.074 m le long de y et fovy 75°, **estimés** en
  projetant les poses effecteur enregistrées sur 7 images `head_stereo_left_rec` de
  `G1_Dex1_Stack_Block`. Erreur de reprojection de 12 à 18 px.

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
