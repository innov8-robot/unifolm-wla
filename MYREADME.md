# UnifoLM-WLA sur Unitree G1-D — guide du fork innov8

Ce fork adapte **UnifoLM-WLA-1.0** d'Unitree au **G1-D** : robot sur base roulante, colonne élévatrice, deux pinces Dex1 et caméra stéréo de tête stock. Ce fichier est le point d'entrée. Il résume tout et renvoie aux documents détaillés.

- **Dépôt** : `git@github.com:innov8-robot/unifolm-wla.git`, fork **public** de `unitreerobotics/unifolm-wla`.
- **Branche de travail** : `g1d-port`. `main` suit le dépôt d'Unitree.
- **Dernière mise à jour** : 1er octobre 2026.

---

## Sommaire

1. [Où en est-on](#1-où-en-est-on) et [TODO](#todo)
2. [Arborescence](#2-arborescence)
3. [Environnements](#3-environnements)
4. [Simulation MuJoCo](#4-simulation-mujoco)
5. [Téléopération et enregistrement](#5-téléopération-et-enregistrement)
6. [Chaîne complète, de la démo au modèle](#6-chaîne-complète-de-la-démo-au-modèle)
7. [Contrat iso WLA, l'essentiel](#7-contrat-iso-wla-lessentiel)
8. [Pièges connus](#8-pièges-connus)
9. [Questions ouvertes](#9-questions-ouvertes)
10. [Git, données et sécurité](#10-git-données-et-sécurité)
11. [Index des documents](#11-index-des-documents)

---

## 1. Où en est-on

| Étape | État |
|---|---|
| Analyse du code WLA et des datasets d'entraînement | **Fait**, voir `docs/G1D_Constats.md` |
| Sim MuJoCo du G1-D calée sur l'entraînement | **Fait**, 22 vérifications vertes |
| Téléop intégrée au fork, sans Orbbec, env `g1d_teleop` | **Fait**, testée sans robot |
| Caméra stéréo stock remise sur le robot, config teleimager | **À faire**, sur le robot |
| Convertisseur d'enregistrements → format WLA | **Fait**, deux hypothèses à valider |
| Client sim ↔ serveur WLA, test zero-shot | **Fait** en sim, pas de prise réussie |
| Poids UnifoLM-WLA-1.0-Base téléchargés | **Fait** |
| Correction du gel du robot-state projector pour le fine-tuning | **Fait et vérifié** |
| Validation de toute la chaîne en sim, tâche cube | **Fait** : 23/30 après fine-tuning, contre 0/20 en zero-shot |
| Tâche Novares en sim (prise peinte) | **Fait** : 27/30 avec 50 démos, 22/30 avec 25, 27/30 avec 10 |
| Boucle RECAP / Delta-0 | Brique implémentée, audit corrigé. 1re itération en sim : **pas de gain** (14/30 contre 18/30 pour la référence), modèle de valeur trop faible |
| Empilement Novares en sim (pièces à plat, prise par le côté) | Expert 30 % ; modèle **0/30** (25, 100 démos, 12 000 pas, 1 itération DAgger) : l'expert est le goulot |
| Mode politique avec correction en delta dans la téléop | Fait, testé hors robot, **non validé sur le robot** |
| Rotation du buste G1-D (sim, conversion, téléop au joystick droit) | Fait, testé hors robot ; **indice moteur à vérifier** |
| Fine-tuning sur de vraies démos | Recette prête et validée en sim |
| Push de `g1d-port` sur GitHub | **Fait**, à refaire après chaque étape |

### Outils

- **Dataset Studio** (`dataset_studio/`, PySide6, thème du cockpit de mpc_any) : voir et modifier les enregistrements de la téléop. Liste des épisodes (durée, issue, pinces utilisées, base / colonne, images manquantes), lecture des 4 caméras synchronisées, courbes (articulations état / consigne, pinces, buste, colonne, base), suppression vers une corbeille avec renumérotation sans trou, restauration, rognage début / fin, consigne et issue modifiables, conversion au format WLA en un clic.
  ```bash
  conda activate g1d_teleop
  ~/Documents/project/manip/unifolm-wla/studio.sh        # depuis n'importe quel dossier (défaut : teleop/utils/data)
  ```
  Raccourcis : Espace lecture · ←/→ pas à pas · Maj+←/→ ±1 s · ↑/↓ épisode · I / O rognage · Suppr supprimer.
  Mode d'emploi complet : [dataset_studio/README.md](dataset_studio/README.md).

### TODO

Par ordre de priorité. Cocher au fur et à mesure.

- [x] **Pousser `g1d-port` sur GitHub** : fait le 29 septembre 2026. Pousser à nouveau après chaque étape.
- [ ] **Remettre la stéréo stock sur le robot** : la config teleimager du robot déclare encore l'Orbbec. Sans ça, tout nouvel enregistrement sera au mauvais format. *Procédure et bloc de config prêts : `teleoperation/REAMDEG1D.md`, section « Remettre la caméra stéréo stock ». Reste à l'appliquer sur le robot.*
- [x] **Télécharger les poids et installer l'env du modèle** : fait. Poids dans `playground/Pretrained_models/UnifoLM-WLA-1.0-Base/`, env `.venv`, torch 2.8 CUDA 12.8 validé sur la RTX 5090. Sans flash-attention : le code bascule sur l'attention standard de PyTorch.
- [x] **Écrire le client de test** : `sim/wla_client.py`, boucle fermée sim ↔ serveur WLA qui fonctionne, environ 0,4 s par inférence. Premier zero-shot : vue rectifiée, le modèle reste quasi immobile. Vue brute, il approche la main droite et ferme la pince, mais 12 cm trop haut. Un essai par vue, donc non concluant. *La correspondance de pince sim ↔ Dex1 y est supposée linéaire.*
- [x] **Écrire le convertisseur** : `g1d_wla/convert_teleop.py`, testé sur 2 épisodes de `mon_test`, lu sans erreur par le dataloader WLA avec `configs/g1d.yaml`. *Deux hypothèses à valider sur le robot : l'indice du tangage du buste et la hauteur de bassin équivalente.*
- [ ] **Mesurer sur le robot** :
  - [ ] la correspondance entre la pince et l'unité Dex1 de WLA ;
  - [ ] le contenu des 35 moteurs enregistrés, pour savoir s'ils contiennent le tangage du buste ;
  - [ ] la hauteur de colonne, à ajouter à l'enregistrement ;
  - [ ] si possible, la calibration de la stéréo de tête.
- [ ] **Installer le poste de démo** : une table à environ 0,87 m et le buste penché d'environ 0.166 rad.
- [x] **Valider la chaîne complète en sim** : 150 démos expertes de la tâche cube, fine-tuning de 3 000 pas, puis **23 prises sur 30** positions jamais vues. Le zero-shot faisait 0 sur 20. Voir « Validation en sim » plus bas.
- [x] **Améliorer la vitesse et réduire le nombre de démos, en sim** : chunks entiers, 28/30 en 151 pas au lieu de 222. Real-time chunking ajouté, avec raccord doux : 25/30 en 151 pas en replanifiant tous les 10 pas. **10 démos suffisent** pour 25/30, et 25 démos donnent 30/30.
- [x] **Tâche Novares en sim** : prise peinte de mpc_any. 50 démos : **27/30**, à la vitesse de l'expert. 25 démos : 22/30. 10 démos : 27/30.
- [ ] **Boucle RECAP / Delta-0 (amélioration par essais et corrections)** : brique implémentée, auditée et corrigée. 1re itération en sim dans la zone « à gauche » (`sim/experiments/queue_recap.sh`) : référence 18/30, RECAP 14/30, témoin sans avantage 13/30. Pas d'effet du conditionnement, et ajouter les rollouts dégrade. Cause probable : modèle de valeur qui apprend par cœur (erreur 128 pas hors entraînement). Pistes : plus de rollouts, valeur plus simple (succès/échec de l'épisode), fine-tuning depuis n10 plutôt que depuis Base. Voir section 15 des constats.
- [ ] **Empilement Novares en sim** : scène corrigée (pièces à plat, prise par le côté), expert à 30 %. Modèle 0/30 dans tous les essais : 25 démos, 100 démos, 25 démos × 12 000 pas (boucle ouverte 8,3 mm), 1 itération DAgger. **Prochaine étape : fiabiliser l'expert** (pièce qui glisse, reprise depuis un état quelconque), puis évaluer sur des placements que l'expert réussit. Voir section 14 des constats.
- [ ] **Vérifier sur le robot l'indice du moteur de rotation du buste** (hypothèse 12) avant d'utiliser `--torso-yaw-index`. Voir section 16 des constats.
- [ ] **Essayer le mode politique avec corrections sur le robot** : `--policy-uri` dans la téléop, procédure dans `teleoperation/REAMDEG1D.md`. Corrigé après audit de sécurité et testé hors robot. Premier essai : vitesse bridée et arrêt d'urgence à portée.
- [ ] **Enregistrer, puis fine-tuner** : la recette est prête et testée sur `mon_test`. Elle tourne à environ 1,7 s par pas sur la RTX 5090. Le correctif du projecteur gelé est **vérifié** : 1 397 M paramètres entraînables, soit la tête DiT plus les 6,87 M du projecteur. Reste à enregistrer de vraies démos iso, voir les points précédents.

---

## 2. Arborescence

```
unifolm-wla/
├── MYREADME.md                ← ce fichier
├── README.md                  # README d'Unitree, inchangé
├── docs/
│   ├── Ma_Reflection.md       # briefing initial du portage, corrigé par G1D_Constats.md
│   ├── G1D_Constats.md        # TOUT ce qu'on a vérifié, avec niveaux de confiance
│   ├── robot_action_state_processing_en.md   # spec officielle des espaces état/action
│   └── train_action_expert_en.md             # doc officielle entraînement/fine-tuning
├── unifolm_wla/               # code du modèle, d'Unitree
├── model_server/              # serveur d'inférence WLA, websocket + msgpack
├── examples/                  # scripts de fine-tuning et d'évaluation, d'Unitree
├── g1d_wla/                   # conventions iso WLA (frames.py), FK numpy, convertisseur, RECAP (recap.py)
├── sim/                       # notre sim MuJoCo du G1-D + Dex1
│   ├── README.md
│   ├── g1d_sim/               # G1DSim : physique, FK/IK, caméras, repères WLA
│   ├── assets/                # scènes, URDF, meshes (pièce Novares NON versionnée)
│   ├── smoke.py               # 22 vérifications de bout en bout
│   ├── cube_task.py, novares_task.py, stack_task.py, sim_tasks.py   # tâches et experts
│   ├── record_sim_demos.py, recap_rollouts.py, wla_client.py        # démos, rollouts, évaluation
│   ├── dagger_corrections.py, open_loop_check.py                    # corrections DAgger, contrôle en boucle ouverte
│   └── experiments/           # files de travaux (queue_novares, queue_recap, queue_stack, queue_dagger)
└── teleoperation/             # notre téléop, xr_teleoperate personnalisé
    ├── setup_env.sh           # crée l'env conda g1d_teleop
    ├── REAMDEG1D.md           # démarrage téléop pas à pas
    ├── CAMERA_VIDEO.md        # flux caméra, réseau, dépannage
    ├── README.md              # mémo de commandes
    └── Tele_OP/
        ├── xr_teleoperate/    # + teleimager, televuer, dex-retargeting
        └── unitree_sdk2_python/
```

---

## 3. Environnements

| Env | Pour quoi | Création |
|---|---|---|
| `g1d_teleop`, conda | téléopération et enregistrement | `bash teleoperation/setup_env.sh` |
| `unitree_lerobot`, conda | sim MuJoCo : mujoco 3.x, pinocchio 3.x, imageio | déjà présent sur la machine |
| env `uv` du projet, `.venv` | modèle WLA : entraînement, serveur, convertisseur | voir ci-dessous |

### Téléop

```bash
bash teleoperation/setup_env.sh                          # une seule fois
WITH_DEX_RETARGETING=1 bash teleoperation/setup_env.sh   # seulement pour les mains Dex3, Inspire ou Brainco
conda activate g1d_teleop
```

- Les versions sont reprises de l'ancien env `tv`, qui a téléopéré le G1-D : pinocchio 3.1.0 avec casadi, numpy 1.26.4, vuer 0.0.60 et cyclonedds 0.10.2.
- Le SDK Unitree, teleimager et televuer sont installés en éditable depuis `teleoperation/`.
- Les chemins des certificats du casque sont des variables de l'env conda. Plus besoin d'export dans `.bashrc`.
- Les certificats `cert.pem` et `key.pem` sont locaux et non versionnés. Pour les régénérer, voir la section 1.1 du README de xr_teleoperate.

### Modèle WLA

```bash
# ⚠ hors de conda : sinon uv prend le Python 3.13 et le compilateur de conda, et evdev ne compile pas
env -i HOME=$HOME PATH=/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin LANG=C.UTF-8 uv sync --python /usr/bin/python3.12
hf download unitreerobotics/UnifoLM-WLA-1.0-Base --local-dir playground/Pretrained_models/UnifoLM-WLA-1.0-Base
```

flash-attention n'est pas installé : le code bascule tout seul sur l'attention standard de PyTorch.

Voir `docs/train_action_expert_en.md` pour l'installation complète.

---

## 4. Simulation MuJoCo

```bash
MUJOCO_GL=egl ~/miniconda3/envs/unitree_lerobot/bin/python sim/smoke.py   # 22/22 attendu
```

Sur un clone neuf, sans les maillages Novares non versionnés, `G1DSim()` et le smoke test basculent seuls sur la scène cube.

```python
import sys; sys.path.insert(0, "sim")
from g1d_sim import G1DSim

sim = G1DSim(head_view="rec")   # "rec" = œil gauche rectifié, "raw" = œil gauche brut
sim.go_ready()                  # buste penché à 0.166 rad, bras aux angles de départ du G1

imgs = sim.render_all()         # head_left, cam_wrist_left, cam_wrist_right : RGB 640×480
T_ee = sim.ee_pose_wla("left")  # effecteur WLA dans la base WLA : ce que le modèle attend
lb   = sim.lower_body_wla()     # jambes G1 debout + taille G1 équivalente au buste (tangage, lacet)
sim.track_ee_wla("left", T)     # applique une pose effecteur WLA absolue
sim.set_gripper("left", 1.0)    # 0 = ouverte, 1 = fermée
sim.step()                      # 1 pas à 30 Hz
```

Ce qui est calé sur les datasets d'entraînement G1 :

- **Caméra de tête** : pose de la d435 du G1, champ estimé par reprojection. Deux vues possibles, rectifiée ou brute.
- **Buste** : tangage à la médiane d'entraînement.
- **Table** : hauteur par rapport au bassin comme chez le G1.
- **Pose de départ** : angles médians de départ du G1.
- **Repères** : base WLA et effecteur WLA exacts.

Détails et limites dans `sim/README.md` et dans la section 7 bis de `docs/G1D_Constats.md`.

---

## 5. Téléopération et enregistrement

Procédure complète dans `teleoperation/REAMDEG1D.md`. En résumé :

**1. Robot, PC2**, par SSH sur `unitree@192.168.123.164`. Le mot de passe n'est pas versionné.

```bash
sudo systemctl restart teleimager.service        # caméras : tête + 2 poignets
sudo systemctl restart dex1_1_gripper.service    # pinces : les 2 moteurs doivent répondre
```

**2. PC**

```bash
conda activate g1d_teleop
cd teleoperation/Tele_OP/xr_teleoperate/teleop
python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --img-server-ip=192.168.123.164 \
    --input-mode=controller --arm=G1_29 --ee=dex1 --record --task-name=<nom> --task-goal="<instruction>"
```

**3. Casque** : ouvrir `https://<ip-wifi-du-PC>:8012` pour accepter le certificat, puis `https://vuer.ai/?ws=wss://<ip-wifi-du-PC>:8012&grid=False`. Dans le terminal, `r` démarre la téléop, `s` démarre ou arrête un épisode, `q` quitte.

**Colonne** : `python height_tool.py read | home | goto --rel 0.10`, depuis `teleop/`.

**Options utiles** :
- `--right-only` : un seul bras ;
- `--no-wrist-pip` : sans vignettes poignets dans le casque ;
- `--headless` : sans affichage.

### Ce qui est enregistré

Dans `teleop/utils/data/<task-name>/episode_XXXX/`, à 30 Hz :

- **Images** : tête binoculaire coupée en deux yeux **bruts** de 640×480, plus les deux poignets en 640×480.
- **État** : angles mesurés des bras, pinces et 35 moteurs du corps.
- **Action** : angles commandés des bras, sortie de l'IK, et pinces.
- **Absent** : poses effecteur, pose du buste, hauteur de colonne. Elles se recalculent à la conversion, sauf la hauteur de colonne, à ajouter à l'enregistrement.

### Configuration caméra à respecter

La caméra de tête doit être la **stéréo stock**, en binoculaire 480×1280, avec le numéro de série `01.00.00`. L'Orbbec a été retirée du code. La config côté robot, `cam_config_server.yaml` sur PC2, doit être remise en stéréo.

---

## 6. Chaîne complète, de la démo au modèle

```
Téléop xr_teleoperate ──► JSON + JPEG ──► convertisseur ──► LeRobot v3 format WLA
   (teleoperation/)                  (g1d_wla/convert_teleop.py) (clés des datasets G1 Dex1)
                                                                   │
            ┌──────────────────────────────────────────────────────┘
            ▼
Fine-tuning WLA, VLM gelé ──► checkpoint ──► model_server ◄──► sim/wla_client.py ou téléop --policy-uri
 (examples/unifolm_wla/train_files/run_finetune_g1d.sh)
```

### Convertisseur

```bash
.venv/bin/python -m g1d_wla.convert_teleop \
    --raw-dir teleoperation/Tele_OP/xr_teleoperate/teleop/utils/data/mon_test \
    --out-dir playground/Datasets/g1d/mon_test --repo-id innov8/g1d_mon_test
```

- Le dataset produit se range sous `playground/Datasets/g1d/<tâche>/`, que lit `unifolm_wla/dataloader/multi_source_dataset/configs/g1d.yaml`.
- La source y est nommée `UnifoLM_G1_Dex1`, pour réutiliser les normaliseurs du modèle Base.
- Les constantes de repère sont dans `g1d_wla/frames.py`, source unique partagée avec la sim.
- **Pince** : la téléop enregistre déjà l'angle moteur Dex1, de 0 fermée à 5,4 ouverte. C'est l'unité de l'entraînement, donc aucune conversion n'est faite.

Ce qu'il produit, détail en section 10 de `docs/G1D_Constats.md` :

- **Effecteur, état** : FK des angles **mesurés** dans la base WLA, avec l'effecteur WLA, en euler `xyz`.
- **Effecteur, action** : FK des angles **commandés**, même repère.
- **Jambes** : posture debout du G1, la constante `G1_STANDING_LEGS` de la sim.
- **Taille** : taille G1 de même orientation de torse que le buste G1-D, `waist_from_torso(tangage, lacet)` ; `[0, 0, tangage]` buste non tourné.
- **Commande de base** : vitesses de la base enregistrées, et hauteur de bassin **constante**, `G1_BASE_HEIGHT` = 0,732 m ou `--base-height`. Elle ne dépend pas encore de la colonne.
- **Caméras** : `head_stereo_left` pour l'œil gauche brut, `head_stereo_right`, `wrist_left` et `wrist_right`. Pour la sim, qui n'a qu'un œil, seulement `head_stereo_left` et les poignets. La disposition est déduite de l'en-tête. Pour la téléop `--right-only`, passer `--layout right-only`.
- **Pince** : unité Dex1, copiée telle quelle.
- **Normalisation** : statistiques précollectées du dépôt, clé `UnifoLM_G1_Dex1`. **Ne pas les recalculer.**

### Test zero-shot en sim

```bash
# 1. serveur, env uv
.venv/bin/python -m model_server.action_server_wbc_msgpack_unitree \
    --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
    --unnorm_key UnifoLM_G1_Dex1 --port 8600
# 2. client, env de la sim ; sorties dans sim/wla_out/, hors git
MUJOCO_GL=egl ~/miniconda3/envs/unitree_lerobot/bin/python sim/wla_client.py \
    --instruction "pick up the black part and put it in the box" --episodes 1 --max-steps 300 --head-view raw
```

- Le serveur occupe environ 11,6 Go de mémoire graphique.
- `--debug_save_dir` écrit à chaque requête les images reçues et le chunk prédit.

### Validation en sim : démos, fine-tuning, évaluation

Chaîne complète sans robot, sur une tâche simple : saisir un cube rouge de 4 cm et le soulever. Elle passe par les **mêmes** convertisseur, config et recette que les vraies démos.

```bash
SIMPY=~/miniconda3/envs/unitree_lerobot/bin/python
# 1. démos expertes au format xr_teleoperate (~1,5 s/épisode)
MUJOCO_GL=egl $SIMPY sim/record_sim_demos.py --n 150 --out playground/sim_raw/sim_cube
# 2. conversion au format WLA (~15 min)
.venv/bin/python -m g1d_wla.convert_teleop --raw-dir playground/sim_raw/sim_cube \
    --out-dir playground/Datasets/g1d_sim/sim_cube --repo-id innov8/g1d_sim_cube
# 3. fine-tuning, ~1 h 30
run_id=g1d_sim_cube_v1 bash examples/unifolm_wla/train_files/run_finetune_g1d.sh \
    --datasets.vla_data.data_config_path unifolm_wla/dataloader/multi_source_dataset/configs/g1d_sim.yaml \
    --datasets.vla_data.per_device_batch_size 2 --trainer.gradient_accumulation_steps 1 \
    --trainer.max_train_steps 3000 --trainer.num_warmup_steps 100 --trainer.save_interval 1500
# 4. évaluation : serveur sur le checkpoint, puis client
.venv/bin/python -m model_server.action_server_wbc_msgpack_unitree --unnorm_key UnifoLM_G1_Dex1 \
    --ckpt_path playground/Checkpoints/g1d_sim_cube_v1/checkpoints/steps_3000_model.safetensors
MUJOCO_GL=egl $SIMPY sim/wla_client.py --scene cube --instruction "pick up the red cube" \
    --head-view raw --episodes 30 --max-steps 300
```

| Modèle | Pas par épisode | Réussites |
|---|---|---|
| Base, zero-shot | 200 | 0 / 20 |
| Fine-tuné, 3 000 pas | 200 | 4 / 30 |
| Fine-tuné, 3 000 pas | 300 | **23 / 30** |

- **Mêmes positions** : les 30 positions de cube sont identiques d'une ligne à l'autre, et aucune n'a été vue à l'entraînement.
- **Lenteur** : elle venait de l'exécution de 20 pas sur 30 par chunk, pas du modèle. Voir « Vitesse et nombre de démos » juste en dessous.
- **Perte** : environ 0,027 au pas 100, 0,004 au pas 1 000, et entre 0,002 et 0,01 à la fin.
- **Échecs** : une partie ont lieu dans le fond de la zone, en y très négatif.

### Vitesse et nombre de démos

Détail et analyse : section 13 de `docs/G1D_Constats.md`. L'expert scripté réussit en 127 pas.

| Démos | Réglage d'exécution | Réussites | Pas médian jusqu'à la réussite |
|---|---|---|---|
| 150 | 10 pas par chunk | 8 / 30 | 266 |
| 150 | 20 pas par chunk | 24 / 30 | 222 |
| 150 | chunk entier, 30 pas | 28 / 30 | 151 |
| 150 | 10 pas + préfixe de 20 | 23 / 30 | **148** |
| **25** | chunk entier | **30 / 30** | 173 |
| 25 | 10 pas + préfixe de 20 | 20 / 30 | 146 |
| **10** | chunk entier | **25 / 30** | 157 |
| 10 | 10 pas + préfixe de 20 | 18 / 30 | 150 |
| 10 | 10 pas + préfixe de 20 + raccord doux de 5 | **25 / 30** | **151** |
| 10 | 15 pas + préfixe de 10 + raccord doux de 5 | 13 / 30 | 213 |
| 10 | 25 pas + préfixe de 5 | 14 / 30 | 176 |

- **Vitesse** : exécuter les chunks en entier est le plus simple et le plus fiable. Replanifier souvent ralentit le robot, sauf avec le **préfixe de real-time chunking**, que nous avons ajouté au serveur et à la tête d'action.
- **Raccord doux** : avec un préfixe de 20 pas et un raccord doux de 5 pas, la replanification tous les 10 pas devient aussi fiable que les chunks entiers, tout en étant trois fois plus réactive.
- **Préfixe court** : un préfixe court, de 5 ou 10 pas, dégrade nettement. Le préfixe doit couvrir l'essentiel du chunk.
- **Nombre de démos** : **10 démos suffisent** pour 25/30 sur cette tâche en sim. 25 démos donnent 30/30, et 150 démos ne font pas mieux.
- **Vitesse plafond** : on reste à environ 150 pas contre 127 pour l'expert, soit environ 18 % plus lent.
- **Pour le vrai robot** : commencer par des chunks entiers et viser 25 à 50 démos pour une première tâche.

```bash
# chunks entiers (fiable)
MUJOCO_GL=egl $SIMPY sim/wla_client.py --scene cube --instruction "pick up the red cube" \
    --head-view raw --episodes 30 --max-steps 300 --exec-steps 30 --stop-on-success
# replanification tous les 10 pas avec préfixe et raccord doux (aussi fiable, 3 fois plus réactif)
... --exec-steps 10 --rtc-prefix 20 --rtc-soft 5
```

### Tâche Novares en sim : prise peinte

Tâche plus dure que le cube : saisir la pièce Novares par la **prise peinte** dans mpc_any, puis la soulever. Code : `sim/novares_task.py`, tâches enregistrées dans `sim/sim_tasks.py`.

- **Prise** : lue dans les zones peintes de mpc_any, avec la même logique que `mpc_any/.../perception/zones_prise.py`. L'axe des mors passe entre les deux zones. On échantillonne 24 approches autour de cet axe, on rejette celles où la paume traverse la pièce, et on prend la plus verticale atteignable. La prise fait 52 mm de large, en approche verticale.
- **Variance** : la pièce garde sa pose stable, avec ±3 cm en x et en y et ±20° de lacet à chaque épisode.
- **Expert** : il monte la main, fait un transfert articulaire au-dessus de la pièce, puis descend. Il réussit 46 prises sur 50. L'enregistreur ne garde que les réussites propres : 50 démos gardées sur 54 essais.
- **Fichiers non versionnés**, tous dans `sim/assets/meshes/` :
  - `_Novares_Piece1_centered.stl`, le maillage visuel ;
  - les 90 `__Novares_Piece1_centered_partNN.stl`, sa décomposition convexe pour les collisions ;
  - `_Novares_Piece1_centered.zones.json`, copié de `mpc_any/configs/projects/usine/novares.zones.json`, même STL ;
  - `_Novares_Piece1_centered.stack.json`, le décalage d'emboîtement pour l'empilement.

```bash
MUJOCO_GL=egl $SIMPY sim/record_sim_demos.py --task novares --n 50 --out playground/sim_raw/sim_novares
.venv/bin/python -m g1d_wla.convert_teleop --raw-dir playground/sim_raw/sim_novares \
    --out-dir playground/Datasets/g1d_sim_novares/sim_novares --repo-id innov8/g1d_sim_novares
run_id=g1d_sim_novares_v1 bash examples/unifolm_wla/train_files/run_finetune_g1d.sh \
    --datasets.vla_data.data_config_path unifolm_wla/dataloader/multi_source_dataset/configs/g1d_sim_novares.yaml \
    --datasets.vla_data.per_device_batch_size 2 --trainer.gradient_accumulation_steps 1 --trainer.max_train_steps 3000
MUJOCO_GL=egl $SIMPY sim/wla_client.py --scene novares --instruction "pick up the black part" \
    --head-view raw --episodes 30 --max-steps 300 --exec-steps 30 --stop-on-success
```

- ⚠ **Conversion gourmande en mémoire vive**, à cause de l'encodage vidéo AV1. Ne pas la lancer pendant un entraînement, qui garde son optimiseur en mémoire vive.
**Résultats**, sur 30 placements de la pièce jamais vus. L'expert réussit en **151 pas** médians.

| Démos | Réglage d'exécution | Réussites | Pas médian jusqu'à la réussite |
|---|---|---|---|
| 50 | chunk entier | **27 / 30** | **149** |
| 50 | 10 pas + préfixe de 20 | 25 / 30 | 154 |
| 25 | chunk entier | 22 / 30 | 146 |
| 25 | 10 pas + préfixe de 20 | 21 / 30 | 151 |
| 10 | chunk entier | **27 / 30** | 161 |
| 10 | 10 pas + préfixe de 20 | 27 / 30 | 160 |

- **Tâche plus dure** : elle s'apprend aussi bien que le cube. Avec 50 démos, le modèle réussit 9 fois sur 10, **à la vitesse de l'expert**.
- **Nombre de démos** : 10 démos font aussi bien que 50, à 27/30. Le creux à 25 démos, 22/30, est probablement du bruit : un seul entraînement par configuration, et 30 essais.
- **Hors distribution** : le modèle à 10 démos généralise mal dans certaines zones décalées. Sur 15 essais : 30/30 à 4–7 cm plus loin, mais **7/15 à 4–7 cm à gauche**, 11/15 pièce tournée de 26 à 46°, et 14/15 à 5–8 cm à droite.
- **File automatique** : `sim/experiments/queue_novares.sh`. Elle enchaîne entraînement et évaluation pour 50, 25 puis 10 démos. Journal : `playground/queue_logs/queue.log`.

### Fine-tuning

```bash
# 1. convertir les démos sous playground/Datasets/g1d/<tâche>/ (voir plus haut)
# 2. lancer, run_id au choix ; toute surcharge --clé valeur est transmise
run_id=g1d_pick_v1 bash examples/unifolm_wla/train_files/run_finetune_g1d.sh
# essai court : ... --trainer.max_train_steps 5 --trainer.save_interval 100000
```

Ce que la recette G1-D change par rapport à la recette officielle :

- **Config** : `unifolm_wla/config/training/g1d_finetune_frozen_vlm.yaml`.
- **Gel du VLM par sous-modules** : le robot-state projector s'entraîne. Vérifié sur un vrai lancement : 1 397 M paramètres entraînables, soit la tête DiT plus les 6,87 M du projecteur.
- **Optimiseur déporté en mémoire vive**, par `deepspeed_zero2_offload.yaml` : sans ça, ni cette recette ni l'officielle ne tiennent sur les 23,4 Go de la RTX 5090 Laptop.
- **Données** : `configs/g1d.yaml`. Les normaliseurs sauvegardés avec le checkpoint sont identiques au bit près à ceux du modèle Base.
- **Compilation de DeepSpeed** : le script lui présente les bibliothèques CUDA 12 de l'env, dans `playground/cuda12_shim`, car le CUDA système est une version 13. Il faut aussi ninja, déjà présent dans l'env.
- **Coût** : environ 1,7 s par pas, lot de 1, et environ 12 Go de disque par checkpoint sauvegardé.

## 7. Contrat iso WLA, l'essentiel

Référence complète : section 9 de `docs/G1D_Constats.md`.

| Sujet | Ce que le modèle attend |
|---|---|
| Images | 3 vues, dans l'ordre `head_left`, `cam_wrist_left`, `cam_wrist_right`. Œil **gauche** seulement. 640×480 à 30 fps, vues par le modèle en 320×448 |
| Couleurs | le serveur attend du **BGR** |
| Vue de tête | rectifiée pour la config Dex1 du dépôt, mais **75 %** des frames publiques n'ont que la brute. Nos données utilisent l'œil gauche **brut**, `head_stereo_left` |
| Repère base | **bassin** du G1. Sur le G1-D, bassin virtuel sous le buste : translation (-0.004, 0, 0.044) puis tangage du buste |
| Effecteur | `wrist_yaw_link` + 0.105 m le long de x, orientation du poignet. Vérifié exact par FK |
| Actions bras | relatives à la pose **mesurée** au début du chunk, 30 pas à 30 Hz |
| Jambes, taille | slots **valides** : envoyer la posture debout du G1 et le tangage du buste |
| Buste | penché d'environ 0.166 rad, entre 0.13 et 0.18 selon la tâche |
| Table | environ 0,07 m au-dessus du bassin. Sur le G1-D réel, colonne en bas : table à environ **0,87 m** |
| Prompt | construit par le serveur, avec la ligne `Control Mode: Arms: EE, LOW BODY: JOINT` |

---

## 8. Pièges connus

- **Projecteur gelé** : la recette officielle gèle le robot-state projector, ainsi que la version LoRA. Le commentaire du YAML dit le contraire.
- **TCP de la sim** : c'est `tcp_pose`. Il est 4 cm plus loin que l'effecteur WLA et tourné de 90°. Ne **jamais** l'envoyer au modèle : utiliser `ee_pose_wla`.
- **Noms des joints** : les métadonnées des datasets G1 sont fausses pour l'ordre des joints du poignet. L'ordre réel est l'ordre standard du G1.
- **Chemin de la config de données** : dans `unitree.yaml`, l'entrée WBT pointe vers le dossier Dex1.
- **Taille d'image par défaut** : le serveur est en 320×448. C'est bien ce que voit le modèle, mais le rééchantillonnage diffère un peu de l'entraînement.
- **Divergence silencieuse** : MuJoCo remet la sim à zéro sans prévenir. Le smoke test le détecte.
- **Mains Brainco, Inspire ou Dex3** : il faut l'env avec `WITH_DEX_RETARGETING=1`.
- **Cache de l'IK** : il n'est pas versionné. Le premier lancement sur un clone neuf est plus lent.
- **Sans flash-attention** : le VLM basculait déjà sur l'attention de PyTorch, mais pas la tête d'action DiT, qui plantait. Elle est corrigée dans `mmdit.py` et bascule maintenant aussi. Rien ne change si flash-attention est installé.

---

## 9. Questions ouvertes

| Question | Comment trancher |
|---|---|
| Quelle vue de tête, rectifiée ou brute, le modèle Base a le plus vue ? | Tester les deux en zero-shot |
| Intrinsèques et baseline de notre stéréo de tête | Calibrer sur le robot |
| Intrinsèques des caméras de poignet | Reprojection sur les vues de poignet |
| Correspondance pince sim ou robot ↔ unité Dex1 WLA | Mesurer les valeurs ouverte et fermée des deux côtés |
| Hauteur exacte de la table dans les datasets G1 | Plan de table par la caméra calibrée |
| Le G1-D peut-il manipuler buste penché d'environ 0.166 rad ? | Tester sur le robot |
| Lequel des moteurs de `body.qpos` est le tangage du buste du G1-D ? Dans l'énumération G1_29 de la téléop, 12 = lacet, 13 = roulis et 14 = tangage de la taille. Dans `mon_test`, 12 et 13 valent environ 0,09 rad, et 14 vaut 0. Toute la chaîne utilise 13 par cohérence | Incliner le buste sur le robot et regarder lequel bouge |

---

## 10. Git, données et sécurité

- **Enregistrements** : `teleoperation/.../teleop/utils/data/`, **hors git**, exclus par `data/`.
  - `mon_test` : 51 épisodes, 1,9 Go, au bon format stéréo.
  - Les enregistrements Orbbec ont été **supprimés** le 29 septembre 2026, soit 9,8 Go. Ils n'étaient pas au format WLA.
- **Partage des données** : un dataset Hugging Face **privé**, une fois converties.
- **Ancien dépôt de téléop** : `innov8-robot/teleoperation`, laissé tel quel. Son historique est sauvegardé dans `../_backup_teleoperation_git_20260929/`. Il contient le mot de passe SSH du robot.
- **Hors git aussi** : les meshes de la pièce **Novares**, possiblement confidentiels, les certificats du casque, les réglages Claude locaux et la référence de hauteur de colonne.
- **Fork public** : relire avant de pousser tout ce qui touche au client ou au réseau interne. Les docs citent des IP du réseau robot et des noms de projets internes.
- **Docs en Markdown** : le `.gitignore` d'Unitree ignore les `.md` des sous-dossiers. Nos docs sont réintégrées par des exceptions explicites. Pour un nouveau `.md` dans un sous-dossier, ajouter une exception.

---

## 11. Index des documents

| Document | Contenu |
|---|---|
| `docs/G1D_Constats.md` | Constats vérifiés, contrat iso complet, données d'entraînement, enregistrement |
| `docs/Ma_Reflection.md` | Briefing initial. Plusieurs points y sont corrigés par les constats |
| `sim/README.md` | Utilisation, conventions et calibrations de la sim |
| `teleoperation/REAMDEG1D.md` | Démarrage de la téléop dans l'ordre |
| `teleoperation/CAMERA_VIDEO.md` | Flux caméra, réseau, ports, dépannage |
| `teleoperation/README.md` | Mémo de commandes de téléop |
| `docs/robot_action_state_processing_en.md` | Spec officielle des espaces état et action |
| `docs/train_action_expert_en.md` | Doc officielle d'installation, d'entraînement et de serveur |
| `teleoperation/Tele_OP/xr_teleoperate/README.md` | Doc officielle de xr_teleoperate |


python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --img-server-ip=192.168.123.164 --input-mode=controller --arm=G1_29 --ee=dex1 --torso-pitch 0.166  --frequency 60 --ik-smooth light --timing --torso-yaw-index 12 --torso-yaw-max 1.0 --torso-yaw-rate 0.5 --base --column --record --task-name=napkin --task-goal="fold a green napkin"



Voici les deux commandes, à copier telles quelles, chacune sur une seule ligne.

**Terminal 1 : le serveur du modèle**
```bash
cd ~/Documents/project/manip/unifolm-wla && env -i HOME=$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 .venv/bin/python -m model_server.action_server_wbc_msgpack_unitree --ckpt_path playground/Checkpoints/g1d_novares_box/final_model/model.safetensors --unnorm_key UnifoLM_G1_Dex1 --port 8600
```
Attendez **« server listening on »**, environ 1 minute pour charger le modèle. Laissez ce terminal ouvert.

**Terminal 2 : la téléop en mode politique**
```bash
conda activate g1d_teleop && cd ~/Documents/project/manip/unifolm-wla/teleoperation/Tele_OP/xr_teleoperate/teleop && adb reverse tcp:8012 tcp:8012 && python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --img-server-ip=192.168.123.164 --input-mode=controller --arm=G1_29 --ee=dex1 --torso-pitch 0.166 --policy-uri ws://127.0.0.1:8600 --policy-instruction "pick up the black object and put it inside a box" --policy-max-speed 1 --record --task-name=novares_box_policy
```

**Ensuite**
1. Dans le casque, ouvrez `https://vuer.ai/?ws=wss://localhost:8012&grid=False`, puis Virtual Reality.
2. Tapez **`r`** dans le terminal 2.
3. **A** droit : le modèle prend la main.
4. **X** gauche si l'essai est réussi, **Y** gauche s'il est raté, **B** droit pour l'annuler. **Grip** maintenu pour corriger.
5. **`q`** pour quitter. Arrêtez ensuite le serveur avec Ctrl+C dans le terminal 1.