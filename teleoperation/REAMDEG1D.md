# Démarrage téléop G1 (dans l'ordre)

## 1) ROBOT (PC2) — via SSH
```bash
ssh unitree@192.168.123.164          # mot de passe : celui du robot (non versionné)

# --- Caméras (image server) ---
sudo systemctl restart teleimager.service
journalctl -u teleimager.service -n 15 --no-pager     # OK si : head_camera / left_wrist / right_wrist is ready
# si aucune /dev/video* :  sudo modprobe -r uvcvideo && sudo modprobe uvcvideo   puis relancer le restart

# --- Pinces (dex1 gripper server) ---
sudo systemctl restart dex1_1_gripper.service
journalctl -u dex1_1_gripper.service -n 10 --no-pager # OK si 2 moteurs : Side: left ET Side: right
```

## 2) PC (laptop)
```bash
conda activate g1d_teleop          # créé par : bash teleoperation/setup_env.sh
cd teleoperation/Tele_OP/xr_teleoperate/teleop   # depuis la racine du dépôt unifolm-wla

# téléop complète (bras + pinces + caméra tête + vignettes poignets en VR)
python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --img-server-ip=192.168.123.164 --input-mode=controller --ee=dex1
```

## 3) CASQUE VR
```text
# IP wifi du PC (si elle a changé) :  ip -brief addr show wlp131s0f0    (ex : 10.3.8.62)
# 1. Accepter le certificat : ouvrir   https://10.3.8.62:8012   -> Advanced -> Proceed
# 2. Ouvrir   https://vuer.ai/?ws=wss://10.3.8.62:8012&grid=False   -> Virtual Reality
# 3. Terminal :  [r] démarrer la téléop   |   [s] démarrer/arrêter un enregistrement   |   [q] quitter
#    Robot debout en Regular mode (R1+X) si tu utilises --motion
```

---

## Variantes de lancement
```bash
# sans caméra / affichage (debug rapide)
python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --headless --input-mode=controller

# sans les pinces
python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --img-server-ip=192.168.123.164 --input-mode=controller

# avec enregistrement
python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --img-server-ip=192.168.123.164 --input-mode=controller --ee=dex1 --record --task-name=mon_test

python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --img-server-ip=192.168.123.164 --input-mode=controller --ee=dex1 --record --task-name=mon_test --right-only

# désactiver les vignettes poignets dans le casque
#   ... --ee=dex1 --no-wrist-pip

# avec déplacement de la base (robot debout, Regular mode R1+X)
#   ... --ee=dex1 --motion
```

---

## Remettre la caméra stéréo stock (après l'essai Orbbec)

À faire une fois sur le robot. Le code Orbbec a été retiré du dépôt : un serveur configuré en `type: orbbec` ne démarrera plus la tête.

```bash
# depuis le PC, à la racine du dépôt unifolm-wla
scp teleoperation/robot_config/head_camera_stereo.yaml unitree@192.168.123.164:~/
ssh unitree@192.168.123.164
F=/home/unitree/unitree_eai_environment/service/teleimager/cam_config_server.yaml
sudo cp $F $F.bak.$(date +%Y%m%d)            # sauvegarde
# remplacer le bloc head_camera: de $F par le contenu de ~/head_camera_stereo.yaml, puis :
sudo systemctl restart teleimager.service
journalctl -u teleimager.service -n 15 --no-pager   # attendu : head_camera ... is ready
```

Vérification depuis le PC, dans l'env `g1d_teleop` :

```python
from teleimager.image_client import ImageClient
c = ImageClient(host="192.168.123.164", request_bgr=True); c.get_cam_config()
print(c.get_head_frame().bgr.shape)   # attendu : (480, 1280, 3)
```

Si le serveur du robot contient encore le code Orbbec, ce n'est pas gênant tant que la config est en `type: uvc`.

---

## Mode politique avec corrections (RECAP / Delta-0)

La politique WLA pilote les bras, et l'opérateur, casque sur la tête, corrige en direct. Les essais enregistrés servent ensuite à réentraîner le modèle : voir `docs/G1D_Constats.md`, section 15.

⚠ **Non validé sur le robot.** Premiers essais : vitesse bridée, zone dégagée, arrêt d'urgence à portée, et quelqu'un prêt à appuyer sur `q`.

**1. Sur le PC d'inférence**, env uv du projet, avec le modèle à améliorer :
```bash
.venv/bin/python -m model_server.action_server_wbc_msgpack_unitree --unnorm_key UnifoLM_G1_Dex1 \
    --ckpt_path playground/Checkpoints/<run>/final_model/model.safetensors --port 8600
    # + --advantage positive pour un modèle déjà entraîné par RECAP
```

**2. Sur le PC de téléop**, env `g1d_teleop` :
```bash
python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --img-server-ip=192.168.123.164 \
    --input-mode=controller --arm=G1_29 --ee=dex1 --record --task-name=recap_r1 --task-goal="pick up the black part" \
    --policy-uri ws://<ip-du-PC-d-inférence>:8600 --policy-max-speed 0.10
```

**3. Pendant les essais :**

| Commande | Effet |
|---|---|
| `r` | Démarrer. Les bras tiennent leur pose tant qu'aucun essai n'est en cours |
| `A` droit, ou `s` | Commencer un essai : la politique prend la main, et l'enregistrement démarre |
| **Grip** d'une manette, maintenu | Correction : le déplacement de la manette depuis l'appui s'ajoute au geste du bras de ce côté. Au relâchement, le modèle replanifie |
| Gâchette pendant la correction | L'opérateur tient la pince de ce côté |
| `X` gauche | Fin d'essai **réussi**, épisode sauvegardé |
| `Y` gauche | Fin d'essai **raté**, épisode sauvegardé |
| `B` droit | Annuler l'essai en cours, épisode jeté |
| `q` | Quitter |

**Conseils :**
- corriger tôt et petit ;
- laisser aussi quelques échecs aller au bout, car ils servent au modèle de valeur ;
- viser 30 à 50 essais par cycle.

**Options :**

| Option | Défaut | Rôle |
|---|---|---|
| `--policy-max-speed` | 0,10 m/s | Vitesse maximale des cibles |
| `--policy-exec-steps` | 30 | Pas exécutés par chunk |
| `--policy-advantage` | — | Condition envoyée au modèle, par exemple `positive` |
| `--torso-pitch-index` | 13 | Indice du tangage du buste dans les 35 moteurs. **Hypothèse à vérifier** |
| `--torso-pitch` | — | Tangage constant, en rad, à la place de l'indice |
| `--right-only` | — | Un seul bras, le gauche reste immobile |

L'aide de la téléop n'affiche pas ces options, pas plus que les options d'origine comme `--task-goal`. Elle ne montre que celles du casque.

**Test hors robot :**
```bash
cd teleoperation/Tele_OP/xr_teleoperate/teleop && python test_policy_bridge.py
```
Il couvre les repères, la pince et la logique de correction avec un faux serveur.

**Ensuite, sur le PC d'inférence** : modèle de valeur, étiquetage, conversion et réentraînement.
```bash
.venv/bin/python -m g1d_wla.recap train-value --episodes <démos> teleoperation/.../utils/data/recap_r1 --out playground/recap/value_r1.pt
.venv/bin/python -m g1d_wla.recap label --value playground/recap/value_r1.pt --episodes teleoperation/.../utils/data/recap_r1
.venv/bin/python -m g1d_wla.recap label --all-positive --episodes <démos>
# conversion de chaque dossier avec --advantage on, puis fine-tuning avec advantage_key dans la config de données
```
Le modèle de valeur a besoin du pas de réussite de chaque démo. Les démos réelles n'en ont pas encore, et il faudra le marquer, par exemple en réenregistrant avec `X` en fin de démo.

