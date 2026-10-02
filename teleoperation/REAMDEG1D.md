# Démarrage téléop G1 (dans l'ordre)

## 1) ROBOT (PC2) — via SSH
```bash
ssh unitree@192.168.123.164          # mot de passe : celui du robot (non versionné)

# --- Caméras (image server) ---
sudo systemctl restart teleimager.service
journalctl -u teleimager.service -n 15 --no-pager     # OK si : head_camera / left_wrist / right_wrist is ready
# si aucune /dev/video* :  sudo modprobe -r uvcvideo && sudo modprobe uvcvideo   puis relancer le restart

# --- Pinces ---
# G1-D : Dex1 CÂBLÉES EN INTERNE (moteurs 31/33 du LowCmd) -> PAS de service à lancer.
# dex1_1_gripper.service et rt/dex1/* ne servent qu'aux Dex1 en USB (--dex1-bus usb).
```

**Caméras sur ce robot (1er octobre 2026)** : la tête stéréo stock est publiée par teleimager sur le port **55558**, pour cohabiter avec la RealSense d'un autre projet (`~/rs_stream.py`, ports 55555/55565, à ne pas arrêter). Les caméras de poignet ne sont pas détectées : désactivées dans `cam_config_server.yaml` (sauvegarde `.bak.20261001_174425`), à réactiver une fois rebranchées. La téléop lit la config du serveur (port 60000) et suit le port toute seule.

**Réglé automatiquement depuis le 2 octobre 2026** : le robot exécute `/usr/local/bin/g1d_cam_rebind.sh` avant chaque démarrage de teleimager (drop-in `/etc/systemd/system/teleimager.service.d/rebind.conf`, copies dans `teleoperation/robot_config/`). Le script rattache uvcvideo aux 3 caméras par numéro de série (tête 01.00.00, poignets JR0001 / JR0002), sans toucher à la RealSense. Retrait : supprimer ces deux fichiers puis `sudo systemctl daemon-reload`. La procédure manuelle ci-dessous reste valable en dépannage.

**Caméras qui disparaissent après un plantage de teleimager** : à l'arrêt, teleimager relâche les caméras et veut recharger le pilote `uvcvideo`, ce qui échoue car la RealSense de l'autre projet l'utilise (« Module uvcvideo is in use »). Les caméras restent alors SANS pilote et teleimager ne les retrouve plus (« Cannot find UVCCamera »). Réparation sur le robot, sans toucher à la RealSense :
```bash
sudo systemctl stop teleimager.service
for i in 1-2.1:1.0 1-2.1:1.1 1-3.1:1.0 1-3.1:1.1 1-3.2:1.0 1-3.2:1.1; do echo $i | sudo tee /sys/bus/usb/drivers/uvcvideo/bind; done
sudo systemctl start teleimager.service      # attendu : 3 × « is ready », ports 55556/55557/55558
```
(1-2.1 = tête stéréo, 1-3.1 = JR0001 = poignet DROIT, 1-3.2 = JR0002 = poignet GAUCHE ; vérifier avec `lsusb -t`. Si la tête est en échec « UVC probe control », la réinitialiser : `echo 0 | sudo tee /sys/bus/usb/devices/1-2.1/authorized; sleep 2; echo 1 | sudo tee …/authorized`.) Les trois caméras partagent un bus USB 2 : la tête tombe à ~25 images/s.

**Buste** : la rotation est le moteur **12** (vérifié dans mpc_any, kp 180 / kd 2,6). Le **tangage n'est ni commandé ni mesuré** (indices 13/14 à 0,000) : enregistrer avec `--torso-pitch <angle mesuré>`, pas `--torso-pitch-index`. `python read_lowstate.py` lit les 35 moteurs sans rien commander.

**Un seul programme sur les moteurs** : les caméras se partagent, pas `rt/lowcmd`.

## 2) PC (laptop)
```bash
conda activate g1d_teleop          # créé par : bash teleoperation/setup_env.sh
cd teleoperation/Tele_OP/xr_teleoperate/teleop   # depuis la racine du dépôt unifolm-wla

# téléop complète (bras + pinces + caméra tête + vignettes poignets en VR)
python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --img-server-ip=192.168.123.164 \
    --input-mode=controller --arm=G1_29 --ee=dex1 --torso-pitch 0.166
# Dex1 internes par défaut (--dex1-bus internal) ; + --torso-yaw-index 12 pour tourner le buste au joystick droit
```

## 3) CASQUE VR — par câble USB (recommandé : le Wi-Fi donne une grosse latence)

Mesuré le 1er octobre 2026 avec `--timing` : par le Wi-Fi, les poses des manettes arrivent PAR PAQUETS
(cible qui saute de 90 à 270 mm en 1/60 s, puis immobile) ; par le câble USB, 2 à 33 mm par pas.

```bash
sudo apt install adb                 # une fois
# Pico 4 Ultra : Paramètres > Général > À propos > cliquer 7-10 fois sur « Version du logiciel »,
# puis Paramètres > Développeur > Débogage USB. Brancher le câble DIRECTEMENT sur le PC (pas le dock),
# accepter « Autoriser le débogage USB » dans le casque.
adb devices                          # attendu : <numéro>  device
adb reverse tcp:8012 tcp:8012        # à refaire après chaque rebranchement
```
Dans le navigateur du casque : `https://localhost:8012` (accepter le certificat), puis
`https://vuer.ai/?ws=wss://localhost:8012&grid=False` -> Virtual Reality. Bien **localhost**, pas l'IP Wi-Fi.
Vérification côté PC : `ss -tn sport = :8012` ne doit plus montrer l'IP Wi-Fi du casque.

Réglages conseillés : `--frequency 60 --ik-smooth light` (enregistrement toujours à 30 Hz).

## 3 bis) CASQUE VR — par Wi-Fi (dépannage seulement)
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


## Rotation du buste (G1-D, `torso_Joint`)

**Non validé sur le robot.** Le G1-D a un moteur qui tourne le buste à gauche et à droite, au-dessus du tangage. WLA le connaît : c'est le lacet de la taille du G1, déjà présent dans les données d'entraînement d'Unitree.

- **Activer** : `--torso-yaw-index <n>`, l'indice du moteur dans les 35. **HYPOTHÈSE à vérifier** avant tout essai : dans la disposition G1_29, la taille en lacet est à l'indice 12. Bornes : `--torso-yaw-max 0.6` rad et `--torso-yaw-rate 0.5` rad/s.
- **Commande** : **joystick droit, gauche/droite** (zone morte 0,2). Sans `--motion`, ce joystick ne pilotait rien ; la commande de base enregistrée vaut désormais 0, puisque la base ne bouge pas.
- **Effet sur les bras** : l'IK est résolue dans un repère lié au torse, donc tourner le buste emporte les bras, comme quand on pivote sur soi-même. La caméra de tête tourne avec.
- **Mode politique** : la rotation prédite par le modèle est appliquée. Le joystick y ajoute une correction, comptée comme intervention. Pendant une tenue, le buste reste figé.
- **Enregistrement** : l'angle mesuré est dans `body.qpos`. Les indices sont écrits dans `info.body_layout`, que le convertisseur lit.
- **Arrêt** : le buste est ramené droit, à vitesse bornée, avant le retour des bras au repos.

**Pourquoi le joystick plutôt que suivre la tête** : regarder ailleurs ferait tourner le robot sans le vouloir, et le casque ne voit pas le bassin de l'opérateur. Le joystick est explicite, se dose facilement, et laisse la tête libre.

## Latence (audit du 1er octobre 2026)

Mesuré hors robot (moteurs supposés parfaits) : délai pour suivre 90 % d'un déplacement de 5 cm de la manette.

| Lissage IK (`--ik-smooth`) | 30 Hz (défaut) | 60 Hz (`--frequency 60`) |
|---|---|---|
| `standard` (amont, 4 pas) | 133 ms | 67 ms |
| `light` (2 pas) | 67 ms | **33 ms** |
| `off` | 67 ms | 33 ms |

- L'IK elle-même prend moins de 1,2 ms : ce n'est pas elle.
- La partie logicielle se réduit d'environ 100 ms avec `--frequency 60 --ik-smooth light`. L'enregistrement reste à 30 Hz (`--record-fps 30`, un pas sur deux gardé). Le mode politique impose 30 Hz.
- Le reste vient des moteurs (gains amont kp 80 aux épaules et coudes), du Wi-Fi casque → PC et du casque lui-même. `--timing` affiche toutes les 2 s la fréquence de boucle, le temps d'IK, l'écart consigne-mesure des bras et les sauts de cible (des sauts réguliers = poses de manette qui arrivent par paquets = Wi-Fi).
- Wi-Fi : pendant une session, `ss -tn sport = :8012` donne l'IP du casque, puis `ping <ip>`. Viser un Wi-Fi 5 GHz dédié.

## Commandes de la manette (téléop, sans `--motion`)

| Commande | Effet |
|---|---|
| `r` (clavier) | Démarrer la téléop |
| **A** droit / `s` | Démarrer puis arrêter + sauvegarder un enregistrement |
| **B** droit | Annuler l'enregistrement en cours |
| **Y** gauche | **Pause + recalage** : le suivi s'arrête, les bras vont doucement (2,5 s) en posture de calibration — position zéro du G1, bras le long du corps, avant-bras vers l'avant, coudes ~80°. Prenez la même posture, puis **Y** à nouveau : la pose actuelle des manettes devient celle des mains du robot (recalage en position, pas de saut). Pinces figées et pas non enregistrés pendant la pause. Hors mode politique (Y = essai raté) |
| Gâchettes | Pinces |
| Joystick droit ←/→ | Rotation du buste (`--torso-yaw-index 12`). **Au lancement, le buste revient au centre** (0 rad) à vitesse bornée, avant `r` |
| Joystick droit ↑/↓ | **Colonne** (`--column`) : monter / descendre le buste (axe dominant du joystick : pas de rotation en même temps). Au lancement, descente en **butée basse** = référence 0 (le zéro du capteur dérive). Bornes : jamais sous la butée, `--column-max 0.40` m au-dessus, ralentissement dans les 3 derniers cm. Hauteur enregistrée (`states.column`) -> `action.base_command[3]` = 0,732 + hauteur |
| Joystick **gauche** | **Base roulante** (`--base`) : ↑/↓ avancer/reculer, ←/→ tourner. Bornes `--base-max-vx 0.3` m/s, `--base-max-vyaw 0.4` rad/s (châssis : 1,0 / 0,6). Arrêt si la manette gauche ne bouge plus depuis 0,3 s (casque déconnecté), pendant la pause Y, et à la sortie (zéros répétés : la base roule ~1,5 s sinon). Enregistré dans `action.base_command` |
| `q` (clavier) | Quitter (buste ramené droit, bras rentrés lentement) |

Le recalage porte sur la position des mains ; l'orientation reste celle des manettes : tenir les manettes « avant-bras vers l'avant » à la reprise.

**Base roulante (`--base`)** : service RPC « agv » du robot (api 1001, `Move(vx, vy, vyaw)`, vy ignoré), repris de mpc_any. Ce service contourne les limiteurs du châssis : les bornes sont dans `robot_control/g1d_base.py`. Pour les démos WLA, placer le robot AVANT d'appuyer sur A : les données d'entraînement d'Unitree sont à base immobile pour les tâches de table. Incompatible avec `--motion` et `--policy-uri`.

**Colonne (`--column`)** : `rt/cmd_hispeed` (Point32, VITESSE en `.z`, streamée à 50 Hz) et `rt/hispeed_state` (hauteur en `.y`), repris de mpc_any (`runtime/column.py`). La butée haute réelle n'est pas confirmée (mpc_any suppose 0,70 m ; URDF : 2 × 0,21 m) : `--column-max` reste prudent. ⚠ Au lancement les bras DESCENDENT avec la colonne : rien sous les bras. Pour WLA, la hauteur est convertie en hauteur de bassin G1 équivalente (0,732 + h) ; le G1 ne fait que s'accroupir (q99 = 0,794 m), donc au-delà d'environ +6 cm au-dessus de la butée basse, hors des données d'entraînement.

**Annonces vocales (`--voice`, défaut `pc`)** : « Téléopération démarrée », « Enregistrement, épisode N », « Épisode N sauvegardé », « Épisode annulé », « Sauvegarde en cours, attendez », « Pause » / « Reprise » (Y), « Essai réussi / raté » (mode politique). `--voice robot` : haut-parleur du robot (service « voice », en anglais) ; `--voice off` : silence.

**Mode politique — entre deux essais** (aucun essai en cours) : **Y** = bras en garde (posture de calibration, coudes ~80°), Y de nouveau = garde relâchée ; joystick gauche = base, joystick droit ↑/↓ = colonne (avec `--base` / `--column`). Un essai lancé (A) depuis la garde part de cette posture. **Pendant un essai** : Y = essai raté, base et colonne bloquées à l'arrêt.

**Affichage dans le casque : pass-through par défaut** (depuis le 2 octobre 2026). Le flux vidéo vers le casque (JPEG 1280×480 à 30 images/s + vignettes des poignets) faisait saccader l'affichage, les manettes et donc le robot à intervalle régulier : le navigateur du casque n'arrivait pas à suivre. En pass-through, le casque montre la pièce réelle et ne reçoit plus d'image ; l'enregistrement et le mode politique lisent toujours les caméras. Pour revoir la caméra du robot dans le casque : `--display-mode immersive` (ou `ego`).
