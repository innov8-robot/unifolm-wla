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
conda activate tv
cd /home/thomas/Documents/project/teleoperation/Tele_OP/xr_teleoperate/teleop

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
