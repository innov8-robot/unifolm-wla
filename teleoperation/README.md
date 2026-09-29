python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --img-server-ip=10.3.10.124 --input-mode=controller --motion
python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --img-server-ip=10.3.10.124  --motion --ee=brainco
python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --headless   --motion --ee=brainco
python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --headless --input-mode=controller --motion
python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --headless --motion --ee=dex1 --input-mode=controller


# Environnement (une seule fois, depuis la racine du dépôt unifolm-wla) :
bash teleoperation/setup_env.sh          # crée l'env conda "g1d_teleop", certificats inclus
conda activate g1d_teleop
cd teleoperation/Tele_OP/xr_teleoperate/teleop/


changer ipv4 192.168.123.99 255.255.255.0


python height_tool.py read  
python height_tool.py goto 0.02

python height_tool.py home              # descend à la butée basse, mémorise le fond
python height_tool.py goto --rel 0.10   # 10 cm AU-DESSUS du fond (répétable !)
python height_tool.py read    

~/.cache/huggingface/lerobot/ 