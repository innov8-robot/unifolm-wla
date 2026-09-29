python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --img-server-ip=10.3.10.124 --input-mode=controller --motion
python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --img-server-ip=10.3.10.124  --motion --ee=brainco
python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --headless   --motion --ee=brainco
python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --headless --input-mode=controller --motion
python teleop_hand_and_arm.py --network-interface=enx0c3796e0bc5b --headless --motion --ee=dex1 --input-mode=controller


conda create -n tv python=3.10 -y
conda activate tv
export XR_TELEOP_CERT=/home/thomas/Documents/project/teleoperation/Tele_OP/xr_teleoperate/teleop/televuer/cert.pem
export XR_TELEOP_KEY=/home/thomas/Documents/project/teleoperation/Tele_OP/xr_teleoperate/teleop/televuer/key.pem
cd Tele_OP/xr_teleoperate/teleop/


conda install pinocchio=3.1.0 -c conda-forge -y
pip install casadi "vuer[all]==0.0.60" params-proto==2.13.2 pyyaml pyzmq
pip install meshcat
pip install matplotlib
pip install rerun-sdk
pip install "rerun-sdk" "numpy==1.26.4"
pip install sshkeyboard


changer ipv4 192.168.123.99 255.255.255.0


python height_tool.py read  
python height_tool.py goto 0.02

python height_tool.py home              # descend à la butée basse, mémorise le fond
python height_tool.py goto --rel 0.10   # 10 cm AU-DESSUS du fond (répétable !)
python height_tool.py read    

~/.cache/huggingface/lerobot/ 