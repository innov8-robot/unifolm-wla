#!/usr/bin/env bash
# Crée l'environnement conda de téléop G1-D, installé depuis CE dépôt (unifolm-wla/teleoperation).
#
#   bash teleoperation/setup_env.sh                 # env "g1d_teleop"
#   ENV_NAME=autre bash teleoperation/setup_env.sh
#   WITH_DEX_RETARGETING=1 bash teleoperation/setup_env.sh   # mains Dex3 / Inspire / Brainco
#
# Versions reprises de l'env "tv" qui a téléopéré le G1-D + Dex1. Le retargeting des mains est
# optionnel : inutile pour la Dex1, il tire torch et un second pinocchio en pip qui masquerait
# celui de conda-forge (requis pour pinocchio.casadi, utilisé par l'IK des bras).
set -euo pipefail

ENV_NAME="${ENV_NAME:-g1d_teleop}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XR="$HERE/Tele_OP/xr_teleoperate"
SDK="$HERE/Tele_OP/unitree_sdk2_python"
CONDA="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"

if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "L'env $ENV_NAME existe déjà : supprimez-le d'abord ($CONDA env remove -n $ENV_NAME) ou changez ENV_NAME." >&2
    exit 1
fi

"$CONDA" create -y -n "$ENV_NAME" -c conda-forge python=3.10 pinocchio=3.1.0 casadi=3.6.7 numpy=1.26.4
PIP=("$CONDA" run -n "$ENV_NAME" --no-capture-output python -m pip)

"${PIP[@]}" install "vuer[all]==0.0.60" params-proto==2.13.2 pyyaml pyzmq logging-mp==0.2.0 \
    meshcat==0.3.2 matplotlib rerun-sdk==0.23.1 sshkeyboard==2.3.1 opencv-python==4.11.0.86 "numpy==1.26.4" \
    "PySide6>=6.6"                                    # Dataset Studio (python -m dataset_studio)
"${PIP[@]}" install -e "$SDK"                       # tire cyclonedds==0.10.2
"${PIP[@]}" install -e "$XR/teleop/teleimager" --no-deps
"${PIP[@]}" install -e "$XR/teleop/televuer"
# voix naturelle des annonces (Piper, hors ligne) + voix française ; sans elle : repli sur spd-say
"${PIP[@]}" install piper-tts
VOICES="$HOME/.local/share/piper-voices"; mkdir -p "$VOICES"
for ext in onnx onnx.json; do
    [[ -f "$VOICES/fr_FR-siwis-medium.$ext" ]] || curl -sSLf -o "$VOICES/fr_FR-siwis-medium.$ext" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.$ext"
done
if [[ "${WITH_DEX_RETARGETING:-0}" == "1" ]]; then
    "${PIP[@]}" install -e "$XR/teleop/robot_control/dex-retargeting"
fi

# Certificats HTTPS du casque (non versionnés) : pointés par variables d'env de l'env conda.
if [[ -f "$XR/teleop/televuer/cert.pem" && -f "$XR/teleop/televuer/key.pem" ]]; then
    "$CONDA" env config vars set -n "$ENV_NAME" \
        XR_TELEOP_CERT="$XR/teleop/televuer/cert.pem" XR_TELEOP_KEY="$XR/teleop/televuer/key.pem"
else
    echo "⚠ cert.pem / key.pem absents de $XR/teleop/televuer : les générer (README xr_teleoperate §1.1)." >&2
fi

"$CONDA" run -n "$ENV_NAME" python - <<'EOF'
import pinocchio, pinocchio.casadi, casadi, vuer, televuer, teleimager, unitree_sdk2py, cyclonedds
print("OK : pinocchio", pinocchio.__version__, "| casadi", casadi.__version__, "| numpy", __import__("numpy").__version__)
EOF
echo "Env $ENV_NAME prêt : conda activate $ENV_NAME && cd $XR/teleop"
