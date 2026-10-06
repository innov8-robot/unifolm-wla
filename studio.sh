#!/usr/bin/env bash
# Dataset Studio, lançable depuis n'importe quel dossier, avec l'interpréteur de l'env g1d_teleop
# (même si un autre env / le .venv est actif dans le terminal) :
#   ./studio.sh                 # ouvre teleoperation/.../teleop/utils/data
#   ./studio.sh <dossier>       # une tâche ou un dossier de tâches
args=()
for a in "$@"; do [[ -e "$a" ]] && a="$(realpath "$a")"; args+=("$a"); done
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate g1d_teleop 2>/dev/null \
    || { echo "env g1d_teleop introuvable" >&2; exit 1; }
cd "$(dirname "$(realpath "$0")")" && exec "$CONDA_PREFIX/bin/python" -m dataset_studio "${args[@]}"
