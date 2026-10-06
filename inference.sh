#!/usr/bin/env bash
# Inference Studio : choisir un modèle, lancer serveur + téléop en mode politique, piloter les essais.
# Utilise TOUJOURS l'interpréteur de l'env g1d_teleop (même si un autre env / le .venv est actif dans le
# terminal), avec ses variables (certificats du casque).
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate g1d_teleop 2>/dev/null \
    || { echo "env g1d_teleop introuvable" >&2; exit 1; }
cd "$(dirname "$(realpath "$0")")" && exec "$CONDA_PREFIX/bin/python" -m inference_studio "$@"
