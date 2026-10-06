#!/usr/bin/env bash
# Inference Studio : choisir un modèle, lancer serveur + téléop en mode politique, piloter les essais.
# Active l'env g1d_teleop (ses variables : certificats du casque) puis lance l'application.
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate g1d_teleop || { echo "env g1d_teleop introuvable" >&2; exit 1; }
cd "$(dirname "$(realpath "$0")")" && exec python -m inference_studio "$@"
