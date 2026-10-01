#!/usr/bin/env bash
# Dataset Studio, lançable depuis n'importe quel dossier (env g1d_teleop) :
#   ./studio.sh                 # ouvre teleoperation/.../teleop/utils/data
#   ./studio.sh <dossier>       # une tâche ou un dossier de tâches
args=()
for a in "$@"; do [[ -e "$a" ]] && a="$(realpath "$a")"; args+=("$a"); done
cd "$(dirname "$(realpath "$0")")" && exec python -m dataset_studio "${args[@]}"
