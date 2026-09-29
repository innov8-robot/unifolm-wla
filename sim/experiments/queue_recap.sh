#!/usr/bin/env bash
# Expérience RECAP en sim (tâche Novares, zone décalée = « cuisine inversée ») — une itération complète.
#   setsid nohup bash playground/queue_recap.sh > playground/queue_logs/recap.log 2>&1 &
#
#  0. démos : 10 démos Novares ré-enregistrées (même graine que le jeu n10 -> mêmes épisodes) avec
#     le pas de réussite, pour le modèle de valeur ;
#  1. référence : politique n10 (file Novares) évaluée dans la zone décalée ;
#  2. rollouts : la politique n10 joue 40 épisodes dans la zone décalée, opérateur simulé ;
#  3. modèle de valeur (démos + rollouts), étiquetage de l'avantage par morceau de 30 pas ;
#  4. fine-tuning conditionné par l'avantage (démos + rollouts), depuis le modèle Base ;
#  5. évaluation : zone décalée (« Advantage: positive ») et zone d'origine ;
#  6. témoin : mêmes données SANS conditionnement (l'effet vient-il de l'avantage ou des données ?).
#
# Mêmes garde-fous que queue_novares.sh (attend une machine libre, pose playground/.gpu_busy).
set -uo pipefail
cd /home/thomas/Documents/project/manip/unifolm-wla
ROOT=$PWD
LOGS=$ROOT/playground/queue_logs
mkdir -p "$LOGS" playground/recap
SIMPY=$HOME/miniconda3/envs/unitree_lerobot/bin/python
VENV=(env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8)
BUSY=$ROOT/playground/.gpu_busy
CFG=unifolm_wla/dataloader/multi_source_dataset/configs
RAW=playground/sim_raw
BASE_CKPT=playground/Checkpoints/g1d_sim_novares_n10/final_model/model.safetensors
say() { echo "[$(date '+%F %T')] $*"; }

machine_free() {
    [[ -e "$BUSY" ]] && return 1
    local pid
    for pid in $(pgrep -f "train_unifolm_wla.py|action_server_wbc_msgpack_unitree|queue_novares.sh"); do
        [[ "$(ps -o comm= -p "$pid" 2>/dev/null)" == python* || "$(ps -o args= -p "$pid")" == "bash playground/queue_novares.sh" ]] && return 1
    done
    local gpu ram
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    ram=$(free -g | awk '/^Mem:/ {print $7}')
    (( gpu < 2000 && ram >= 18 ))
}
wait_free() {
    say "attente d'une machine libre"
    local ok=0
    while (( ok < 12 )); do
        if machine_free; then ok=$((ok + 1)); else ok=0; fi
        sleep 10
    done
    say "machine libre"
}
SRV=""
serve() {  # serve <ckpt> <tag> [--advantage positive]
    local ckpt=$1 tag=$2; shift 2
    "${VENV[@]}" .venv/bin/python -m model_server.action_server_wbc_msgpack_unitree --ckpt_path "$ckpt" \
        --unnorm_key UnifoLM_G1_Dex1 --port 8600 "$@" > "$LOGS/recap_server_$tag.log" 2>&1 &
    SRV=$!
    until grep -q "server listening on" "$LOGS/recap_server_$tag.log"; do
        kill -0 $SRV 2>/dev/null || { say "ÉCHEC serveur $tag"; exit 1; }
        sleep 5
    done
}
unserve() { [[ -n "$SRV" ]] && kill $SRV && wait $SRV 2>/dev/null; SRV=""; }
evaluate() {  # evaluate <scene> <tag> [--advantage positive]
    local scene=$1 tag=$2; shift 2
    MUJOCO_GL=egl $SIMPY sim/wla_client.py --scene "$scene" --instruction "pick up the black part" --head-view raw \
        --episodes 30 --max-steps 300 --exec-steps 30 --stop-on-success --videos 2 --seed 7 "$@" \
        --out "sim/wla_out/recap_${tag}" 2>&1 | grep -E "réussite|Error|Traceback" | sed "s|^|    [$tag] |"
}
train() {  # train <run_id> <config>
    local run=$1 cfg=$2 free_gb
    free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
    if (( free_gb < 16 )); then say "ARRÊT : ${free_gb} Go libres"; exit 1; fi
    say "entraînement $run ($cfg)"
    rm -rf "playground/Checkpoints/$run"
    run_id=$run bash examples/unifolm_wla/train_files/run_finetune_g1d.sh \
        --datasets.vla_data.data_config_path $CFG/$cfg \
        --datasets.vla_data.per_device_batch_size 2 --datasets.vla_data.num_workers 4 \
        --trainer.gradient_accumulation_steps 1 --trainer.max_train_steps 3000 --trainer.num_warmup_steps 100 \
        --trainer.save_interval 1000000 --trainer.eval_interval 1000000 --trainer.logging_frequency 50 \
        > "$LOGS/recap_train_$run.log" 2>&1 || { say "ÉCHEC entraînement $run"; exit 1; }
}
cleanup() { unserve; rm -f "$BUSY"; }
trap cleanup EXIT

say "expérience RECAP (Novares, zone décalée)"
wait_free
touch "$BUSY"
[[ -f "$BASE_CKPT" ]] || { say "ARRÊT : checkpoint n10 absent ($BASE_CKPT)"; exit 1; }

say "0. démos avec pas de réussite"
MUJOCO_GL=egl $SIMPY sim/record_sim_demos.py --task novares --n 10 --seed 1000 --out $RAW/recap_demos_n10 --overwrite \
    > "$LOGS/recap_demos.log" 2>&1 || { say "ÉCHEC démos"; exit 1; }
tail -1 "$LOGS/recap_demos.log" | sed 's|^|    |'

say "1. référence : politique n10, zone décalée et zone d'origine"
serve "$BASE_CKPT" base
evaluate novares_shift n10_shift
evaluate novares n10_origin

say "2. rollouts de la politique n10 dans la zone décalée, opérateur simulé"
MUJOCO_GL=egl $SIMPY sim/recap_rollouts.py --task novares_shift --episodes 40 --seed 2000 \
    --out $RAW/recap_rollouts_r1 --overwrite > "$LOGS/recap_rollouts.log" 2>&1 || { say "ÉCHEC rollouts"; exit 1; }
tail -1 "$LOGS/recap_rollouts.log" | sed 's|^|    |'
unserve

say "3. modèle de valeur et étiquetage"
"${VENV[@]}" .venv/bin/python -m g1d_wla.recap train-value --episodes $RAW/recap_demos_n10 $RAW/recap_rollouts_r1 \
    --out playground/recap/value_r1.pt > "$LOGS/recap_value.log" 2>&1 || { say "ÉCHEC valeur"; exit 1; }
grep "validation" "$LOGS/recap_value.log" | tail -1 | sed 's|^|    |'
"${VENV[@]}" .venv/bin/python -m g1d_wla.recap label --all-positive --episodes $RAW/recap_demos_n10 > "$LOGS/recap_label.log" 2>&1
"${VENV[@]}" .venv/bin/python -m g1d_wla.recap label --value playground/recap/value_r1.pt \
    --episodes $RAW/recap_rollouts_r1 >> "$LOGS/recap_label.log" 2>&1 || { say "ÉCHEC étiquetage"; exit 1; }
grep "pas de la politique" "$LOGS/recap_label.log" | sed 's|^|    |'

say "4. conversion et fine-tuning conditionné par l'avantage"
for part in recap_demos_n10 recap_rollouts_r1; do
    "${VENV[@]}" .venv/bin/python -m g1d_wla.convert_teleop --raw-dir $RAW/$part \
        --out-dir playground/Datasets/g1d_recap_r1/$part --repo-id innov8/$part --advantage on --overwrite \
        > "$LOGS/recap_convert_$part.log" 2>&1 || { say "ÉCHEC conversion $part"; exit 1; }
done
sed -e 's|data_path: "g1d_sim_novares/".*|data_path: "g1d_recap_r1/"\n    advantage_key: "advantage"\n    advantage_dropout: 0.3|' \
    $CFG/g1d_sim_novares.yaml > $CFG/g1d_recap_r1.yaml
sed -e 's|data_path: "g1d_sim_novares/".*|data_path: "g1d_recap_r1/"|' $CFG/g1d_sim_novares.yaml > $CFG/g1d_recap_r1_noadv.yaml
train g1d_recap_r1 g1d_recap_r1.yaml

say "5. évaluation du modèle RECAP"
serve playground/Checkpoints/g1d_recap_r1/final_model/model.safetensors recap --advantage positive
evaluate novares_shift r1_shift --advantage positive
evaluate novares r1_origin --advantage positive
unserve

say "6. témoin : mêmes données, sans conditionnement"
train g1d_recap_r1_noadv g1d_recap_r1_noadv.yaml
serve playground/Checkpoints/g1d_recap_r1_noadv/final_model/model.safetensors noadv
evaluate novares_shift r1noadv_shift
evaluate novares r1noadv_origin
unserve
say "expérience RECAP terminée"
