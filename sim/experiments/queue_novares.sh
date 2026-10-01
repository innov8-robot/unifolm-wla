#!/usr/bin/env bash
# File de travaux Novares (sim) : 50, 25 puis 10 démos -> fine-tuning -> évaluation.
# Démarre seule quand la machine est libre ; lancer avec :
#   setsid nohup bash sim/experiments/queue_novares.sh > playground/queue_logs/queue.log 2>&1 &
#
# Garde-fous (machine partagée avec une autre session Claude) :
#   - attend, 120 s d'affilée : aucun entraînement ni serveur WLA, GPU < 2 Go utilisés, >= 18 Go
#     de RAM disponibles, et pas de fichier playground/.gpu_busy (quiconque peut le créer pour
#     retenir la file ; elle attend qu'il disparaisse) ;
#   - pose playground/.gpu_busy pendant ses travaux et le retire à la fin ;
#   - disque : s'arrête s'il reste moins de 16 Go avant un entraînement.
set -uo pipefail
cd /home/thomas/Documents/project/manip/unifolm-wla
ROOT=$PWD
LOGS=$ROOT/playground/queue_logs
mkdir -p "$LOGS"
SIMPY=$HOME/miniconda3/envs/unitree_lerobot/bin/python
BUSY=$ROOT/playground/.gpu_busy
CFG=unifolm_wla/dataloader/multi_source_dataset/configs
say() { echo "[$(date '+%F %T')] $*"; }

machine_free() {
    [[ -e "$BUSY" ]] && return 1
    # seulement de VRAIS processus python : un shell dont la ligne de commande contient ces noms
    # (une commande de surveillance, par exemple) ne compte pas
    local pid
    for pid in $(pgrep -f "train_unifolm_wla.py|action_server_wbc_msgpack_unitree"); do
        [[ "$(ps -o comm= -p "$pid" 2>/dev/null)" == python* ]] && return 1
    done
    local gpu ram
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    ram=$(free -g | awk '/^Mem:/ {print $7}')
    [[ $gpu =~ ^[0-9]+$ && $ram =~ ^[0-9]+$ ]] || return 1     # nvidia-smi absent ou illisible : pas libre
    (( gpu < 2000 && ram >= 18 ))
}

OWN=0
acquire() {  # verrou ATOMIQUE (noclobber) : deux files qui voient la machine libre ne démarrent pas ensemble
    while :; do
        wait_free
        if ( set -o noclobber; echo "$$ $(basename "$0")" > "$BUSY" ) 2>/dev/null; then OWN=1; return; fi
        say "verrou pris par une autre file, nouvelle attente"
    done
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
cleanup() { [[ -n "$SRV" ]] && kill $SRV 2>/dev/null; [[ $OWN == 1 ]] && rm -f "$BUSY"; }
trap cleanup EXIT

subset() {  # subset <n> : dataset des n premières démos (conversion gourmande en RAM : machine libre)
    local n=$1
    local out=playground/Datasets/g1d_sim_novares_n$n/sim_novares
    sed "s|data_path: \"g1d_sim_novares/\"|data_path: \"g1d_sim_novares_n$n/\"|" \
        $CFG/g1d_sim_novares.yaml > $CFG/g1d_sim_novares_n$n.yaml
    [[ -f "$out/meta/info.json" ]] && { say "sous-ensemble n$n déjà présent"; return 0; }
    say "conversion du sous-ensemble n$n"
    env -i HOME="$HOME" PATH=/usr/bin:/bin .venv/bin/python -m g1d_wla.convert_teleop \
        --raw-dir playground/sim_raw/sim_novares --out-dir "$out" --repo-id innov8/g1d_sim_novares_n$n \
        --episodes $(seq 0 $((n - 1))) --overwrite > "$LOGS/convert_n$n.log" 2>&1
}

train_eval() {  # train_eval <tag> <config>
    local tag=$1 cfg=$2
    local run=g1d_sim_novares_$tag free_gb
    free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
    if (( free_gb < 16 )); then say "ARRÊT : ${free_gb} Go libres, il en faut 16 pour $run"; exit 1; fi
    say "entraînement $run ($cfg)"
    rm -rf "playground/Checkpoints/$run"
    run_id=$run bash examples/unifolm_wla/train_files/run_finetune_g1d.sh \
        --datasets.vla_data.data_config_path $CFG/$cfg \
        --datasets.vla_data.per_device_batch_size 2 --datasets.vla_data.num_workers 4 \
        --trainer.gradient_accumulation_steps 1 --trainer.max_train_steps 3000 --trainer.num_warmup_steps 100 \
        --trainer.save_interval 1000000 --trainer.eval_interval 1000000 --trainer.logging_frequency 50 \
        > "$LOGS/train_$tag.log" 2>&1 || { say "ÉCHEC entraînement $run (voir $LOGS/train_$tag.log)"; return 1; }
    local ckpt=playground/Checkpoints/$run/final_model/model.safetensors
    say "évaluation $run"
    env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 .venv/bin/python -m model_server.action_server_wbc_msgpack_unitree \
        --ckpt_path "$ckpt" --unnorm_key UnifoLM_G1_Dex1 --port 8600 > "$LOGS/server_$tag.log" 2>&1 &
    SRV=$!
    local srv=$SRV
    until grep -q "server listening on" "$LOGS/server_$tag.log"; do
        kill -0 $srv 2>/dev/null || { say "ÉCHEC serveur $run"; return 1; }
        sleep 5
    done
    local mode
    for mode in "30 0" "10 20"; do
        set -- $mode
        MUJOCO_GL=egl $SIMPY sim/wla_client.py --scene novares --instruction "pick up the black part" \
            --head-view raw --episodes 30 --max-steps 300 --exec-steps $1 --rtc-prefix $2 --stop-on-success \
            --videos 2 --out sim/wla_out/novares_${tag}_e$1_p$2 2>&1 | grep -E "réussite|Error|Traceback" \
            | sed "s|^|    [$tag, exec $1, préfixe $2] |"
    done
    kill $srv
    SRV=""
    wait $srv 2>/dev/null
    say "fini $run"
}

say "file Novares : n50, n25, n10"
acquire     # attend une machine libre puis prend le verrou (atomique)
subset 25
subset 10
train_eval n50 g1d_sim_novares.yaml
train_eval n25 g1d_sim_novares_n25.yaml
train_eval n10 g1d_sim_novares_n10.yaml
say "file terminée"
