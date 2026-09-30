#!/usr/bin/env bash
# Empilement Novares en sim : 25 démos expertes (réussites seulement) -> fine-tuning -> évaluation.
#   N=100 setsid nohup bash sim/experiments/queue_stack.sh > playground/queue_logs/stack_n100.log 2>&1 &
# Attend la fin de l'enregistrement des démos (playground/sim_raw/stack_n25/summary.json), puis une
# machine libre (mêmes garde-fous que queue_novares.sh : playground/.gpu_busy, GPU, RAM).
set -uo pipefail
cd /home/thomas/Documents/project/manip/unifolm-wla
ROOT=$PWD
LOGS=$ROOT/playground/queue_logs
mkdir -p "$LOGS"
SIMPY=$HOME/miniconda3/envs/unitree_lerobot/bin/python
VENV=(env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8)
BUSY=$ROOT/playground/.gpu_busy
CFG=unifolm_wla/dataloader/multi_source_dataset/configs
N=${N:-25}
RAW=playground/sim_raw/stack_n$N
RUN=g1d_sim_stack_n$N
say() { echo "[$(date '+%F %T')] $*"; }

machine_free() {
    [[ -e "$BUSY" ]] && return 1
    local pid
    for pid in $(pgrep -f "train_unifolm_wla.py|action_server_wbc_msgpack_unitree"); do
        [[ "$(ps -o comm= -p "$pid" 2>/dev/null)" == python* ]] && return 1
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
cleanup() { [[ -n "$SRV" ]] && kill $SRV 2>/dev/null; rm -f "$BUSY"; }
trap cleanup EXIT

say "file empilement : attente des $N démos"
until [[ -f $RAW/summary.json ]]; do sleep 30; done
say "démos prêtes : $(python3 -c "import json;d=json.load(open('$RAW/summary.json'));print(d['kept'],'gardées sur',d['tried'],'essais')")"
wait_free
touch "$BUSY"

[[ $N == 25 ]] || sed "s|g1d_sim_stack_n25/|$RUN/|; s|25 démos|$N démos|" $CFG/g1d_sim_stack_n25.yaml > $CFG/$RUN.yaml
say "conversion"
"${VENV[@]}" .venv/bin/python -m g1d_wla.convert_teleop --raw-dir $RAW \
    --out-dir playground/Datasets/$RUN/sim_stack --repo-id innov8/$RUN --overwrite \
    > "$LOGS/stack_n${N}_convert.log" 2>&1 || { say "ÉCHEC conversion"; exit 1; }

free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
(( free_gb >= 13 )) || { say "ARRÊT : ${free_gb} Go libres, il en faut 13"; exit 1; }
say "entraînement $RUN"
rm -rf "playground/Checkpoints/$RUN" playground/cache/arrow_cache
run_id=$RUN bash examples/unifolm_wla/train_files/run_finetune_g1d.sh \
    --datasets.vla_data.data_config_path $CFG/$RUN.yaml \
    --datasets.vla_data.per_device_batch_size 2 --datasets.vla_data.num_workers 4 \
    --trainer.gradient_accumulation_steps 1 --trainer.max_train_steps 3000 --trainer.num_warmup_steps 100 \
    --trainer.save_interval 1000000 --trainer.eval_interval 1000000 --trainer.logging_frequency 50 \
    > "$LOGS/stack_n${N}_train.log" 2>&1 || { say "ÉCHEC entraînement"; exit 1; }

say "évaluation (expert : ~270 pas jusqu'au lâcher)"
"${VENV[@]}" .venv/bin/python -m model_server.action_server_wbc_msgpack_unitree \
    --ckpt_path playground/Checkpoints/$RUN/final_model/model.safetensors --unnorm_key UnifoLM_G1_Dex1 \
    --port 8600 > "$LOGS/stack_n${N}_server.log" 2>&1 &
SRV=$!
until grep -q "server listening on" "$LOGS/stack_n${N}_server.log"; do
    kill -0 $SRV 2>/dev/null || { say "ÉCHEC serveur"; exit 1; }
    sleep 5
done
MUJOCO_GL=egl $SIMPY sim/wla_client.py --scene stack --instruction "stack the black part on the other one" \
    --head-view raw --episodes 30 --max-steps 450 --exec-steps 30 --stop-on-success --videos 4 --seed 7 \
    --out sim/wla_out/stack_n$N 2>&1 | grep -E "réussite|Error|Traceback" | sed "s|^|    [stack n$N] |"
say "file empilement terminée"
