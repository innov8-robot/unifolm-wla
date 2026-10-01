#!/usr/bin/env bash
# Empilement en sim, une itération « à la DAgger » : la politique joue, l'opérateur simulé (expert
# rejoué depuis l'état courant) reprend la main quand elle dévie ; on réentraîne sur démos + corrections.
#   setsid nohup bash sim/experiments/queue_dagger.sh > playground/queue_logs/dagger_r1.log 2>&1 &
#
#  1. rollouts de la politique n25_12k (100 épisodes), reprise si : pièce poussée de 3 cm, saisie
#     ratée (pince fermée à fond 15 pas), ou pas de réussite au pas 330 ;
#  2. corrections : réussites de la politique en entier, reprises réussies depuis 10 pas avant ;
#  3. fine-tuning DEPUIS n25_12k sur démos n25 + corrections (4000 pas) ;
#  4. évaluation 30 épisodes + boucle ouverte.
set -uo pipefail
cd /home/thomas/Documents/project/manip/unifolm-wla
ROOT=$PWD
LOGS=$ROOT/playground/queue_logs
mkdir -p "$LOGS"
SIMPY=$HOME/miniconda3/envs/unitree_lerobot/bin/python
VENV=(env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8)
BUSY=$ROOT/playground/.gpu_busy
CFG=unifolm_wla/dataloader/multi_source_dataset/configs
POLICY=g1d_sim_stack_n25_12k
RUN=g1d_stack_dagger_r1
ROLL=playground/sim_raw/stack_dagger_r1
INSTR="stack the black part on the other one"
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
serve() {  # serve <ckpt> <tag>
    "${VENV[@]}" .venv/bin/python -m model_server.action_server_wbc_msgpack_unitree --ckpt_path "$1" \
        --unnorm_key UnifoLM_G1_Dex1 --port 8600 > "$LOGS/dagger_server_$2.log" 2>&1 &
    SRV=$!
    until grep -q "server listening on" "$LOGS/dagger_server_$2.log"; do
        kill -0 $SRV 2>/dev/null || { say "ÉCHEC serveur $2"; exit 1; }
        sleep 5
    done
}
unserve() { [[ -n "$SRV" ]] && kill $SRV && wait $SRV 2>/dev/null; SRV=""; }
cleanup() { unserve; rm -f "$BUSY"; }
trap cleanup EXIT

say "itération DAgger 1 (empilement)"
wait_free
touch "$BUSY"

say "1. rollouts de $POLICY avec opérateur simulé"
serve playground/Checkpoints/$POLICY/final_model/model.safetensors policy
MUJOCO_GL=egl $SIMPY sim/recap_rollouts.py --task stack --instruction "$INSTR" --episodes 100 --seed 5000 \
    --max-steps 700 --takeover-step 330 --miss-steps 15 --knock-cm 3 --out $ROLL --overwrite \
    > "$LOGS/dagger_rollouts.log" 2>&1 || { say "ÉCHEC rollouts"; exit 1; }
tail -1 "$LOGS/dagger_rollouts.log" | sed 's|^|    |'
unserve

say "2. corrections"
$SIMPY sim/dagger_corrections.py --rollouts $ROLL --out ${ROLL}_corr --overwrite | sed 's|^|    |'
cp $ROLL/summary.json ${ROLL}_corr/rollouts_summary.json
rm -rf $ROLL      # les corrections sont des liens durs : seuls les épisodes ratés libèrent de la place

say "3. conversion et fine-tuning depuis $POLICY"
for part in "playground/sim_raw/stack_n25 demos" "${ROLL}_corr corrections"; do
    set -- $part
    "${VENV[@]}" .venv/bin/python -m g1d_wla.convert_teleop --raw-dir $1 --out-dir playground/Datasets/$RUN/$2 \
        --repo-id innov8/${RUN}_$2 --advantage off --overwrite > "$LOGS/dagger_convert_$2.log" 2>&1 \
        || { say "ÉCHEC conversion $2"; exit 1; }
done
sed "s|g1d_sim_stack_n25/|$RUN/|" $CFG/g1d_sim_stack_n25.yaml > $CFG/$RUN.yaml
free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
(( free_gb >= 13 )) || { say "ARRÊT : ${free_gb} Go libres, il en faut 13"; exit 1; }
rm -rf "playground/Checkpoints/$RUN" playground/cache/arrow_cache
run_id=$RUN bash examples/unifolm_wla/train_files/run_finetune_g1d.sh \
    --datasets.vla_data.data_config_path $CFG/$RUN.yaml \
    --datasets.vla_data.per_device_batch_size 2 --datasets.vla_data.num_workers 4 \
    --trainer.gradient_accumulation_steps 1 --trainer.max_train_steps 4000 --trainer.num_warmup_steps 100 \
    --trainer.save_interval 1000000 --trainer.eval_interval 1000000 --trainer.logging_frequency 50 \
    --trainer.pretrained_checkpoint playground/Checkpoints/$POLICY/final_model/model.safetensors \
    > "$LOGS/dagger_train.log" 2>&1 || { say "ÉCHEC entraînement"; exit 1; }
grep -m1 "Loaded pretrained checkpoint" "$LOGS/dagger_train.log" | sed 's|.*Loaded|    Loaded|'

say "4. évaluation"
serve playground/Checkpoints/$RUN/final_model/model.safetensors eval
MUJOCO_GL=egl $SIMPY sim/wla_client.py --scene stack --instruction "$INSTR" --head-view raw --episodes 30 \
    --max-steps 450 --exec-steps 30 --stop-on-success --videos 4 --seed 7 --out sim/wla_out/stack_dagger_r1 \
    2>&1 | grep -E "réussite|Error|Traceback" | sed "s|^|    [$RUN] |"
MUJOCO_GL=egl $SIMPY sim/open_loop_check.py --dataset playground/Datasets/g1d_sim_stack_n25/sim_stack \
    --raw playground/sim_raw/stack_n25 --episodes 0 1 2 --instruction "$INSTR" 2>&1 | grep -E "BILAN|Error" \
    | sed "s|^|    [$RUN, boucle ouverte] |"
unserve
say "itération DAgger 1 terminée"
