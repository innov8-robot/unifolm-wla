#!/usr/bin/env bash
# Fine-tuning WLA sur des démos RÉELLES du G1-D (depuis le modèle Base), avec le verrou GPU partagé.
#   CFG=g1d_novares_box.yaml RUN=g1d_novares_box STEPS=15000 \
#       setsid nohup bash sim/experiments/train_real.sh > playground/queue_logs/train_novares_box.log 2>&1 &
set -uo pipefail
cd /home/thomas/Documents/project/manip/unifolm-wla
ROOT=$PWD
BUSY=$ROOT/playground/.gpu_busy
CFG_DIR=unifolm_wla/dataloader/multi_source_dataset/configs
: "${CFG:?CFG=<config de données>}" "${RUN:?RUN=<nom du run>}"
STEPS=${STEPS:-15000}
say() { echo "[$(date '+%F %T')] $*"; }
OWN=0
cleanup() { [[ $OWN == 1 ]] && rm -f "$BUSY"; }
trap cleanup EXIT
until ( set -o noclobber; echo "$$ train_real $RUN" > "$BUSY" ) 2>/dev/null; do
    say "GPU occupé ($(cat "$BUSY" 2>/dev/null)), attente"; sleep 60
done
OWN=1
free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
(( free_gb >= 13 )) || { say "ARRÊT : ${free_gb} Go libres, il en faut 13"; exit 1; }
say "entraînement $RUN ($CFG, $STEPS pas)"
rm -rf "playground/Checkpoints/$RUN" playground/cache/arrow_cache
run_id=$RUN bash examples/unifolm_wla/train_files/run_finetune_g1d.sh \
    --datasets.vla_data.data_config_path $CFG_DIR/$CFG \
    --datasets.vla_data.per_device_batch_size 2 --datasets.vla_data.num_workers 4 \
    --trainer.gradient_accumulation_steps 1 --trainer.max_train_steps $STEPS --trainer.num_warmup_steps 200 \
    --trainer.save_interval 1000000 --trainer.eval_interval 1000000 --trainer.logging_frequency 50 \
    && say "entraînement terminé : playground/Checkpoints/$RUN/final_model/model.safetensors" \
    || { say "ÉCHEC entraînement"; exit 1; }
