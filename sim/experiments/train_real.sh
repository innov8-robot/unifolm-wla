#!/usr/bin/env bash
# Fine-tuning WLA sur des démos RÉELLES du G1-D (depuis le modèle Base), avec le verrou GPU partagé.
#   CFG=g1d_novares_box.yaml RUN=g1d_novares_box STEPS=15000 \
#       setsid nohup bash sim/experiments/train_real.sh > playground/queue_logs/train_novares_box.log 2>&1 &
# Arrêt propre : touch playground/Checkpoints/<RUN>/STOP (sauvegarde puis arrêt) ; reprise : RESUME=1 + mêmes CFG/RUN/STEPS.
set -uo pipefail
cd /home/thomas/Documents/project/manip/unifolm-wla
ROOT=$PWD
BUSY=$ROOT/playground/.gpu_busy
CFG_DIR=unifolm_wla/dataloader/multi_source_dataset/configs
: "${CFG:?CFG=<config de données>}" "${RUN:?RUN=<nom du run>}"
STEPS=${STEPS:-15000}
SAVE_EVERY=${SAVE_EVERY:-2500}      # sauvegarde reprenable tous les N pas (seule la dernière est gardée)
RESUME=${RESUME:-0}                 # RESUME=1 : reprendre au dernier checkpoints/steps_N du run
INIT=${INIT:-}                      # INIT=<model.safetensors> : partir de ce modèle au lieu du modèle Base
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
if [[ $RESUME == 1 ]]; then
    ls playground/Checkpoints/$RUN/checkpoints/steps_* >/dev/null 2>&1 || { say "ARRÊT : rien à reprendre dans $RUN"; exit 1; }
    EXTRA=(--trainer.is_resume true)
    say "reprise de $RUN au dernier checkpoint : $(ls playground/Checkpoints/$RUN/checkpoints/ | grep steps_ | tail -1)"
else
    rm -rf "playground/Checkpoints/$RUN" playground/cache/arrow_cache
    EXTRA=()
    if [[ -n $INIT ]]; then
        [[ -f $INIT ]] || { say "ARRÊT : INIT introuvable ($INIT)"; exit 1; }
        EXTRA=(--trainer.pretrained_checkpoint "$INIT")
        say "départ depuis $INIT"
    fi
fi
run_id=$RUN bash examples/unifolm_wla/train_files/run_finetune_g1d.sh \
    --datasets.vla_data.data_config_path $CFG_DIR/$CFG \
    --datasets.vla_data.per_device_batch_size 2 --datasets.vla_data.num_workers 4 \
    --trainer.gradient_accumulation_steps 1 --trainer.max_train_steps $STEPS --trainer.num_warmup_steps 200 \
    --trainer.save_interval $SAVE_EVERY --trainer.keep_last_checkpoints 1 \
    --trainer.eval_interval 1000000 --trainer.logging_frequency 50 "${EXTRA[@]}" \
    && say "entraînement terminé : playground/Checkpoints/$RUN/final_model/model.safetensors" \
    || { say "ÉCHEC entraînement"; exit 1; }
