#!/usr/bin/env bash
# Fine-tuning de UnifoLM-WLA-1.0-Base sur nos démos G1-D (VLM gelé, projecteur + DiT entraînés).
#   bash examples/unifolm_wla/train_files/run_finetune_g1d.sh [surcharges --clé valeur ...]
# ex. essai court : ... --trainer.max_train_steps 5 --trainer.save_interval 100000
# Par défaut, l'optimiseur est déporté en mémoire vive : sans ça, ni cette recette ni la recette
# officielle ne tiennent sur 24 Go (mesuré sur RTX 5090 Laptop). DS_ACCEL_CONFIG pour changer.
set -euo pipefail
export WANDB_MODE=${WANDB_MODE:-disabled}
base_model_dir=${base_model_dir:-playground/Pretrained_models/UnifoLM-WLA-1.0-Base}
run_root_dir=${run_root_dir:-playground/Checkpoints}
run_id=${run_id:-g1d_finetune_frozen_vlm}
mkdir -p "${run_root_dir}/${run_id}"
cp "$0" "${run_root_dir}/${run_id}/"
# DeepSpeed compile son Adam CPU au premier lancement. Le CUDA système (13, sans curand) ne
# correspond pas à torch (CUDA 12.8) : on lui présente les bibliothèques CUDA 12 de l'env.
SHIM="$PWD/playground/cuda12_shim"
if [[ ! -e "$SHIM/lib64/libcurand.so" ]]; then
    NV="$PWD/.venv/lib/python3.12/site-packages/nvidia"
    mkdir -p "$SHIM/lib64" "$SHIM/bin"
    ln -sf "$NV/curand/lib/libcurand.so.10" "$SHIM/lib64/libcurand.so"
    ln -sf "$NV/cuda_runtime/lib/libcudart.so.12" "$SHIM/lib64/libcudart.so"
    ln -sfn /usr/local/cuda/include "$SHIM/include"
    ln -sf /usr/local/cuda/bin/nvcc "$SHIM/bin/nvcc"
fi
# hors conda (voir MYREADME.md §3)
env -i HOME="$HOME" PATH="$PWD/.venv/bin:/usr/bin:/bin:/usr/local/cuda/bin" LANG=C.UTF-8 WANDB_MODE="$WANDB_MODE" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DS_SKIP_CUDA_CHECK=1 CUDA_HOME="$SHIM" \
    LD_LIBRARY_PATH="$SHIM/lib64" \
.venv/bin/accelerate launch \
  --config_file "${DS_ACCEL_CONFIG:-unifolm_wla/config/deepseeds/deepspeed_zero2_offload.yaml}" \
  --num_processes "${NUM_PROCESSES:-1}" \
  unifolm_wla/training/train_unifolm_wla.py \
  --config_yaml unifolm_wla/config/training/g1d_finetune_frozen_vlm.yaml \
  --framework.qwenvl.base_vlm "${base_model_dir}/tokenizer" \
  --trainer.pretrained_checkpoint "${base_model_dir}/checkpoints/model.safetensors" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" "$@"
