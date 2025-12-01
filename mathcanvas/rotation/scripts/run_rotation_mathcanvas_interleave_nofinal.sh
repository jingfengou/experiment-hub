#!/usr/bin/env bash
# Use mathcanvas_interleave_reasoner.py with UniHandler on rotation data (final step image removed).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Paths
DATA_JSON=${DATA_JSON:-data_handlers/rotation_nofinal/rotation_nofinal.jsonl}
DATA_ROOT=${DATA_ROOT:-data_handlers/rotation_nofinal}
MODEL_PATH=${MODEL_PATH:-/workspace/oujingfeng/modelckpt/BAGEL-Canvas}
CKPT_DIR=${CKPT_DIR:-/workspace/oujingfeng/modelckpt/BAGEL-Canvas}
CKPT_FILE=${CKPT_FILE:-model.safetensors}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/rotation_mathcanvas_interleave_nofinal}

# GPUs
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L | wc -l)
else
  NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

torchrun --nproc_per_node=$NUM_GPUS --master_port=23459 \
  mathcanvas_interleave_reasoner.py \
  --dataset_type uni \
  --dataset_name rotation_nofinal \
  --input_path "$DATA_JSON" \
  --checkpoint_dir "$CKPT_DIR" \
  --checkpoint_file "$CKPT_FILE" \
  --model_path "$MODEL_PATH" \
  --image_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --add_timestamp false \
  --run_description default \
  --max_iterations 10 \
  --skip_completed \
  --do_sample true \
  --text_temperature 0.3 \
  --cfg_text_scale 4.0 \
  --cfg_img_scale 2.0 \
  --cfg_interval 0.0 1.0 \
  --timestep_shift 3.0 \
  --num_timesteps 50 \
  --cfg_renorm_min 0.0 \
  --cfg_renorm_type text_channel \
  --enable_taylorseer false
