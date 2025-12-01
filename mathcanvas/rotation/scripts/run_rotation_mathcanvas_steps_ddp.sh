#!/usr/bin/env bash
# Interleave reasoner for rotation with step hints（包含所有步骤图）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

MODEL_PATH=${MODEL_PATH:-/workspace/oujingfeng/modelckpt/BAGEL-Canvas}
CKPT_DIR=${CKPT_DIR:-/workspace/oujingfeng/modelckpt/BAGEL-Canvas}
CKPT_FILE=${CKPT_FILE:-model.safetensors}
DATA_JSON=${DATA_JSON:-data_handlers/rotation_interleave/steps/rotation_steps.jsonl}
DATA_ROOT=${DATA_ROOT:-data_handlers/rotation_interleave/steps}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/rotation_mathcanvas_interleave_steps}
MAX_SAMPLES=${MAX_SAMPLES:-1000}

if [ ! -f "$DATA_JSON" ]; then
  echo "Preparing rotation steps jsonl..."
  python scripts/inference/prepare_rotation_interleave_jsonl.py \
    --dataset-json /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json \
    --dataset-root /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset \
    --output-root data_handlers/rotation_interleave \
    --mode steps \
    --max-samples "$MAX_SAMPLES"
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L | wc -l)
else
  NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

torchrun --nproc_per_node=$NUM_GPUS --master_port=23457 \
  mathcanvas_interleave_reasoner.py \
  --dataset_type uni \
  --dataset_name rotation_steps \
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
