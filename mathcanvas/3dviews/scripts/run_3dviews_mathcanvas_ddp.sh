#!/usr/bin/env bash
# 非图文交替：纯理解（无步骤提示）DDP 推理

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p logs outputs

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

MODEL_DIR=${MODEL_DIR:-/workspace/oujingfeng/modelckpt/BAGEL-Canvas}
CKPT_DIR=${CKPT_DIR:-/workspace/oujingfeng/modelckpt/BAGEL-Canvas}
CKPT_FILE=${CKPT_FILE:-model.safetensors}
DATA_JSON=${DATA_JSON:-/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/3Dviews/thinking_chain_dataset/dataset.json}
DATA_ROOT=${DATA_ROOT:-/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/3Dviews/thinking_chain_dataset}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/3dviews_mathcanvas_ddp}
MAX_SAMPLES=${MAX_SAMPLES:-0}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L | wc -l)
else
  NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

torchrun --nproc_per_node=$NUM_GPUS --master_port=23457 \
  scripts/inference/infer_3dviews_mathcanvas_ddp.py \
  --model-dir "$MODEL_DIR" \
  --ckpt-dir "$CKPT_DIR" \
  --ckpt-file "$CKPT_FILE" \
  --dataset-file "$DATA_JSON" \
  --dataset-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --max-samples "$MAX_SAMPLES" \
  --sample-fraction 1.0 \
  --sample-region head \
  --text-temperature 0.3 | tee logs/3dviews_mathcanvas_ddp.log
