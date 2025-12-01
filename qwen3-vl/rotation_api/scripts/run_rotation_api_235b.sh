#!/usr/bin/env bash
# Run Qwen3-VL-235B API on rotation benchmark (base or nofinal).

set -euo pipefail

QWEN_API_KEY=${QWEN_API_KEY:-sk-3b5a378c695644b5a23463b3d34c6768}
MODE=${MODE:-base}  # base | nofinal
MODEL=${MODEL:-qwen3-vl-plus}
INPUT_JSON=${INPUT_JSON:-/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json}
DATA_ROOT=${DATA_ROOT:-/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/rotation_api_qwen3vl235b_${MODE}}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
MAX_TOKENS=${MAX_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-0.7}
SLEEP=${SLEEP:-0.0}
BASE_URL=${BASE_URL:-${BASE_URL:-}}

echo "MODE        : $MODE"
echo "MODEL       : $MODEL"
echo "INPUT_JSON  : $INPUT_JSON"
echo "DATA_ROOT   : $DATA_ROOT"
echo "OUTPUT_DIR  : $OUTPUT_DIR"
echo "MAX_SAMPLES : $MAX_SAMPLES"
echo "MAX_TOKENS  : $MAX_TOKENS"
echo "TEMPERATURE : $TEMPERATURE"
echo "SLEEP       : $SLEEP"
echo "BASE_URL    : ${BASE_URL:-<default>}"

python evaluation/rotation_api_infer_235b.py \
  --input_json "$INPUT_JSON" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --mode "$MODE" \
  --api_key "$QWEN_API_KEY" \
  --max_samples "$MAX_SAMPLES" \
  --max_tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --sleep "$SLEEP" \
  ${BASE_URL:+--base_url "$BASE_URL"}

echo "Done. Outputs at $OUTPUT_DIR"
