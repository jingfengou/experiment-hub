#!/usr/bin/env bash
# Qwen3-VL-Plus API：旋转基准 “中间步骤、不含最终图像” 推理（stream + enable_thinking 可选）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export BASE_URL=${BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}
export API_KEY=${API_KEY:-${DASHSCOPE_API_KEY:-${QWEN_API_KEY:-${OPENAI_API_KEY:-}}}}

INPUT_JSON=${INPUT_JSON:-/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json}
DATA_ROOT=${DATA_ROOT:-/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/oujingfeng/project/think_with_generated_images/Qwen3-VL/outputs/rotation_api_qwen3vl_plus_nofinal_steps}
MODEL=${MODEL:-qwen3-vl-plus}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
MAX_TOKENS=${MAX_TOKENS:-512}
SLEEP=${SLEEP:-0.2}
ENABLE_THINKING=${ENABLE_THINKING:-false}
THINKING_BUDGET=${THINKING_BUDGET:-8192}

if [ -z "$API_KEY" ]; then
  echo "请设置 API_KEY / DASHSCOPE_API_KEY / QWEN_API_KEY 环境变量"
  exit 1
fi

python evaluation/rotation_api_infer_nofinal_steps.py \
  --input_json "$INPUT_JSON" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --base_url "$BASE_URL" \
  --api_key "$API_KEY" \
  --max_samples "$MAX_SAMPLES" \
  --max_tokens "$MAX_TOKENS" \
  --sleep "$SLEEP" \
  ${ENABLE_THINKING:+--enable_thinking} \
  --thinking_budget "$THINKING_BUDGET"
