#!/bin/bash

# Qwen3-VL 云 API 推理旋转基准
# 需要环境变量：QWEN_API_KEY（或 OPENAI_API_KEY）
# 可选：BASE_URL，例如 https://dashscope.aliyuncs.com/compatible-mode/v1

INPUT_JSON=/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json
DATA_ROOT=/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data

MODEL=${MODEL:-qwen-vl-8b-instruct}   # 可改为 qwen-vl-32b-instruct
OUTPUT_DIR=${OUTPUT_DIR:-outputs/rotation_api_${MODEL}}
MAX_SAMPLES=${MAX_SAMPLES:-0}        # 0 表示全部
SLEEP=${SLEEP:-0.0}                  # 每次请求的间隔秒

echo "MODEL       : $MODEL"
echo "INPUT_JSON  : $INPUT_JSON"
echo "DATA_ROOT   : $DATA_ROOT"
echo "OUTPUT_DIR  : $OUTPUT_DIR"
echo "MAX_SAMPLES : $MAX_SAMPLES"
echo "BASE_URL    : ${BASE_URL:-<default>}"

python rotation_api_infer.py \
  --input_json "$INPUT_JSON" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --max_samples "$MAX_SAMPLES" \
  --sleep "$SLEEP" \
  ${BASE_URL:+--base_url "$BASE_URL"} \
  ${QWEN_API_KEY:+--api_key "$QWEN_API_KEY"}
