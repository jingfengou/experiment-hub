#!/usr/bin/env bash
# Run official HF Qwen3-VL Instruct on rotation benchmark (base & nofinal, first 1000).

set -euo pipefail

# 解码/采样参数（可覆盖）
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export greedy=${greedy:-false}
export top_p=${top_p:-0.8}
export top_k=${top_k:-20}
export temperature=${temperature:-0.3}
export repetition_penalty=${repetition_penalty:-1.0}
export presence_penalty=${presence_penalty:-1.5}
export out_seq_length=${out_seq_length:-512}
GREEDY_FLAG=""
if [[ "${greedy,,}" == "true" ]]; then
  GREEDY_FLAG="--greedy"
fi

MODEL_ID=${MODEL_ID:-"Qwen/Qwen3-VL-8B-Instruct"}
MODEL_ID_32B=${MODEL_ID_32B:-"Qwen/Qwen3-VL-32B-Instruct"}
DATA_JSON=${DATA_JSON:-/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json}
DATA_ROOT=${DATA_ROOT:-/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
OUT_BASE=${OUT_BASE:-outputs/qwen3vl8b_rotation_base_official}
OUT_NOFINAL=${OUT_NOFINAL:-outputs/qwen3vl8b_rotation_nofinal_official}

echo "MODEL_ID    : $MODEL_ID"
echo "MODEL_ID_32B: $MODEL_ID_32B"
echo "DATA_JSON   : $DATA_JSON"
echo "DATA_ROOT   : $DATA_ROOT"
echo "MAX_SAMPLES : $MAX_SAMPLES"
echo "OUT_BASE    : $OUT_BASE"
echo "OUT_NOFINAL : $OUT_NOFINAL"
echo "greedy=${greedy} temperature=${temperature} top_p=${top_p} top_k=${top_k} rep_penalty=${repetition_penalty} presence_penalty=${presence_penalty} max_new_tokens=${out_seq_length}"

python evaluation/run_qwen3vl_rotation_official.py \
  --model_id "$MODEL_ID" \
  --mode base \
  --max_samples "$MAX_SAMPLES" \
  --data_json "$DATA_JSON" \
  --data_root "$DATA_ROOT" \
  --out_dir "$OUT_BASE" \
  --temperature "$temperature" \
  --max_new_tokens "$out_seq_length" \
  --top_p "$top_p" \
  --top_k "$top_k" \
  --repetition_penalty "$repetition_penalty" \
  --presence_penalty "$presence_penalty" \
  $GREEDY_FLAG

python evaluation/run_qwen3vl_rotation_official.py \
  --model_id "$MODEL_ID" \
  --mode nofinal \
  --max_samples "$MAX_SAMPLES" \
  --data_json "$DATA_JSON" \
  --data_root "$DATA_ROOT" \
  --out_dir "$OUT_NOFINAL" \
  --temperature "$temperature" \
  --max_new_tokens "$out_seq_length" \
  --top_p "$top_p" \
  --top_k "$top_k" \
  --repetition_penalty "$repetition_penalty" \
  --presence_penalty "$presence_penalty" \
  $GREEDY_FLAG

python evaluation/run_qwen3vl_rotation_official.py \
  --model_id "$MODEL_ID_32B" \
  --mode base \
  --max_samples "$MAX_SAMPLES" \
  --data_json "$DATA_JSON" \
  --data_root "$DATA_ROOT" \
  --out_dir "${OUT_BASE}_32b" \
  --temperature "$temperature" \
  --max_new_tokens "$out_seq_length" \
  --top_p "$top_p" \
  --top_k "$top_k" \
  --repetition_penalty "$repetition_penalty" \
  --presence_penalty "$presence_penalty" \
  $GREEDY_FLAG

python evaluation/run_qwen3vl_rotation_official.py \
  --model_id "$MODEL_ID_32B" \
  --mode nofinal \
  --max_samples "$MAX_SAMPLES" \
  --data_json "$DATA_JSON" \
  --data_root "$DATA_ROOT" \
  --out_dir "${OUT_NOFINAL}_32b" \
  --temperature "$temperature" \
  --max_new_tokens "$out_seq_length" \
  --top_p "$top_p" \
  --top_k "$top_k" \
  --repetition_penalty "$repetition_penalty" \
  --presence_penalty "$presence_penalty" \
  $GREEDY_FLAG

echo "Done. Outputs at $OUT_BASE, $OUT_NOFINAL, ${OUT_BASE}_32b, ${OUT_NOFINAL}_32b"
