#!/usr/bin/env bash
set -euo pipefail

# Centralized sync helper to pull experiment scripts/reports from scattered repos.
# Update the source paths below if directory layout changes.

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC_BASE="/workspace/oujingfeng/project/think_with_generated_images"
MC="${SRC_BASE}/MathCanvas/BAGEL-Canvas"
QV="${SRC_BASE}/Qwen3-VL"
RV="${SRC_BASE}/ROVER"

sync_file() {
  local src="$1"
  local dest_dir="$2"
  if [[ -f "$src" ]]; then
    mkdir -p "$dest_dir"
    cp -p "$src" "$dest_dir/"
  else
    echo "warn: missing $src" >&2
  fi
}

# MathCanvas - rotation
sync_file "${MC}/mathcanvas_interleave_reasoner.py" "${ROOT}/mathcanvas/rotation/scripts"
sync_file "${MC}/scripts/inference/prepare_rotation_interleave_jsonl.py" "${ROOT}/mathcanvas/rotation/scripts"
sync_file "${MC}/scripts/inference/prepare_rotation_nofinal_jsonl.py" "${ROOT}/mathcanvas/rotation/scripts"
sync_file "${MC}/scripts/inference/infer_rotation_mathcanvas_ddp.py" "${ROOT}/mathcanvas/rotation/scripts"
sync_file "${MC}/scripts/inference/infer_rotation_mathcanvas_steps_ddp.py" "${ROOT}/mathcanvas/rotation/scripts"
sync_file "${MC}/scripts/inference/infer_rotation_mathcanvas_steps_nofinal_ddp.py" "${ROOT}/mathcanvas/rotation/scripts"
sync_file "${MC}/scripts/inference/run_rotation_mathcanvas_ddp.sh" "${ROOT}/mathcanvas/rotation/scripts"
sync_file "${MC}/scripts/inference/run_rotation_mathcanvas_steps_ddp.sh" "${ROOT}/mathcanvas/rotation/scripts"
sync_file "${MC}/scripts/inference/run_rotation_mathcanvas_steps_nofinal_ddp.sh" "${ROOT}/mathcanvas/rotation/scripts"
sync_file "${MC}/scripts/inference/run_rotation_mathcanvas_interleave_nofinal.sh" "${ROOT}/mathcanvas/rotation/scripts"
sync_file "${MC}/outputs/rotation_eval_summary.md" "${ROOT}/mathcanvas/rotation/reports"
sync_file "${MC}/outputs/3dviews_eval_summary.md" "${ROOT}/mathcanvas/3dviews/reports"
sync_file "${MC}/scripts/evaluation/eval_3dviews_outputs.py" "${ROOT}/mathcanvas/3dviews/scripts"

# MathCanvas - 3Dviews
sync_file "${MC}/scripts/inference/prepare_3dviews_interleave_jsonl.py" "${ROOT}/mathcanvas/3dviews/scripts"
sync_file "${MC}/scripts/inference/infer_3dviews_mathcanvas_ddp.py" "${ROOT}/mathcanvas/3dviews/scripts"
sync_file "${MC}/scripts/inference/infer_3dviews_mathcanvas_steps_ddp.py" "${ROOT}/mathcanvas/3dviews/scripts"
sync_file "${MC}/scripts/inference/run_3dviews_mathcanvas_ddp.sh" "${ROOT}/mathcanvas/3dviews/scripts"
sync_file "${MC}/scripts/inference/run_3dviews_mathcanvas_steps_ddp.sh" "${ROOT}/mathcanvas/3dviews/scripts"
sync_file "${MC}/scripts/inference/run_3dviews_mathcanvas_interleave_base.sh" "${ROOT}/mathcanvas/3dviews/scripts"
sync_file "${MC}/scripts/inference/run_3dviews_mathcanvas_interleave_steps.sh" "${ROOT}/mathcanvas/3dviews/scripts"

# MathCanvas - Hunyuan3D projection helper
sync_file "${MC}/scripts/inference/generate_rotation_hunyuan3d_projections.py" "${ROOT}/mathcanvas/hunyuan3d/scripts"

# Qwen3-VL - rotation API & offline
sync_file "${QV}/evaluation/rotation_api_infer.py" "${ROOT}/qwen3-vl/rotation_api/scripts"
sync_file "${QV}/evaluation/rotation_api_infer_nofinal_steps.py" "${ROOT}/qwen3-vl/rotation_api/scripts"
sync_file "${QV}/evaluation/rotation_api_infer_235b.py" "${ROOT}/qwen3-vl/rotation_api/scripts"
sync_file "${QV}/evaluation/run_rotation_api.sh" "${ROOT}/qwen3-vl/rotation_api/scripts"
sync_file "${QV}/evaluation/run_rotation_api_235b.sh" "${ROOT}/qwen3-vl/rotation_api/scripts"
sync_file "${QV}/evaluation/run_rotation_api_qwen3vl_plus_nofinal_steps.sh" "${ROOT}/qwen3-vl/rotation_api/scripts"
sync_file "${QV}/evaluation/eval_rotation_api_outputs.py" "${ROOT}/qwen3-vl/rotation_api/scripts"
sync_file "${QV}/evaluation/run_qwen3vl_rotation_official.py" "${ROOT}/qwen3-vl/rotation_official/scripts"
sync_file "${QV}/evaluation/run_qwen3vl_rotation_official.sh" "${ROOT}/qwen3-vl/rotation_official/scripts"
sync_file "${QV}/evaluation/eval_qwen3vl_rotation_outputs.py" "${ROOT}/qwen3-vl/rotation_official/scripts"
sync_file "${QV}/evaluation/run_qwen3vl_3dviews_official.py" "${ROOT}/qwen3-vl/rotation_official/scripts"
sync_file "${QV}/outputs/qwen3vl8b_rotation_eval_summary.md" "${ROOT}/qwen3-vl/reports"
sync_file "${QV}/outputs/qwen3vl8b_rotation_base_official_eval_summary.json" "${ROOT}/qwen3-vl/reports"
sync_file "${QV}/outputs/qwen3vl8b_rotation_nofinal_official_eval_summary.json" "${ROOT}/qwen3-vl/reports"
sync_file "${QV}/outputs/rotation_api_qwen3vl235b_base_eval_summary.json" "${ROOT}/qwen3-vl/reports"
sync_file "${QV}/outputs/rotation_api_eval_summary.md" "${ROOT}/qwen3-vl/reports"

# ROVER scripts/results
sync_file "${RV}/prepare_rover_for_mathcanvas.py" "${ROOT}/rover/scripts"
sync_file "${RV}/export_rover_generations.py" "${ROOT}/rover/scripts"
sync_file "${RV}/evaluate_rover.py" "${ROOT}/rover/scripts"
sync_file "${RV}/inspect_rover_dataset.py" "${ROOT}/rover/scripts"
sync_file "${RV}/rover_schema_sample.json" "${ROOT}/rover"

# Documentation from home directory
sync_file "/workspace/oujingfeng/plan.md" "${ROOT}/docs"
sync_file "/workspace/oujingfeng/work_summary.md" "${ROOT}/docs"

echo "Sync complete."
