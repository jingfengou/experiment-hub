#!/usr/bin/env python3
"""
评估 run_qwen3vl_rotation_official.py 生成的 rotation 基准输出，计算精度并生成记录文件。

用法示例：
python evaluation/eval_qwen3vl_rotation_outputs.py \
  --data_json /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json \
  --outputs_dir /workspace/oujingfeng/project/think_with_generated_images/Qwen3-VL/outputs/qwen3vl8b_rotation_base_official
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def load_data(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data


def extract_answer(text: str) -> Optional[str]:
    """从模型输出里抽取 A/B/C/D，失败返回 None。"""
    lower = text.lower()

    patterns = [
        r"<answer>\s*([abcd])\s*</answer>",
        r"final answer\s*(is|:)?\s*([abcd])\b",
        r"answer\s*(is|:)?\s*([abcd])\b",
        r"so (the )?answer\s*(is|:)?\s*([abcd])\b",
        # r"option\s*([abcd])\b",
        # r"choice\s*([abcd])\b",
        # r"choose\s*([abcd])\b",
        # r"\(\s*([abcd])\s*\)\s*$",
        # r"\b([abcd])\)\s*$",
    ]

    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            # 取最后一个捕获组以兼容不同 pattern
            return m.group(m.lastindex).upper() if m.lastindex else None

    # 尝试从末尾一行的单独大写 A-D 中解析
    tail_line = text.strip().splitlines()[-1] if text.strip() else ""
    m = re.search(r"\b([ABCD])\b", tail_line)
    if m:
        return m.group(1)
    return None


def evaluate(args):
    data = load_data(Path(args.data_json))
    outputs_dir = Path(args.outputs_dir)
    record_path = Path(args.record_file) if args.record_file else outputs_dir.with_name(outputs_dir.name + "_eval.jsonl")
    summary_path = Path(args.summary_file) if args.summary_file else outputs_dir.with_name(outputs_dir.name + "_eval_summary.json")

    files = sorted(outputs_dir.glob("sample*.txt"))
    total = len(files)
    if total == 0:
        raise FileNotFoundError(f"未找到输出文件: {outputs_dir}/sample*.txt")

    correct = 0
    missing_pred = 0
    records = []

    for f in files:
        m = re.search(r"sample(\d+)\.txt", f.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx >= len(data):
            # 超出数据范围的样本跳过
            continue

        gt = (data[idx].get("Answer") or "").strip().upper()
        pred_raw = f.read_text(encoding="utf-8", errors="ignore")
        pred = extract_answer(pred_raw)
        if not pred:
            missing_pred += 1
        if pred and gt and pred == gt:
            correct += 1

        records.append({
            "sample_idx": idx,
            "file": f.name,
            "gt": gt,
            "pred": pred,
            "correct": bool(pred and gt and pred == gt),
        })

    record_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")

    summary = {
        "outputs_dir": str(outputs_dir),
        "data_json": str(args.data_json),
        "evaluated": len(records),
        "correct": correct,
        "accuracy": correct / len(records) if records else 0.0,
        "missing_prediction": missing_pred,
        "total_files": total,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", required=True, help="rotation 数据集 JSON 路径")
    ap.add_argument("--outputs_dir", required=True, help="模型输出目录（包含 samplexxxxx.txt）")
    ap.add_argument("--record_file", default="", help="评估记录保存路径，默认 outputs_dir_eval.jsonl")
    ap.add_argument("--summary_file", default="", help="摘要保存路径，默认 outputs_dir_eval_summary.json")
    args = ap.parse_args()
    evaluate(args)
