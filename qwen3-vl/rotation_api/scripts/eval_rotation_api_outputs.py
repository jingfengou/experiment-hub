#!/usr/bin/env python3
"""
Evaluate rotation API outputs saved by rotation_api_infer*.py scripts.

Assumptions:
- Each sample is a folder sample_XXXX containing response.json with an "answer" field.
- Ground truth answers are in the dataset JSON (Answer: A/B/C/D).
"""
import argparse
import json
import re
from pathlib import Path
from typing import List, Optional


def extract_answer(text: str) -> Optional[str]:
    # Prefer <answer>...</answer>
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        inside = m.group(1)
        m2 = re.search(r"\b([A-D])\b", inside, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).upper()
        # Allow non-word boundaries (e.g., 选项C)
        m3 = re.search(r"([A-D])", inside, flags=re.IGNORECASE)
        if m3:
            return m3.group(1).upper()

    # Look for phrases like "选项C" / "Option C" / "答案C"
    m = re.search(r"(?:选项|option|答案|answer)[：: ]*([A-D])", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Fallback: last A-D anywhere
    matches = re.findall(r"([A-D])", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return None


def load_answers(path: Path) -> List[str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    answers = []
    for item in data:
        ans = (item.get("Answer") or "").strip().upper()
        answers.append(ans[0] if ans else "")
    return answers


def main():
    parser = argparse.ArgumentParser(description="Evaluate rotation API outputs.")
    parser.add_argument("--output-dir", required=True, help="Directory containing sample_XXXX folders.")
    parser.add_argument(
        "--dataset-json",
        default="/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json",
        help="Rotation dataset JSON with Answer field.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    answers = load_answers(Path(args.dataset_json))

    sample_dirs = sorted([p for p in out_dir.iterdir() if p.is_dir()])
    total = len(sample_dirs)
    correct = missing_pred = missing_gt = 0
    for d in sample_dirs:
        m = re.search(r"sample_(\d+)", d.name)
        if not m:
            continue
        idx = int(m.group(1)) - 1  # sample_0001 corresponds to index 0
        gt = answers[idx] if idx < len(answers) else ""
        if not gt:
            missing_gt += 1
        resp = d / "response.json"
        if not resp.exists():
            missing_pred += 1
            continue
        payload = json.loads(resp.read_text(encoding="utf-8"))
        answer_text = payload.get("answer", "") or payload.get("reasoning", "")
        pred = extract_answer(answer_text) or ""
        if not pred:
            missing_pred += 1
        if pred and gt and pred == gt:
            correct += 1

    evaluated = total - missing_gt
    acc = correct / evaluated if evaluated else 0.0
    summary = {
        "outputs_dir": str(out_dir),
        "data_json": str(Path(args.dataset_json).resolve()),
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": acc,
        "missing_prediction": missing_pred,
        "total_files": total,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
