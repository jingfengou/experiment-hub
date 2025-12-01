#!/usr/bin/env python3
"""
Evaluate MathCanvas 3Dviews outputs (non-interleave sample*.txt or interleave reasoning_result.json folders).

Assumptions:
- Ground truth answers live in the 3Dviews dataset JSON (`Answer` field with A/B/C/D).
- Non-interleave outputs are sampleXXXXX.txt with text containing the prediction.
- Interleave outputs are per-sample folders (e.g., thinking_chain_0001/) containing reasoning_result.json;
  the prediction is parsed from concatenated reasoning text.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def extract_answer(text: str) -> Optional[str]:
    """Best-effort parse of model output to an option letter."""
    # Prefer <answer>...</answer>
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        inside = m.group(1)
        m2 = re.search(r"\b([A-D])\b", inside, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).upper()

    # Fallback: last standalone A-D
    matches = re.findall(r"\b([A-D])\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return None


def load_dataset(path: Path) -> Tuple[List[str], Dict[str, str]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    answers: List[str] = []
    id2ans: Dict[str, str] = {}
    for item in data:
        ans = (item.get("Answer") or "").strip().upper()
        ans_letter = ans[0] if ans else ""
        answers.append(ans_letter)
        sample_id = item.get("Image_id") or item.get("id") or ""
        if sample_id:
            id2ans[str(sample_id)] = ans_letter
    return answers, id2ans


def eval_plain(out_dir: Path, answers: List[str]) -> Tuple[int, int, int, int]:
    files = sorted(out_dir.glob("sample*.txt"))
    total = len(files)
    correct = missing_pred = missing_gt = 0
    for f in files:
        m = re.search(r"sample(\d+)", f.name)
        if not m:
            continue
        idx = int(m.group(1))
        gt = answers[idx] if idx < len(answers) else ""
        if not gt:
            missing_gt += 1
        pred = extract_answer(f.read_text(encoding="utf-8")) or ""
        if not pred:
            missing_pred += 1
        if pred and gt and pred == gt:
            correct += 1
    evaluated = total - missing_gt
    return total, evaluated, correct, missing_pred


def eval_interleave(out_dir: Path, id2ans: Dict[str, str]) -> Tuple[int, int, int, int]:
    sample_dirs = sorted([p for p in out_dir.iterdir() if p.is_dir()])
    total = len(sample_dirs)
    correct = missing_pred = missing_gt = 0
    for d in sample_dirs:
        rr = d / "reasoning_result.json"
        if not rr.exists():
            continue
        payload = json.loads(rr.read_text(encoding="utf-8"))
        sample_id = payload.get("id") or d.name
        gt = id2ans.get(sample_id, "")
        if not gt:
            missing_gt += 1
        texts: List[str] = []
        for step in payload.get("reasoning_steps", []):
            if isinstance(step, dict) and step.get("type") == "text":
                texts.append(step.get("content", ""))
        if payload.get("summary", {}).get("text"):
            texts.append(str(payload["summary"]["text"]))
        pred = extract_answer("\n".join(texts)) or ""
        if not pred:
            missing_pred += 1
        if pred and gt and pred == gt:
            correct += 1
    evaluated = total - missing_gt
    return total, evaluated, correct, missing_pred


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3Dviews outputs.")
    parser.add_argument("--output-dir", required=True, help="Output directory to evaluate.")
    parser.add_argument(
        "--dataset-json",
        default="/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/3Dviews/thinking_chain_dataset/dataset.json",
        help="3Dviews dataset JSON with Answer field.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    answers, id2ans = load_dataset(Path(args.dataset_json))

    has_plain = any(out_dir.glob("sample*.txt"))
    has_interleave = any((p / "reasoning_result.json").exists() for p in out_dir.iterdir() if p.is_dir())
    if not has_plain and not has_interleave:
        print(f"No recognizable outputs under {out_dir}")
        return

    if has_plain:
        total, evaluated, correct, missing_pred = eval_plain(out_dir, answers)
        acc = correct / evaluated if evaluated else 0.0
        print(f"[plain] total={total}, evaluated={evaluated}, correct={correct}, missing_pred={missing_pred}, acc={acc:.4f}")
    if has_interleave:
        total, evaluated, correct, missing_pred = eval_interleave(out_dir, id2ans)
        acc = correct / evaluated if evaluated else 0.0
        print(f"[interleave] total={total}, evaluated={evaluated}, correct={correct}, missing_pred={missing_pred}, acc={acc:.4f}")


if __name__ == "__main__":
    main()
