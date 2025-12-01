"""
将 MathCanvas/BAGEL-Canvas 推理产物转存为 ROVER 评测要求的文件命名：
  gen_{task_id}.png / gen_{task_id}.txt

假设推理输出结构：
  <inference_dir>/
    <task_id>/
      reasoning_result.json
      images/...

用法（bagel-canvas 环境）：
  python export_rover_generations.py \
    --inference_dir ../MathCanvas/BAGEL-Canvas/outputs/rover/rover_default \
    --dest_dir /path/to/ROVER_GEN_DIR

若未指定 --dest_dir，则尝试读取环境变量 ROVER_GEN_DIR。
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List


def load_reasoning_steps(result_path: Path):
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    task_id = data.get("id") or result_path.parent.name
    steps = data.get("reasoning_steps", [])
    return task_id, steps


def pick_final_image(steps: List[dict], base_dir: Path) -> Path | None:
    images = [s for s in steps if s.get("type") == "image"]
    if not images:
        return None
    final_rel = images[-1].get("content")
    if not final_rel:
        return None
    final_path = base_dir / final_rel
    return final_path if final_path.exists() else None


def collect_text(steps: List[dict]) -> str:
    texts = [s.get("content", "") for s in steps if s.get("type") == "text"]
    return "\n".join(texts).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_dir", required=True, help="mathcanvas_interleave_reasoner 输出目录（包含各 task 子目录）")
    parser.add_argument("--dest_dir", help="ROVER_GEN_DIR，未提供则读取环境变量 ROVER_GEN_DIR")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有 gen_{id}.*")
    args = parser.parse_args()

    dest_dir = args.dest_dir or os.getenv("ROVER_GEN_DIR")
    if not dest_dir:
        raise SystemExit("dest_dir 未指定且环境变量 ROVER_GEN_DIR 缺失。")

    inference_dir = Path(args.inference_dir).resolve()
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inference dir: {inference_dir}")
    print(f"Destination (ROVER_GEN_DIR): {dest_dir}")

    processed = 0
    skipped = 0
    missing_img = 0

    for item_dir in sorted(p for p in inference_dir.iterdir() if p.is_dir()):
        result_path = item_dir / "reasoning_result.json"
        if not result_path.exists():
            print(f"[skip] {item_dir.name}: missing reasoning_result.json")
            skipped += 1
            continue

        task_id, steps = load_reasoning_steps(result_path)
        final_img = pick_final_image(steps, item_dir)
        text_content = collect_text(steps)

        if final_img is None:
            print(f"[warn] {task_id}: no generated image found")
            missing_img += 1
            continue

        dest_img = dest_dir / f"gen_{task_id}.png"
        dest_txt = dest_dir / f"gen_{task_id}.txt"

        if dest_img.exists() and not args.overwrite:
            print(f"[skip] {task_id}: gen file exists (use --overwrite to replace)")
            skipped += 1
            continue

        shutil.copy(final_img, dest_img)
        if text_content:
            dest_txt.write_text(text_content, encoding="utf-8")

        processed += 1
        print(f"[ok] {task_id} -> {dest_img.name} ({'with text' if text_content else 'image only'})")

    print(f"\nDone. processed={processed}, skipped={skipped}, missing_img={missing_img}")
    print(f"Outputs in: {dest_dir}")


if __name__ == "__main__":
    main()
