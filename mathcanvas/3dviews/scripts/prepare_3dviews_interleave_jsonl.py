#!/usr/bin/env python3
"""
Prepare 3Dviews thinking-chain data for mathcanvas_interleave_reasoner.py (UniHandler).

Modes:
- base: only question + question/combined image.
- steps: question + step hints (sanitized) + question/combined image + step images.

Outputs:
  {output_root}/{mode}/3dviews_{mode}.jsonl
  {output_root}/{mode}/images/<Image_id>/* (copied)
"""
import argparse
import json
import re
import shutil
from pathlib import Path
from typing import List


FORBIDDEN_PAT = re.compile(r"(option|选项|answer|正确|正确选项|final answer)", re.IGNORECASE)


def load_data(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data


def clean_step_text(text: str) -> str:
    sentences = re.split(r"(?<=[.?!。！？])\s+", text)
    kept = [s for s in sentences if s and not FORBIDDEN_PAT.search(s)]
    cleaned = " ".join(kept).strip()
    return cleaned


def build_prompt(item: dict, step_desc: List[str], include_steps: bool) -> str:
    instruction = (
        "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
        "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
        "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
    )
    question = item.get("Question", "")
    choices = item.get("Choices", [])
    choices_text = "\n".join([f"{chr(65 + i)}) {c}" for i, c in enumerate(choices)]) if choices else ""
    if choices_text and not choices_text.startswith("\n"):
        choices_text = "\n" + choices_text

    prompt = instruction + "<image>\n\n" + f"Question: {question}{choices_text}\n"
    if include_steps and step_desc:
        hints = "\n".join([f"- {d}" for d in step_desc])
        prompt += f"\nStep hints (sanitized, options hidden):\n{hints}\n"
    prompt += "\nAnswer: "
    return prompt


def copy_image(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy(src, dst)
        return str(dst)
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True, help="Path to 3Dviews dataset.json")
    parser.add_argument("--dataset-root", required=True, help="Root folder containing example_xxxx/")
    parser.add_argument("--output-root", default="data_handlers/3dviews_interleave", help="Output root")
    parser.add_argument("--mode", choices=["base", "steps"], default="base", help="Data mode")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples; 0 means all")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_json).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.output_root).resolve() / args.mode
    images_root = out_root / "images"
    out_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    data = load_data(dataset_path)
    if args.max_samples > 0:
        data = data[: args.max_samples]
    print(f"Loaded {len(data)} samples from {dataset_path}")

    jsonl_path = out_root / f"3dviews_{args.mode}.jsonl"
    include_steps = args.mode == "steps"
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for item in data:
            combined_image = item.get("Combined_image", "")
            question_image = item.get("Question_image", "")
            image_id = item.get("Image_id", "")
            base_dir = dataset_root
            if combined_image:
                base_dir = dataset_root / Path(combined_image).parent
            elif image_id:
                base_dir = dataset_root / image_id

            # step texts (sanitized)
            step_desc: List[str] = []
            if include_steps:
                for s in item.get("Steps", []) or []:
                    thought = s.get("thought", "") or ""
                    cleaned = clean_step_text(thought)
                    if cleaned:
                        step_desc.append(f"Step {s.get('step', '')}: {cleaned}".strip())

            question_text = build_prompt(item, step_desc, include_steps)

            rel_paths = []
            # Prefer combined image, fall back to question image
            first_image = combined_image or question_image
            if first_image:
                src = dataset_root / first_image if not Path(first_image).is_absolute() else Path(first_image)
                if not src.exists() and base_dir is not None:
                    src = base_dir / Path(first_image).name
                dst = images_root / image_id / Path(first_image).name
                copied = copy_image(src, dst)
                if copied:
                    rel_paths.append(str(dst.relative_to(out_root)))

            # step images
            if include_steps:
                for s in item.get("Steps", []) or []:
                    rel = s.get("image")
                    if not rel:
                        continue
                    src = base_dir / rel
                    dst = images_root / image_id / Path(rel).name
                    copied = copy_image(src, dst)
                    if copied:
                        rel_paths.append(str(dst.relative_to(out_root)))

            record = {"id": image_id, "question": question_text}
            if rel_paths:
                record["image"] = rel_paths if len(rel_paths) > 1 else rel_paths[0]
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved JSONL to {jsonl_path}")
    print(f"Images copied under {images_root}")


if __name__ == "__main__":
    main()
