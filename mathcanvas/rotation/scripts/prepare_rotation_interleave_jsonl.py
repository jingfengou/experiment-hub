#!/usr/bin/env python3
"""
Prepare rotation benchmark data for mathcanvas_interleave_reasoner.py (UniHandler).

Modes:
- base: only question + question image (no step images/descriptions).
- steps: include all step descriptions and all step images.
- nofinal: include step descriptions, but drop the final step image.

Outputs:
  {output_root}/{mode}/rotation_{mode}.jsonl
  {output_root}/{mode}/images/<Image_id>/* (copied)

Example:
  python prepare_rotation_interleave_jsonl.py \
    --dataset-json /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json \
    --dataset-root /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset \
    --output-root data_handlers/rotation_interleave \
    --mode nofinal \
    --max-samples 1000
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import List


def load_data(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data


def build_prompt(item: dict) -> str:
    """
    与 infer_rotation_mathcanvas_ddp.py 的 build_prompt 保持一致。
    """
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
    prompt = instruction + "<image>\n\n" + f"Question: {question}{choices_text}\n\nAnswer: "
    return prompt


def copy_image(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy(src, dst)
        return str(dst)
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True, help="Path to rotation JSON file")
    parser.add_argument("--dataset-root", required=True, help="Root folder containing data/<Task>/<Level>/<Image_id>/")
    parser.add_argument("--output-root", default="data_handlers/rotation_interleave", help="Output root")
    parser.add_argument("--mode", choices=["base", "steps", "nofinal"], default="nofinal", help="Data mode")
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

    jsonl_path = out_root / f"rotation_{args.mode}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for item in data:
            task = item.get("Task", "")
            level = item.get("Level", "")
            image_id = item.get("Image_id", "")
            base_dir = dataset_root / "data" / task / level / image_id

            steps = item.get("Rotation_steps", []) or []
            step_desc = []
            use_steps = steps if args.mode == "steps" else steps[:-1] if args.mode == "nofinal" else []
            for s in use_steps:
                axis = s.get("axis")
                angle = s.get("angle")
                desc = s.get("description") or "rotate accordingly"
                axis_part = f" axis={axis}" if axis else ""
                angle_part = f", angle={angle} deg" if angle is not None else ""
                step_desc.append(f"Step {s.get('step', '')}: {desc}{axis_part}{angle_part}".strip())

            # 题面按 infer_rotation_mathcanvas_ddp.py 的 prompt 模板构造
            question_text = build_prompt(item)

            rel_paths = []
            # question image
            q_img_rel = item.get("Question_image", "question.png")
            q_src = base_dir / q_img_rel
            q_dst = images_root / image_id / Path(q_img_rel).name
            copied = copy_image(q_src, q_dst)
            if copied:
                rel_paths.append(str(q_dst.relative_to(out_root)))

            # step images (depending on mode)
            for s in use_steps:
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
