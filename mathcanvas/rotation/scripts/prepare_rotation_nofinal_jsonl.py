#!/usr/bin/env python3
"""
Prepare rotation benchmark data (without final step image) for interleave reasoning.

Outputs a JSONL consumable by UniHandler in mathcanvas_interleave_reasoner.py:
- question: question text + choices + step descriptions (final step image removed)
- image: list of image relative paths: question.png + step images (except last)
- images are copied under {output_root}/images/{Image_id}/...

Usage:
  python prepare_rotation_nofinal_jsonl.py \
    --dataset-json /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json \
    --dataset-root /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset \
    --output-root data_handlers/rotation_nofinal \
    --max-samples 1000
"""
import argparse
import json
from pathlib import Path
import shutil
from typing import List


def load_data(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data


def build_prompt(item: dict, step_desc: List[str]) -> str:
    """
    Prompt 主体与 infer_rotation_mathcanvas_ddp.py 保持一致，并在 Answer 前自然附上中间步骤描述。
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

    step_block = ""
    if step_desc:
        step_block = "\nHere are the intermediate rotation steps (the final step image is missing):\n" + "\n".join(step_desc)

    prompt = instruction + "<image>\n\n" + f"Question: {question}{choices_text}{step_block}\n\nAnswer: "
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True, help="Path to rotation JSON file")
    parser.add_argument("--dataset-root", required=True, help="Root folder containing data/<Task>/<Level>/<Image_id>/")
    parser.add_argument("--output-root", default="data_handlers/rotation_nofinal", help="Output root for JSONL and images")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples; 0 means all")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_json).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.output_root).resolve()
    images_root = out_root / "images"
    out_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    data = load_data(dataset_path)
    if args.max_samples > 0:
        data = data[: args.max_samples]
    print(f"Loaded {len(data)} samples from {dataset_path}")

    jsonl_path = out_root / "rotation_nofinal.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for item in data:
            task = item.get("Task", "")
            level = item.get("Level", "")
            image_id = item.get("Image_id", "")
            base_dir = dataset_root / "data" / task / level / image_id

            # Step descriptions (drop last)
            steps = item.get("Rotation_steps", []) or []
            step_desc = []
            for s in steps[:-1]:
                axis = s.get("axis")
                angle = s.get("angle")
                desc = s.get("description") or "rotate accordingly"
                axis_part = f" axis={axis}" if axis else ""
                angle_part = f", angle={angle} deg" if angle is not None else ""
                step_desc.append(f"Step {s.get('step', '')}: {desc}{axis_part}{angle_part}".strip())

            # 题面按 infer_rotation_mathcanvas_ddp.py 的 prompt 模板构造
            question_text = build_prompt(item, step_desc)

            rel_paths = []
            # question image
            q_img_rel = item.get("Question_image", "question.png")
            q_src = base_dir / q_img_rel
            q_dst = images_root / image_id / Path(q_img_rel).name
            q_dst.parent.mkdir(parents=True, exist_ok=True)
            if q_src.exists():
                shutil.copy(q_src, q_dst)
                rel_paths.append(str(q_dst.relative_to(out_root)))

            # step images (drop last)
            for s in steps[:-1]:
                rel = s.get("image")
                if not rel:
                    continue
                src = base_dir / rel
                dst = images_root / image_id / Path(rel).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.exists():
                    shutil.copy(src, dst)
                    rel_paths.append(str(dst.relative_to(out_root)))

            record = {"id": image_id, "question": question_text}
            if rel_paths:
                record["image"] = rel_paths if len(rel_paths) > 1 else rel_paths[0]
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved JSONL to {jsonl_path}")
    print(f"Images copied under {images_root}")


if __name__ == "__main__":
    main()
