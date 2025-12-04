#!/usr/bin/env python3
"""
Run Qwen3-VL-8B Instruct on 3Dviews benchmark in four modes:
- base_non (question + combined/question image)
- steps_non (question + step hints text + combined/question image)
- interleave_base (question text + question image as one message)
- interleave_steps (question text + question image, then step image+hint per step as separate messages)

Outputs per sample: saves raw response text to sample_XXXX.txt and request/response json for auditing.
"""
import argparse
import json
from pathlib import Path
from typing import List
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


INSTRUCTION = (
    "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
    "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
    "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
    "Keep the reasoning concise (<=3 sentences) and always output both tags even if unsure; pick the most likely option.\n"
)


def load_data(path: Path, max_samples: int):
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data[:max_samples] if max_samples > 0 else data


def build_choices(item: dict) -> str:
    choices = item.get("Choices", [])
    return "\n".join([f"{chr(65 + i)}) {c}" for i, c in enumerate(choices)])


def build_step_hints(item: dict) -> List[str]:
    hints = []
    for s in item.get("Steps", []) or []:
        desc = s.get("thought") or s.get("description") or ""
        if desc:
            hints.append(f"Step {s.get('step', '')}: {desc}".strip())
    return hints


def resolve_image_paths(item: dict, data_root: Path, include_steps: bool):
    image_id = item.get("Image_id", "")
    combined = item.get("Combined_image", "") or item.get("Question_image", "question.png")
    base_dir = data_root / "data" / item.get("Task", "") / item.get("Level", "") / image_id
    imgs = []
    q_path = base_dir / combined
    if q_path.exists():
        imgs.append(("question", q_path))
    if include_steps:
        for s in item.get("Steps", []) or []:
            rel = s.get("image")
            if not rel:
                continue
            p = base_dir / rel
            if p.exists():
                imgs.append(("step", p, s.get("thought") or s.get("description") or ""))
    return imgs


def build_messages(item: dict, mode: str, data_root: Path):
    q = item.get("Question", "").strip()
    choices = build_choices(item)
    hints = build_step_hints(item)
    prompt = "\n".join([INSTRUCTION, f"Question: {q}", choices, "Answer: "])

    content = []
    images = resolve_image_paths(item, data_root, include_steps="steps" in mode)

    if mode.startswith("interleave"):
        # First message: question image + prompt
        if images:
            first = images[0]
            content.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": first[1].as_posix()},
                        {"type": "text", "text": prompt},
                    ],
                }
            )
            step_imgs = images[1:]
        else:
            content.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
            step_imgs = []

        if "steps" in mode:
            for name, path, hint in step_imgs:
                parts = [{"type": "image", "image": path.as_posix()}]
                if hint:
                    parts.append({"type": "text", "text": f"Step hint: {hint}"})
                content.append({"role": "user", "content": parts})
    else:
        # Non-interleave: single message with image + all hints in text
        parts = []
        if images:
            parts.append({"type": "image", "image": images[0][1].as_posix()})
        text_block = prompt
        if "steps" in mode and hints:
            text_block = text_block.replace("Answer: ", f"Step hints:\n" + "\n".join(hints) + "\n\nAnswer: ")
        parts.append({"type": "text", "text": text_block})
        content.append({"role": "user", "content": parts})

    return content


def infer_one(model, processor, messages: list, max_new_tokens: int, temperature: float) -> str:
    input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [c["image"] for m in messages for c in m["content"] if c["type"] == "image"]
    image_inputs = [processor.image_processor(images=Path(p).open("rb").read(), return_tensors="pt")["pixel_values"][0] for p in images] if images else None

    inputs = processor(
        text=[input_text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    # drop prompt tokens
    generate_ids = generate_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-json", required=True, help="Path to 3Dviews dataset json.")
    parser.add_argument("--data-root", required=True, help="Root to 3Dviews data/...")
    parser.add_argument("--output-dir", default="outputs/qwen3vl8b_3dviews", help="Output directory.")
    parser.add_argument("--mode", choices=["base_non", "steps_non", "interleave_base", "interleave_steps"], default="base_non")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct", help="HF model id (remote).")
    parser.add_argument(
        "--model-path",
        default="/workspace/oujingfeng/modelckpt/Qwen3-VL-8B-Instruct",
        help="Local model path for weights (used if exists).",
    )
    parser.add_argument(
        "--processor-id",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Processor source (always used; set to remote id to avoid local config issues).",
    )
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    data = load_data(Path(args.data_json), args.max_samples)
    out_dir = Path(args.output_dir) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    model_src = args.model_path if Path(args.model_path).exists() else args.model_id
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_src, device_map="auto", torch_dtype="auto", trust_remote_code=True
    )
    # Processor: always use processor_id (remote) to avoid local config lacking model_type
    processor = AutoProcessor.from_pretrained(args.processor_id, trust_remote_code=True)

    for i, item in enumerate(data):
        messages = build_messages(item, args.mode, Path(args.data_root))
        try:
            resp = infer_one(model, processor, messages, args.max_new_tokens, args.temperature)
        except Exception as e:
            resp = f"[ERROR] {e}"

        sample_name = f"sample_{i+1:04d}"
        (out_dir / f"{sample_name}.txt").write_text(resp, encoding="utf-8")
        (out_dir / f"{sample_name}_request.json").write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")

        if (i + 1) % 50 == 0:
            print(f"[{args.mode}] processed {i+1}/{len(data)}")


if __name__ == "__main__":
    main()
