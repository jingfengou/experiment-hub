#!/usr/bin/env python3
"""
Use official Qwen3-VL Instruct/Thinking (HF) to run rotation benchmark (base / nofinal).

Modes:
- base   : only question + question image.
- nofinal: question + step descriptions + intermediate step images (drop final step image).

Outputs:
- Text generations per sample stored as sampleXXXXX.txt under out_dir.

Dependencies: transformers>=4.57.0, pillow
"""
import argparse
import json
import os

# 在导入 transformers 之前设置默认镜像，确保 huggingface_hub 初始化时生效
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path
from typing import List

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from tqdm import tqdm
except ImportError:  # 兜底，若缺少 tqdm 则提供简易替代
    def tqdm(iterable, **kwargs):
        return iterable


def load_data(path: Path, max_samples: int):
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data[:max_samples] if max_samples > 0 else data


def build_prompt(item: dict, mode: str) -> str:
    instruction = (
        "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
        "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
        "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
        "Keep the reasoning concise (<=3 sentences) and always output both tags even if unsure; pick the most likely option.\n"
    )
    q = item.get("Question", "").strip()
    choices = item.get("Choices", [])
    choice_lines = "\n".join([f"{chr(65 + i)}) {c}" for i, c in enumerate(choices)])
    steps = item.get("Rotation_steps", []) or []
    use_steps = steps if mode == "nofinal" else []
    step_txt = "\n".join([f"Step {s.get('step', '')}: {s.get('description', '')}" for s in use_steps])
    parts = [instruction, "<image>", f"Question: {q}", choice_lines]
    if step_txt:
        parts.append(step_txt)
    parts.append("Answer: ")
    return "\n".join([p for p in parts if p.strip()])


def build_images(item: dict, mode: str, root: Path) -> List[str]:
    task, level, iid = item["Task"], item["Level"], item["Image_id"]
    base = root / "data" / task / level / iid
    # 优先使用带选项标注的合成图（Combined_image），否则退回 question 图
    combined = item.get("Combined_image") or item.get("combined_image")
    q_img = combined if combined else item.get("Question_image", "question.png")
    paths = [base / q_img]
    if mode == "nofinal":
        steps = (item.get("Rotation_steps", []) or [])[:-1]
        for s in steps:
            if s.get("image"):
                paths.append(base / s["image"])
    return [p.as_posix() for p in paths if p.exists()]


def run(args):
    data = load_data(Path(args.data_json), args.max_samples)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = args.model_id
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype="auto", device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    iterator = tqdm(data, total=len(data), desc=args.mode, dynamic_ncols=True)
    for i, item in enumerate(iterator):
        prompt = build_prompt(item, args.mode)
        images = build_images(item, args.mode, Path(args.data_root))

        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}],
        }]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        # presence_penalty 在部分模型未被支持，这里不传入以避免报错

        gen_ids = model.generate(**inputs, **gen_kwargs)
        gen_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
        text = processor.batch_decode(gen_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        (out_dir / f"sample{i:05d}.txt").write_text(text, encoding="utf-8")
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(file=f"sample{i:05d}.txt")

    print(args.mode, "finished:", len(data))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct", help="HF model id (默认指令版)")
    ap.add_argument("--mode", choices=["base", "nofinal"], required=True)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=16384)
    ap.add_argument(
        "--greedy",
        action="store_true",
        default=False,
        help="禁用采样，使用 greedy 解码",
    )
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--presence_penalty", type=float, default=1.5, help="目前不传给 generate，避免不支持时报错")
    args = ap.parse_args()
    run(args)
