#!/usr/bin/env python3
"""
用 Qwen3-VL 官方 API（如 qwen3-vl-plus）跑旋转基准：
- 带中间过程：逐步提供题干图 + 各步骤图（去掉最后一步图像）+ 步骤描述。
- 不提供最终状态图，让模型根据中间过程推理答案。
- 采用 streaming + enable_thinking，兼容 DashScope compatible-mode。

示例：
  export QWEN_API_KEY=xxx
  python evaluation/rotation_api_infer_nofinal_steps.py \
    --input_json /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json \
    --data_root /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data \
    --output_dir outputs/rotation_api_qwen3vl235b_nofinal_steps \
    --model qwen-vl-plus \
    --max_samples 100
"""
import argparse
import base64
import json
import time
from pathlib import Path
from typing import List

from openai import OpenAI


def load_json(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data


def encode_image_b64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(item: dict) -> str:
    instruction = (
        "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
        "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
        "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
    )
    question = item.get("Question", "").strip()
    choices = item.get("Choices", []) or []
    choices_text = "\n".join([f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(choices)])
    if choices_text and not choices_text.startswith("\n"):
        choices_text = "\n" + choices_text
    return instruction + "<image>\n\n" + f"Question: {question}{choices_text}\n\nAnswer: "


def build_messages(item: dict, data_root: Path):
    task = item.get("Task", "CubeRotation_MultiStep")
    level = item.get("Level", "Level0")
    image_id = item.get("Image_id", "")
    question_img_rel = item.get("Question_image", "question.png")
    combined_img_rel = item.get("Combined_image")
    steps = (item.get("Rotation_steps") or [])[:-1]  # 去掉最终步骤

    def resolve(rel: str) -> Path:
        return data_root / task / level / image_id / rel

    images = []
    if combined_img_rel and (resolve(combined_img_rel)).exists():
        images.append(resolve(combined_img_rel))
    elif question_img_rel:
        images.append(resolve(question_img_rel))

    step_pairs = []
    for s in steps:
        img_rel = s.get("image")
        desc = s.get("description") or ""
        if img_rel:
            img_path = resolve(img_rel)
            if not img_path.exists():
                continue
            step_pairs.append((img_path, desc))

    content = []
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_b64(img)}"}})
    content.append({"type": "text", "text": build_prompt(item)})

    for img, desc in step_pairs:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_b64(img)}"}})
        if desc:
            content.append({"type": "text", "text": f"Step hint: {desc}"})

    return [{"role": "user", "content": content}]


def infer_one(client: OpenAI, model: str, messages: list, max_tokens: int, enable_thinking: bool, thinking_budget: int):
    reasoning_parts = []
    answer_parts = []
    started_answer = False

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
        extra_body={"enable_thinking": enable_thinking, "thinking_budget": thinking_budget},
    )

    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            reasoning_parts.append(rc)
        content = delta.content or ""
        if content:
            started_answer = True
            answer_parts.append(content)

    reasoning = "".join(reasoning_parts).strip()
    answer = "".join(answer_parts).strip()
    return reasoning, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="旋转基准 JSON 文件")
    parser.add_argument("--data_root", required=True, help="图像根目录（含 Task/Level/Image_id）")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--model", default="qwen3-vl-plus", help="API 模型名，如 qwen3-vl-plus / qwen-vl-max / qwen3-vl-235b")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API Base URL")
    parser.add_argument("--api_key", default=None, help="API Key，缺省则读 QWEN_API_KEY/OPENAI_API_KEY/DASHSCOPE_API_KEY")
    parser.add_argument("--max_samples", type=int, default=0, help="截断样本数，0 表示全部")
    parser.add_argument("--max_tokens", type=int, default=256, help="生成最大 token 数")
    parser.add_argument("--sleep", type=float, default=0.0, help="两请求之间的延迟秒数")
    parser.add_argument("--enable_thinking", action="store_true", help="开启思考过程（stream reasoning_content）")
    parser.add_argument("--thinking_budget", type=int, default=8192, help="思考过程最大 token 数")
    args = parser.parse_args()

    input_json = Path(args.input_json).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_json(input_json)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    print(f"Loaded {len(records)} samples from {input_json}")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    processed = 0
    skipped = 0
    for idx, item in enumerate(records):
        img_id = item.get("Image_id", f"sample_{idx:04d}")
        item_dir = output_dir / f"sample_{idx+1:04d}"
        item_dir.mkdir(parents=True, exist_ok=True)

        messages = build_messages(item, data_root)
        (item_dir / "request.json").write_text(
            json.dumps({"messages": messages}, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        try:
            reasoning, answer = infer_one(
                client,
                args.model,
                messages,
                args.max_tokens,
                enable_thinking=args.enable_thinking,
                thinking_budget=args.thinking_budget,
            )
            resp_payload = {
                "id": img_id,
                "model": args.model,
                "reasoning": reasoning,
                "answer": answer,
                "enable_thinking": args.enable_thinking,
                "thinking_budget": args.thinking_budget,
            }
            (item_dir / "response.json").write_text(json.dumps(resp_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] {item_dir.name}: {answer[:80]}")
            processed += 1
        except Exception as e:
            (item_dir / "error.log").write_text(str(e), encoding="utf-8")
            print(f"[err] {item_dir.name}: {e}")
            skipped += 1

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Done. processed={processed}, skipped={skipped}, output={output_dir}")


if __name__ == "__main__":
    main()
