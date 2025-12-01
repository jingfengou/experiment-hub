#!/usr/bin/env python3
"""
使用 Qwen3-VL 235B（或兼容 API）跑 rotation 基准，支持 base / nofinal。

功能：
- 读取原始 rotation JSON（MathCanvas 数据），组装包含图像的对话消息并调用云端 API。
- base 模式：只用带选项的合成图（优先 Combined_image，否则 Question_image）。
- nofinal 模式：附带中间步骤图（去掉最后一步）并在 prompt 中给出步骤描述。
- 输出：每样本一个目录，包含 request.json / response.json（如有错误则 error.log）。

依赖：
pip install "openai>=1.3.0" pillow

运行示例（base，最多 50 条）：
  export QWEN_API_KEY=xxx
  python rotation_api_infer_235b.py \
    --input_json /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json \
    --data_root /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data \
    --output_dir outputs/rotation_api_qwen3vl235b_base \
    --model qwen-vl-235b-instruct \
    --mode base \
    --max_samples 50

nofinal（带中间步骤图，去掉最终图）：
  python rotation_api_infer_235b.py \
    --input_json .../data_modified_with_subject.json \
    --data_root .../dataset/data \
    --output_dir outputs/rotation_api_qwen3vl235b_nofinal \
    --model qwen-vl-235b-instruct \
    --mode nofinal \
    --max_samples 50
"""
import argparse
import base64
import json
import time
from pathlib import Path
from typing import List
import os
from openai import OpenAI
from PIL import Image


def load_json(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data


def encode_image_b64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(item: dict, mode: str, step_desc: List[str]) -> str:
    instruction = (
        "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
        "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
        "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
    )
    q = item.get("Question", "").strip()
    choices = item.get("Choices", [])
    choices_text = "\n".join([f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(choices)])

    step_block = ""
    if mode == "nofinal" and step_desc:
        step_block = "\nHere are the intermediate rotation steps (the final step image is missing):\n" + "\n".join(step_desc)

    prompt = instruction + "<image>\n\n" + f"Question: {q}\n{choices_text}{step_block}\n\nAnswer: "
    return prompt


def build_messages(item: dict, mode: str, data_root: Path) -> List[dict]:
    task = item.get("Task", "")
    level = item.get("Level", "")
    image_id = item.get("Image_id", "")
    base_dir = data_root / task / level / image_id

    # collect images
    images = []
    combined = item.get("Combined_image") or item.get("combined_image")
    q_img = combined if combined else item.get("Question_image", "question.png")
    q_path = base_dir / q_img
    if q_path.exists():
        images.append(q_path)

    step_desc = []
    if mode == "nofinal":
        steps = (item.get("Rotation_steps", []) or [])[:-1]
        for s in steps:
            if s.get("image"):
                img_path = base_dir / s["image"]
                if img_path.exists():
                    images.append(img_path)
            desc = s.get("description") or ""
            axis = s.get("axis")
            angle = s.get("angle")
            parts = [desc]
            if axis:
                parts.append(f"axis={axis}")
            if angle is not None:
                parts.append(f"angle={angle} deg")
            step_desc.append(f"Step {s.get('step', '')}: " + ", ".join([p for p in parts if p]))

    prompt = build_prompt(item, mode, step_desc)
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image_b64(p)}"},
        }
        for p in images
    ]
    content.append({"type": "text", "text": prompt})

    return [{"role": "user", "content": content}]


def infer_one(client: OpenAI, model: str, messages: list, max_tokens: int, temperature: float) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    if isinstance(content, list):
        texts = [c.get("text", "") for c in content if isinstance(c, dict)]
        return "\n".join([t for t in texts if t]).strip()
    return str(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="旋转基准 JSON 文件")
    ap.add_argument("--data_root", required=True, help="图像根目录（含 data/Task/Level/Image_id/...）")
    ap.add_argument("--output_dir", required=True, help="输出目录")
    ap.add_argument("--model", default="qwen3-vl-plus", help="API 模型名，例如 qwen3-vl-plus / qwen-vl-235b-instruct 等")
    ap.add_argument("--base_url", default=None, help="API Base URL，缺省用环境变量 BASE_URL；若未设置且存在 DASHSCOPE_API_KEY，则默认 https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--api_key", default=None, help="API Key，缺省读 DASHSCOPE_API_KEY/QWEN_API_KEY/OPENAI_API_KEY")
    ap.add_argument("--mode", choices=["base", "nofinal"], default="base", help="推理模式：base 或 nofinal")
    ap.add_argument("--max_samples", type=int, default=0, help="截断样本数，0 表示全部")
    ap.add_argument("--max_tokens", type=int, default=512, help="生成最大 token 数")
    ap.add_argument("--temperature", type=float, default=0.0, help="采样温度，默认 0（贪心）")
    ap.add_argument("--sleep", type=float, default=0.0, help="请求间延迟秒数，防限速")
    args = ap.parse_args()

    records = load_json(Path(args.input_json))
    if args.max_samples > 0:
        records = records[: args.max_samples]
    print(f"Loaded {len(records)} samples from {args.input_json}")

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 解析 API key 和 base_url
    env_api_key = (
        args.api_key
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    env_base_url = args.base_url or os.getenv("BASE_URL")
    # 若使用阿里云百炼 DASHSCOPE_API_KEY 且未显式指定 base_url，则默认北京地域兼容地址
    if env_api_key and os.getenv("DASHSCOPE_API_KEY") and not env_base_url:
        env_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not env_api_key:
        raise ValueError("Missing API key: set DASHSCOPE_API_KEY / QWEN_API_KEY / OPENAI_API_KEY or --api_key")

    client = OpenAI(base_url=env_base_url, api_key=env_api_key)

    processed = skipped = 0
    for item in records:
        img_id = item.get("Image_id", "unknown_id")
        item_dir = out_root / img_id
        item_dir.mkdir(parents=True, exist_ok=True)

        messages = build_messages(item, args.mode, data_root)
        # 支持 image_url 结构，防止误判无图
        if not any(
            c
            for msg in messages
            for c in msg.get("content", [])
            if c.get("type") in ("image_url", "image")
        ):
            (item_dir / "error.log").write_text("no image found", encoding="utf-8")
            skipped += 1
            continue

        (item_dir / "request.json").write_text(json.dumps({"messages": messages}, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            answer = infer_one(client, args.model, messages, args.max_tokens, args.temperature)
            resp_payload = {"id": img_id, "model": args.model, "answer": answer}
            (item_dir / "response.json").write_text(json.dumps(resp_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] {img_id}: {answer[:80]}")
            processed += 1
        except Exception as e:
            (item_dir / "error.log").write_text(str(e), encoding="utf-8")
            print(f"[err] {img_id}: {e}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Done. processed={processed}, skipped={skipped}, output={out_root}")


if __name__ == "__main__":
    main()
