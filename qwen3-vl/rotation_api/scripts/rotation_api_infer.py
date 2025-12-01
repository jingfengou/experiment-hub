"""
用 Qwen3-VL 云端 API（8B/32B 等）对旋转基准数据集进行问答推理。

输入：
- JSON 文件：/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json
- 图像根目录：/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data

输出：
- 每个样本一个子目录，保存 request.json 与 response.json（含模型回复文本）。

环境依赖：
pip install "openai>=1.3.0" pillow

运行示例：
  export QWEN_API_KEY=xxxx
  # 可选：export BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  python rotation_api_infer.py \
    --input_json /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json \
    --data_root /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data \
    --output_dir outputs/rotation_api_qwen3vl8b \
    --model qwen-vl-8b-instruct \
    --max_samples 100
"""
import argparse
import base64
import json
from pathlib import Path
from typing import List
import time

from openai import OpenAI
from PIL import Image


def load_json(path: Path) -> List[dict]:
    return json.loads(path.read_text())


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
    choices = item.get("Choices", [])
    choices_text = "\n".join([f"{chr(ord('A') + idx)}) {opt}" for idx, opt in enumerate(choices)])
    if choices_text and not choices_text.startswith("\n"):
        choices_text = "\n" + choices_text
    prompt = instruction + "<image>\n\n" + f"Question: {question}{choices_text}\n\nAnswer: "
    return prompt


def build_message(item: dict, image_path: Path):
    image_b64 = encode_image_b64(image_path)
    prompt = build_prompt(item)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/png;base64,{image_b64}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def infer_one(client: OpenAI, model: str, messages: list, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    content = resp.choices[0].message.content
    if isinstance(content, list):
        texts = [c.get("text", "") for c in content if isinstance(c, dict)]
        return "\n".join([t for t in texts if t]).strip()
    return str(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="旋转基准 JSON 文件")
    parser.add_argument("--data_root", required=True, help="图像根目录（含 Task/Level/Image_id）")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--model", default="qwen-vl-8b-instruct", help="API 模型名，如 qwen-vl-8b-instruct 或 qwen-vl-32b-instruct")
    parser.add_argument("--base_url", default=None, help="API Base URL，缺省则用 openai 默认或 BASE_URL 环境变量")
    parser.add_argument("--api_key", default=None, help="API Key，缺省则读环境变量 QWEN_API_KEY/OPENAI_API_KEY")
    parser.add_argument("--max_samples", type=int, default=0, help="截断样本数，0 表示全部")
    parser.add_argument("--max_tokens", type=int, default=256, help="生成最大 token 数")
    parser.add_argument("--sleep", type=float, default=0.0, help="两请求之间的延迟秒数，防止限速")
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
    for item in records:
        img_id = item.get("Image_id", "unknown_id")
        task = item.get("Task", "CubeRotation_MultiStep")
        level = item.get("Level", "Level0")
        img_rel = item.get("Question_image", "question.png")
        img_path = data_root / task / level / img_id / img_rel

        if not img_path.exists():
            print(f"[skip] {img_id}: image not found at {img_path}")
            skipped += 1
            continue

        messages = build_message(item, img_path)

        item_dir = output_dir / img_id
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / "request.json").write_text(json.dumps({"messages": messages}, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            answer = infer_one(client, args.model, messages, args.max_tokens)
            resp_payload = {"id": img_id, "model": args.model, "answer": answer}
            (item_dir / "response.json").write_text(json.dumps(resp_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] {img_id}: {answer[:80]}")
            processed += 1
        except Exception as e:
            err_path = item_dir / "error.log"
            err_path.write_text(str(e), encoding="utf-8")
            print(f"[err] {img_id}: {e}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Done. processed={processed}, skipped={skipped}, output={output_dir}")


if __name__ == "__main__":
    main()
