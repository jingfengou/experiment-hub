"""
将 Hugging Face 上的 ROVER 数据集导出为 MathCanvas/BAGEL-Canvas 可直接读取的格式。

输出：
- JSONL：包含 id、question、image 字段，供 UniHandler 读取
- images/：保存输入图像（image/image2 若存在）

默认输出目录：../MathCanvas/BAGEL-Canvas/data_handlers/rover

用法（bagel-canvas 环境）：
  # 如需镜像，可先 export HF_ENDPOINT=https://hf-mirror.com
  python prepare_rover_for_mathcanvas.py --split train --max_samples 0

参数：
- --split         数据切分（默认 train）
- --max_samples   截断样本数，0 表示不过滤
- --output_root   输出根目录（默认 ../MathCanvas/BAGEL-Canvas/data_handlers/rover）
"""
import argparse
import json
from pathlib import Path
from typing import List

from datasets import load_dataset


def build_question(prompt: str, target_desc: str, keywords: str) -> str:
    """构造给模型的文本指令。"""
    parts: List[str] = [
        f"Instruction: {prompt.strip()}",
        f"Target description: {target_desc.strip()}",
    ]
    if keywords:
        parts.append(f"Keywords: {keywords.strip()}")
    parts.append("Please edit/generate the final image to satisfy the target description. "
                 "Provide step-by-step reasoning and a final image.")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cheryyunl/ROVER", help="HF 数据集名称")
    parser.add_argument("--subset", default="ROVER-IG", help="HF 子集名称")
    parser.add_argument("--split", default="train", help="要导出的 split")
    parser.add_argument("--max_samples", type=int, default=0, help="截断样本数，0 表示全部")
    parser.add_argument(
        "--output_root",
        default="../MathCanvas/BAGEL-Canvas/data_handlers/rover",
        help="输出根目录（包含 jsonl 与 images 子目录）",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    images_dir = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset} ({args.subset}) split={args.split}")
    ds = load_dataset(args.dataset, args.subset, split=args.split)
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(len(ds), args.max_samples)))
        print(f"Truncated to {len(ds)} samples.")
    else:
        print(f"Total samples: {len(ds)}")

    records = []
    for item in ds:
        task_id = item["id"]
        prompt = item.get("prompt", "")
        keywords = item.get("keywords", "")
        target_desc = item.get("target_description", "")

        img_paths: List[str] = []

        # image1
        img1 = item.get("image")
        if img1 is not None:
            fname = f"{task_id}_img1.png"
            img1.save(images_dir / fname)
            img_paths.append(f"images/{fname}")

        # image2 (optional)
        img2 = item.get("image2")
        if img2 is not None:
            fname2 = f"{task_id}_img2.png"
            img2.save(images_dir / fname2)
            img_paths.append(f"images/{fname2}")

        record = {
            "id": task_id,
            "question": build_question(prompt, target_desc, keywords),
        }

        if len(img_paths) == 1:
            record["image"] = img_paths[0]
        elif len(img_paths) > 1:
            record["image"] = img_paths

        records.append(record)

    jsonl_path = output_root / "rover_dataset.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Exported {len(records)} samples.")
    print(f"JSONL: {jsonl_path}")
    print(f"Images dir: {images_dir}")


if __name__ == "__main__":
    main()
