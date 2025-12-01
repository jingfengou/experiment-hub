#!/usr/bin/env python3
"""
Create a rotation dataset variant that replaces intermediate step images with Hunyuan3D projections.

Inputs:
- Original dataset JSON/root (default: /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset).
- Hunyuan3D outputs under outputs/hunyuan3d_rotation_proj/sample_0001/projections/steps/step*.png.

Outputs:
- A new dataset root containing copied question/combined images and step images (step images replaced when available).
- A copied JSON (same structure/fields) under {output_root}/data_modified_with_subject.json.

Assumptions:
- sample_{idx+1:04d} corresponds to the dataset order.
- If a Hunyuan3D step image is missing, fallback to the original step image.
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


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Replace rotation step images with Hunyuan3D projections.")
    parser.add_argument(
        "--dataset-json",
        default="/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json",
        help="Original rotation dataset JSON.",
    )
    parser.add_argument(
        "--dataset-root",
        default="/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset",
        help="Original dataset root containing data/{Task}/{Level}/{Image_id}.",
    )
    parser.add_argument(
        "--hunyuan-root",
        default="/workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas/outputs/hunyuan3d_rotation_proj",
        help="Root folder with sample_XXXX/projections/steps/*.png from Hunyuan3D.",
    )
    parser.add_argument(
        "--output-root",
        default="data_handlers/rotation_hunyuan3d",
        help="Destination root to write images/json (will create data/...).",
    )
    parser.add_argument("--max-samples", type=int, default=1000, help="Limit samples (0 means all).")
    args = parser.parse_args()

    src_json = Path(args.dataset_json).resolve()
    src_root = Path(args.dataset_root).resolve()
    hunyuan_root = Path(args.hunyuan_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_data_root = out_root / "data"
    out_root.mkdir(parents=True, exist_ok=True)
    out_data_root.mkdir(parents=True, exist_ok=True)

    data = load_data(src_json)
    if args.max_samples > 0:
        data = data[: args.max_samples]
    print(f"Loaded {len(data)} samples from {src_json}")

    updated = 0
    for idx, item in enumerate(data):
        task = item.get("Task", "")
        level = item.get("Level", "")
        image_id = item.get("Image_id", "")
        src_dir = src_root / "data" / task / level / image_id
        dst_dir = out_data_root / task / level / image_id
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Copy question/combined if present
        for name in ["combined.png", "question.png"]:
            copy_if_exists(src_dir / name, dst_dir / name)

        # Copy step images, favor Hunyuan projections
        steps = item.get("Rotation_steps") or []
        for s_idx, step in enumerate(steps, start=1):
            rel = step.get("image", "")
            if not rel:
                continue
            dst_step = dst_dir / rel
            hy_step = hunyuan_root / f"sample_{idx+1:04d}" / "projections" / "steps" / f"step{s_idx}.png"
            used_hy = copy_if_exists(hy_step, dst_step)
            if not used_hy:
                copy_if_exists(src_dir / rel, dst_step)
            else:
                updated += 1

        # Copy aux files
        copy_if_exists(src_dir / "metadata.json", dst_dir / "metadata.json")
        copy_if_exists(src_dir / "full_rotation_script.py", dst_dir / "full_rotation_script.py")

    out_json = out_root / "data_modified_with_subject.json"
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"Wrote {out_json}")
    print(f"Step images replaced (Hunyuan) for {updated} steps (fallback used when missing).")


if __name__ == "__main__":
    main()
