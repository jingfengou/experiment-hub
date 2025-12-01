#!/usr/bin/env python3
"""
Verify that rotation step images were replaced by Hunyuan3D projections.

Compares each step image in the new dataset root against the corresponding
Hunyuan3D projection file. Reports matched, missing (Hunyuan file absent),
and mismatch (both exist but differ).
"""
import argparse
import filecmp
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Verify Hunyuan3D step replacements.")
    parser.add_argument(
        "--dataset-root",
        default="/workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas/data_handlers/rotation_hunyuan3d/data",
        help="New dataset root with replaced step images.",
    )
    parser.add_argument(
        "--hunyuan-root",
        default="/workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas/outputs/hunyuan3d_rotation_proj",
        help="Hunyuan3D projections root (sample_XXXX/projections/steps/step*.png).",
    )
    parser.add_argument("--max-samples", type=int, default=1000, help="Number of samples to check (0 means all).")
    parser.add_argument("--steps-per-sample", type=int, default=2, help="Number of steps to check per sample.")
    args = parser.parse_args()

    new_root = Path(args.dataset_root).resolve()
    hy_root = Path(args.hunyuan_root).resolve()

    matched = missing = mismatch = 0
    checked = 0
    limit = args.max_samples if args.max_samples > 0 else None

    idx = 1
    while limit is None or idx <= limit:
        sid = f"sample_{idx:04d}"
        # Hunyuan dir exists?
        if not (hy_root / sid).exists():
            missing += args.steps_per_sample
            idx += 1
            continue

        for step_idx in range(1, args.steps_per_sample + 1):
            new_file = new_root / "CubeRotation_MultiStep" / "Level0" / sid / "steps" / f"step_{step_idx}.png"
            hy_file = hy_root / sid / "projections" / "steps" / f"step{step_idx}.png"
            if not hy_file.exists():
                missing += 1
                continue
            if new_file.exists() and filecmp.cmp(new_file, hy_file, shallow=False):
                matched += 1
            else:
                mismatch += 1
            checked += 1

        idx += 1

    print(f"checked_pairs={checked}, matched={matched}, missing_hunyuan_or_pair={missing}, mismatch={mismatch}")


if __name__ == "__main__":
    main()
