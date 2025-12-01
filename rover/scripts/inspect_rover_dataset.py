"""
Quick inspector for the ROVER Hugging Face dataset.

Usage (run inside bagel-canvas env):
    source /workspace/oujingfeng/anaconda/anaconda3/bin/activate bagel-canvas
    python inspect_rover_dataset.py --split train[:5] --save rover_schema_sample.json

It prints feature schema and lightweight per-sample field types/sizes.
Optional --save writes the summary to a JSON file (without image bytes).
"""
import argparse
import json
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cheryyunl/ROVER", help="HF dataset name")
    parser.add_argument("--subset", default="ROVER-IG", help="HF subset name")
    parser.add_argument(
        "--split", default="train[:5]", help="Split to sample for inspection"
    )
    parser.add_argument("--save", type=str, help="Optional path to save summary JSON")
    args = parser.parse_args()

    print(f"Loading {args.dataset} ({args.subset}) split={args.split} ...", flush=True)
    ds = load_dataset(args.dataset, args.subset, split=args.split)
    print(f"Loaded {len(ds)} samples.")

    summary = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "num_samples": len(ds),
        "features": {k: str(v) for k, v in ds.features.items()},
        "samples": [],
    }

    for item in ds:
        sample_info = {}
        for k, v in item.items():
            entry = {"type": type(v).__name__}
            if hasattr(v, "size"):
                entry["size"] = getattr(v, "size", None)
            if isinstance(v, (int, float, str)):
                entry["value"] = v
            sample_info[k] = entry
        summary["samples"].append(sample_info)

    print("Feature schema:")
    print(json.dumps(summary["features"], indent=2, ensure_ascii=False))

    for idx, s in enumerate(summary["samples"]):
        print(f"\nSample {idx}:")
        print(json.dumps(s, indent=2, ensure_ascii=False))

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved summary to {args.save}")


if __name__ == "__main__":
    main()
