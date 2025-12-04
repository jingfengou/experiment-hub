#!/usr/bin/env bash
set -euo pipefail

# Download BAGEL weights/configs to a local directory.
# Uses huggingface-cli download (works with older CLI without snapshot-download).

# Default to ByteDance-Seed/BAGEL-7B-MoT; override via MODEL_ID if needed.
MODEL_ID=${MODEL_ID:-ByteDance-Seed/BAGEL-7B-MoT}
DEST=${DEST:-/workspace/oujingfeng/modelckpt/BAGEL-7B-MoT}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

if [ -d "$DEST" ]; then
  echo "Destination $DEST already exists. Skipping download."
  exit 0
fi

echo "Downloading $MODEL_ID to $DEST ..."
HF_ENDPOINT=$HF_ENDPOINT huggingface-cli download "$MODEL_ID" \
  --local-dir "$DEST" \
  --local-dir-use-symlinks False

echo "Done. Files in $DEST"
