#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

model_dir="data/cache/research/mlx_qwen35_9b_4bit"
data_dir="data/processed/research/sft_residual_v1"
adapter_dir="data/processed/research/adapters/qwen35_9b_residual_v1"

if [[ -e "$adapter_dir/adapters.safetensors" ]]; then
  echo "Adapter already exists at $adapter_dir; refusing to overwrite it" >&2
  exit 1
fi

.venv/bin/python - <<'PY'
import hashlib
from pathlib import Path

expected = {
    "data/cache/research/mlx_qwen35_9b_4bit/model-00001-of-00002.safetensors": "a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd",
    "data/cache/research/mlx_qwen35_9b_4bit/model-00002-of-00002.safetensors": "b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d",
    "data/processed/research/sft_residual_v1/train.jsonl": "262dd45abab6cac7f424bbe6b2785491c3b3a03911fbdf2a5039f002e6e5f1c9",
    "data/processed/research/sft_residual_v1/valid.jsonl": "aa326ce05e5d8631e80a5ef39d3bfca7b9dea471f53a025c1a67d8c0d0ac3da5",
}
for name, wanted in expected.items():
    digest = hashlib.sha256()
    with Path(name).open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != wanted:
        raise SystemExit(f"Input SHA-256 mismatch: {name}")
print("Training inputs verified", flush=True)
PY

.venv/bin/mlx_lm.lora \
  --model "$model_dir" \
  --train --data "$data_dir" \
  --fine-tune-type lora --mask-prompt --num-layers 2 \
  --batch-size 1 --iters 256 --val-batches 8 \
  --learning-rate 1e-5 --steps-per-report 16 --steps-per-eval 64 \
  --adapter-path "$adapter_dir" --save-every 64 \
  --max-seq-length 512 --grad-checkpoint --seed 42
