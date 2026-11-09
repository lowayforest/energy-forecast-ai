#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

model_dir="data/cache/research/mlx_qwen35_9b_4bit"
data_dir="data/processed/research/sft_residual_v3"
adapter_dir="data/processed/research/adapters/qwen35_9b_residual_v3"
if [[ -e "$adapter_dir" ]]; then
  echo "V3 adapter directory already exists; refusing to overwrite: $adapter_dir" >&2
  exit 1
fi

.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


weights = {
    "model-00001-of-00002.safetensors": "a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd",
    "model-00002-of-00002.safetensors": "b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d",
}
for name, expected in weights.items():
    if sha256(Path("data/cache/research/mlx_qwen35_9b_4bit") / name) != expected:
        raise SystemExit(f"Model weight changed: {name}")

out = Path("data/processed/research")
manifest = json.loads((out / "sft_residual_v3/manifest.json").read_text())
if manifest["test_status"] != "no test SFT file built":
    raise SystemExit("Unexpected test status")
if sha256(Path("prompts/research_pv_residual_class_v3.md")) != manifest["prompt_sha256"]:
    raise SystemExit("V3 prompt hash mismatch")
if manifest["prompt_sha256"] != "c0b9cc042eb7ef691b4146100462a0b871a0bcf8049a924ddae5fb0c76827b16":
    raise SystemExit("V3 prompt differs from preregistered version")
for split, expected_rows in (("train", 890), ("valid", 184)):
    path = out / f"sft_residual_v3/{split}.jsonl"
    if sha256(path) != manifest["splits"][split]["sha256"]:
        raise SystemExit(f"V3 {split} SFT hash mismatch")
    if manifest["splits"][split]["rows"] != expected_rows:
        raise SystemExit(f"V3 {split} count mismatch")
print("V3 training inputs verified", flush=True)
PY

.venv/bin/mlx_lm.lora \
  --model "$model_dir" \
  --train --data "$data_dir" \
  --fine-tune-type lora --mask-prompt --num-layers 2 \
  --batch-size 1 --iters 890 --val-batches 16 \
  --learning-rate 1e-5 --steps-per-report 32 --steps-per-eval 128 \
  --adapter-path "$adapter_dir" --save-every 256 \
  --max-seq-length 512 --grad-checkpoint --seed 42

.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


out = Path("data/processed/research")
adapter_dir = out / "adapters/qwen35_9b_residual_v3"
provenance = {
    "status": "890 training updates completed successfully",
    "sft_manifest_sha256": sha256(out / "sft_residual_v3/manifest.json"),
    "adapter_sha256": sha256(adapter_dir / "adapters.safetensors"),
    "adapter_config_sha256": sha256(adapter_dir / "adapter_config.json"),
}
(adapter_dir / "training_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(provenance, ensure_ascii=False, indent=2))
PY
