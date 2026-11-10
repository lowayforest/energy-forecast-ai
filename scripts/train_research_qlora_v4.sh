#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

model_dir="data/cache/research/mlx_qwen35_9b_4bit"
data_dir="data/processed/research/sft_residual_v3"
adapter_dir="data/processed/research/adapters/qwen35_9b_class_only_v4"
if [[ -e "$adapter_dir" ]]; then
  echo "V4 adapter directory already exists; refusing to overwrite: $adapter_dir" >&2
  exit 1
fi

.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

model = Path("data/cache/research/mlx_qwen35_9b_4bit")
weights = {
    "model-00001-of-00002.safetensors": "a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd",
    "model-00002-of-00002.safetensors": "b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d",
}
for name, wanted in weights.items():
    if sha256(model / name) != wanted:
        raise SystemExit(f"Model weight changed: {name}")
manifest = Path("data/processed/research/sft_residual_v3/manifest.json")
if sha256(manifest) != "4da10cf8c060aa5e2465db1eac8a62e69055f83f761ea71ebdab45a19ac96f51":
    raise SystemExit("SFT manifest changed")
data = json.loads(manifest.read_text())
if data["test_status"] != "no test SFT file built":
    raise SystemExit("Test data entered training SFT")
print("V4 frozen training inputs verified", flush=True)
PY

.venv/bin/python scripts/train_research_class_only_v4.py \
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
adapter_dir = out / "adapters/qwen35_9b_class_only_v4"
provenance = {
    "status": "890 training updates completed successfully",
    "objective": "single_class_token_only",
    "sft_manifest_sha256": sha256(out / "sft_residual_v3/manifest.json"),
    "class_only_audit_sha256": sha256(out / "v4_class_only_audit.json"),
    "training_script_sha256": sha256(Path("scripts/train_research_class_only_v4.py")),
    "adapter_sha256": sha256(adapter_dir / "adapters.safetensors"),
    "adapter_config_sha256": sha256(adapter_dir / "adapter_config.json"),
}
(adapter_dir / "training_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(provenance, ensure_ascii=False, indent=2))
PY
