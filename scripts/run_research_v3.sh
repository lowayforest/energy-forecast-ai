#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

log_dir="data/processed/research/logs"
adapter_dir="data/processed/research/adapters/qwen35_9b_residual_v3"
mkdir -p "$log_dir"

.venv/bin/python scripts/build_research_sft_v3.py | tee "$log_dir/sft_v3.log"

if [[ -f "$adapter_dir/training_provenance.json" ]]; then
  echo "Using existing completed v3 adapter at $adapter_dir"
elif [[ -e "$adapter_dir" ]]; then
  echo "Partial v3 adapter directory exists at $adapter_dir; inspect before resuming" >&2
  exit 1
else
  /usr/bin/time -l bash scripts/train_research_qlora_v3.sh 2>&1 | tee "$log_dir/qlora_v3_890.log"
fi

.venv/bin/python scripts/evaluate_research_class_v3.py --variant base \
  | tee "$log_dir/mlx_v3_base_valid.log"
.venv/bin/python scripts/evaluate_research_class_v3.py --variant adapter \
  | tee "$log_dir/mlx_v3_adapter_valid.log"
.venv/bin/python scripts/compare_research_class_v3.py \
  | tee "$log_dir/mlx_v3_comparison_valid.log"

echo "V3 validation pipeline finished. The sealed test split was not read."
