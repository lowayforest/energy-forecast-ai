#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

log_dir="data/processed/research/logs"
adapter_dir="data/processed/research/adapters/qwen35_9b_residual_v2"
mkdir -p "$log_dir"

.venv/bin/python scripts/build_research_conditional.py | tee "$log_dir/conditional_v2.log"
.venv/bin/python scripts/build_research_sft_v2.py | tee "$log_dir/sft_v2.log"

if [[ -f "$adapter_dir/training_provenance.json" ]]; then
  echo "Using existing completed v2 adapter at $adapter_dir"
elif [[ -e "$adapter_dir" ]]; then
  echo "Partial v2 adapter directory exists at $adapter_dir; stop and inspect before resuming" >&2
  exit 1
else
  /usr/bin/time -l bash scripts/train_research_qlora_v2.sh 2>&1 | tee "$log_dir/qlora_v2_256.log"
fi

.venv/bin/python scripts/evaluate_research_mlx.py --experiment v2 --variant base \
  | tee "$log_dir/mlx_v2_base_valid.log"
.venv/bin/python scripts/evaluate_research_mlx.py --experiment v2 --variant adapter \
  | tee "$log_dir/mlx_v2_adapter_valid.log"
.venv/bin/python scripts/compare_research_mlx_v2.py \
  | tee "$log_dir/mlx_v2_comparison_valid.log"

echo "V2 validation pipeline finished. The sealed test split was not read."
