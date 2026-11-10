#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

out="data/processed/research"
log_dir="$out/logs"
provenance="$out/adapters/qwen35_9b_class_only_v4/training_provenance.json"
if [[ ! -f "$provenance" ]]; then
  echo "V4 full-training provenance is missing; refusing to evaluate a partial adapter" >&2
  exit 1
fi

.venv/bin/python scripts/evaluate_research_class_v3.py \
  --variant adapter --run-tag v4 > "$log_dir/mlx_v4_adapter_valid.log" 2>&1
.venv/bin/python scripts/evaluate_research_final_test.py \
  --variant adapter --run-tag v4 > "$log_dir/mlx_v4_adapter_old_test.log" 2>&1
.venv/bin/python scripts/compare_research_class_only_v4.py \
  > "$log_dir/mlx_v4_comparison.log" 2>&1
echo "V4 validation and retrospective old-test comparison finished. No new independent test was used."
