"""Build leakage-aware chat SFT examples for physical residual prediction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from audit_research_dataset import CAPACITY_MW, ROOT
from evaluate_research_ollama import make_input


OUT = ROOT / "data/processed/research"
DATA_DIR = OUT / "sft_residual_v1"
PROMPT = ROOT / "prompts/research_pv_residual_sft_v1.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row: dict) -> tuple[str, str]:
    return row["station"], row["target_date"]


def build_split(split: str, system_prompt: str) -> tuple[list[dict], list[float]]:
    daily_path = OUT / f"daily_{split}.jsonl"
    physics_path = OUT / f"physics_{split}.jsonl"
    daily = read_jsonl(daily_path)
    physics = read_jsonl(physics_path)
    physics_by_key = {key(row): row for row in physics}
    if len(physics_by_key) != len(physics) or len(daily) != len(physics):
        raise ValueError(f"{split}: daily/physics duplicate or count mismatch")
    if {key(row) for row in daily} != set(physics_by_key):
        raise ValueError(f"{split}: daily/physics keys mismatch")

    examples = []
    residuals = []
    for row in daily:
        station = row["station"]
        capacity = CAPACITY_MW[station]
        if row["inputs"]["rated_capacity_mw"] != capacity:
            raise ValueError(f"{station}: capacity mismatch")
        physical = physics_by_key[key(row)]
        user_features = make_input(row, physical)
        if "target" in user_features or any(name.startswith("lmd_") for name in user_features):
            raise AssertionError("Target-day observation in SFT user prompt")
        residual = row["target"]["actual_full_load_hours"] - physical["calibrated_physics_mwh"] / capacity
        if not np.isfinite(residual):
            raise ValueError(f"{station}: nonfinite residual")
        rounded_residual = round(float(residual), 4)
        examples.append({"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_features, ensure_ascii=False, separators=(",", ":"))},
            {"role": "assistant", "content": json.dumps({"correction_full_load_hours": rounded_residual})},
        ]})
        residuals.append(float(residual))
    return examples, residuals


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    daily_manifest = json.loads((OUT / "daily_manifest.json").read_text())
    physics_manifest = json.loads((OUT / "physics_manifest.json").read_text())
    for split in ("train", "valid"):
        if sha256(OUT / f"daily_{split}.jsonl") != daily_manifest["splits"][split]["sha256"]:
            raise ValueError(f"{split}: changed daily dataset")
        if sha256(OUT / f"physics_{split}.jsonl") != physics_manifest["predictions"][split]["sha256"]:
            raise ValueError(f"{split}: changed physical predictions")

    system_prompt = PROMPT.read_text().strip()
    manifest = {
        "task": "chat SFT of daily full-load-hours residual over train-fitted physics",
        "prompt_sha256": sha256(PROMPT),
        "target_formula": "correction_full_load_hours = actual_full_load_hours - calibrated_physics_mwh / capacity_mw",
        "forecast_formula": "forecast_mwh = capacity_mw * clip(physics_full_load_hours + correction_full_load_hours, 0, 24)",
        "source_daily_manifest_sha256": sha256(OUT / "daily_manifest.json"),
        "source_physics_manifest_sha256": sha256(OUT / "physics_manifest.json"),
        "splits": {},
    }
    for split in ("train", "valid"):
        examples, residuals = build_split(split, system_prompt)
        path = DATA_DIR / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples))
        values = np.array(residuals)
        manifest["splits"][split] = {
            "rows": len(examples),
            "sha256": sha256(path),
            "residual_mean_hours": round(float(np.mean(values)), 6),
            "residual_std_hours": round(float(np.std(values)), 6),
            "residual_min_hours": round(float(np.min(values)), 6),
            "residual_max_hours": round(float(np.max(values)), 6),
        }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
