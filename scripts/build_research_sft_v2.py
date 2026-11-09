"""Build chat residual targets over the train-only conditional physical anchor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from audit_research_dataset import CAPACITY_MW, ROOT
from evaluate_research_ollama import make_input


OUT = ROOT / "data/processed/research"
SFT_DIR = OUT / "sft_residual_v2"
PROMPT = ROOT / "prompts/research_pv_residual_sft_v1.md"
EXPECTED_PROMPT_SHA256 = "8f5004dc8f08832c6eff3b956884f91a7b9953a0597d6841eb4b4e6c58e313aa"
DAILY_SHA256 = {
    "train": "9df5c2c495d3b79bb34906b105d52d7a4af871cd5a0f0a8fed2a307920d5fd80",
    "valid": "9b22a9b6c07bb0727ab18d7a616c96b0d1e59887b7e9fd9492e3941edc3affe1",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row: dict) -> tuple[str, str]:
    return row["station"], row["target_date"]


def main() -> None:
    conditional_manifest_path = OUT / "conditional_manifest.json"
    conditional_manifest = json.loads(conditional_manifest_path.read_text())
    if conditional_manifest["test_status"] != "not read or scored":
        raise ValueError("Conditional builder touched the sealed test split")
    if sha256(PROMPT) != EXPECTED_PROMPT_SHA256:
        raise ValueError("Residual prompt changed from preregistered v1 wording")
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    system_prompt = PROMPT.read_text().strip()
    manifest = {
        "task": "residual SFT over leave-one-out train and train-fitted validation conditional physics",
        "prompt_sha256": sha256(PROMPT),
        "conditional_manifest_sha256": sha256(conditional_manifest_path),
        "target_formula": "actual_full_load_hours - conditioned_physics_full_load_hours",
        "splits": {},
        "test_status": "no test SFT file built",
    }
    for split in ("train", "valid"):
        daily_path = OUT / f"daily_{split}.jsonl"
        conditional_path = OUT / f"conditional_{split}.jsonl"
        if sha256(daily_path) != DAILY_SHA256[split]:
            raise ValueError(f"Changed daily {split} dataset")
        if sha256(conditional_path) != conditional_manifest["predictions"][split]["sha256"]:
            raise ValueError(f"Changed conditional {split} prediction")
        daily = read_jsonl(daily_path)
        conditional = read_jsonl(conditional_path)
        by_key = {key(row): row for row in conditional}
        if len(by_key) != len(conditional) or len(daily) != len(conditional):
            raise ValueError(f"{split}: duplicate or missing conditional prediction")
        if {key(row) for row in daily} != set(by_key):
            raise ValueError(f"{split}: station/date mismatch")

        examples = []
        residuals = []
        for row in daily:
            station = row["station"]
            capacity = CAPACITY_MW[station]
            condition = by_key[key(row)]
            features = make_input(row, {"calibrated_physics_mwh": condition["conditioned_physics_mwh"]})
            if abs(features["physics_full_load_hours"] - condition["conditioned_physics_full_load_hours"]) > 1e-5:
                raise ValueError("Prompt anchor differs from conditional prediction")
            if "target" in features or any(name.startswith("lmd_") for name in features):
                raise ValueError("Target-day observations entered model prompt")
            if features["rated_capacity_mw"] != capacity:
                raise ValueError("Capacity mismatch")
            residual = row["target"]["actual_full_load_hours"] - features["physics_full_load_hours"]
            if not np.isfinite(residual):
                raise ValueError("Nonfinite residual")
            examples.append({"messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(features, ensure_ascii=False, separators=(",", ":"))},
                {"role": "assistant", "content": json.dumps({"correction_full_load_hours": round(float(residual), 4)})},
            ]})
            residuals.append(residual)
        path = SFT_DIR / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples))
        values = np.array(residuals)
        manifest["splits"][split] = {
            "rows": len(examples),
            "sha256": sha256(path),
            "residual_mean_h": round(float(np.mean(values)), 6),
            "residual_std_h": round(float(np.std(values)), 6),
        }
    (SFT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
