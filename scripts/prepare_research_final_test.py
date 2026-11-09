"""Build label-free frozen test inputs from train-fitted physics calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from audit_research_dataset import CAPACITY_MW, ROOT
from evaluate_research_ollama import make_input


OUT = ROOT / "data/processed/research"
PROMPT = ROOT / "prompts/research_pv_residual_class_v3.md"
EXPECTED = {
    "daily_test.jsonl": "7fd4970581a695f2ef497017aab5065e7e0d5db3e9e737b5f971c060b97a14fe",
    "physics_test.jsonl": "097c7bb40d52609e3223380ebc84c3320f5edd542f36ffd50a6864dddc121d57",
    "conditional_manifest.json": "c08e802f57d7074f20e675f4a51f8a1a1be9be0d227cc5b4002974b04370d6a3",
    "sft_residual_v3/manifest.json": "4da10cf8c060aa5e2465db1eac8a62e69055f83f761ea71ebdab45a19ac96f51",
}
PROMPT_SHA256 = "4098a64de03d2bae1cd2b35307faee82022feca2715e5ff15dc806fd1436ee0c"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row: dict) -> tuple[str, str]:
    return row["station"], row["target_date"]


def main() -> None:
    for name, expected in EXPECTED.items():
        if sha256(OUT / name) != expected:
            raise ValueError(f"Changed frozen input: {name}")
    if sha256(PROMPT) != PROMPT_SHA256:
        raise ValueError("Changed frozen prompt")
    manifest = json.loads((OUT / "conditional_manifest.json").read_text())
    if manifest["test_status"] != "not read or scored":
        raise ValueError("Conditional calibration was changed")
    sft_manifest = json.loads((OUT / "sft_residual_v3/manifest.json").read_text())
    if sft_manifest["test_status"] != "no test SFT file built":
        raise ValueError("Test labels entered training SFT")

    # Drop labels immediately; the only label read after predictions is in the scorer.
    daily = [{k: v for k, v in row.items() if k != "target"}
             for row in read_jsonl(OUT / "daily_test.jsonl")]
    physics_rows = read_jsonl(OUT / "physics_test.jsonl")
    physics = {key(row): row for row in physics_rows}
    if len(daily) != 236 or len(physics) != 236 or {key(r) for r in daily} != set(physics):
        raise ValueError("Incomplete test inputs")
    threshold = manifest["threshold_nwp_ghi_kwh_m2"]
    conditional = []
    examples = []
    for row in daily:
        station, day = key(row)
        if row["split"] != "test" or station not in CAPACITY_MW:
            raise ValueError("Unexpected test split or station")
        capacity = CAPACITY_MW[station]
        band = "high" if row["inputs"]["nwp_ghi_kwh_m2"] > threshold else "low"
        offset = manifest["groups"][station][band]["train_mean_residual_h"]
        physical_h = physics[station, day]["calibrated_physics_mwh"] / capacity
        conditioned_h = round(float(np.clip(physical_h + offset, 0, 24)), 5)
        conditioned_mwh = round(conditioned_h * capacity, 5)
        conditional.append({
            "station": station, "target_date": day, "ghi_band": band,
            "conditional_offset_hours": round(offset, 6),
            "conditioned_physics_full_load_hours": conditioned_h,
            "conditioned_physics_mwh": conditioned_mwh,
        })
        features = make_input(row, {"calibrated_physics_mwh": conditioned_mwh})
        if abs(features["physics_full_load_hours"] - conditioned_h) > 1e-5:
            raise ValueError("Test prompt anchor mismatch")
        if "target" in features or any(name.startswith("lmd_") for name in features):
            raise ValueError("Test label entered inference prompt")
        examples.append({"messages": [
            {"role": "system", "content": PROMPT.read_text().strip()},
            {"role": "user", "content": json.dumps(features, ensure_ascii=False, separators=(",", ":"))},
        ]})
    if len({key(row) for row in conditional}) != 236:
        raise ValueError("Duplicate test station/date")
    conditional_path = OUT / "conditional_test.jsonl"
    examples_path = OUT / "v3_final_test_inputs.jsonl"
    conditional_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in conditional))
    examples_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in examples))
    output = {
        "rows": 236,
        "source_sha256": EXPECTED,
        "prompt_sha256": PROMPT_SHA256,
        "conditional_test_sha256": sha256(conditional_path),
        "test_inputs_sha256": sha256(examples_path),
        "test_labels_in_prediction_inputs": False,
    }
    (OUT / "v3_final_test_inputs_manifest.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
