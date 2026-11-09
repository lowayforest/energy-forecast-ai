"""Compare the frozen conditional-physics/MLX v2 experiment on validation only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from audit_research_dataset import CAPACITY_MW, ROOT
from compare_research_mlx import metrics, paired_improvement


OUT = ROOT / "data/processed/research"
DAILY_VALID_SHA256 = "9b22a9b6c07bb0727ab18d7a616c96b0d1e59887b7e9fd9492e3941edc3affe1"
PHYSICS_VALID_SHA256 = "62aedeb0d51d9d92604f907fdbb37cf4a66c396fc97a00cf5f6785678b4a1881"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row: dict) -> tuple[str, str]:
    return row["station"], row["target_date"]


def keyed(rows: list[dict], actual: dict) -> dict:
    result = {key(row): row for row in rows}
    if len(result) != len(rows) or set(result) != set(actual):
        raise ValueError("Duplicate, missing, or extra validation prediction")
    return result


def main() -> None:
    daily_path = OUT / "daily_valid.jsonl"
    physics_path = OUT / "physics_valid.jsonl"
    if sha256(daily_path) != DAILY_VALID_SHA256 or sha256(physics_path) != PHYSICS_VALID_SHA256:
        raise ValueError("Validation source hash mismatch")
    daily = read_jsonl(daily_path)
    actual = {key(row): row["target"]["actual_full_load_hours"] for row in daily}
    if len(actual) != 184 or len(daily) != 184:
        raise ValueError("Unexpected validation count")

    conditional_manifest_path = OUT / "conditional_manifest.json"
    conditional_manifest = json.loads(conditional_manifest_path.read_text())
    conditional_path = OUT / "conditional_valid.jsonl"
    if sha256(conditional_path) != conditional_manifest["predictions"]["valid"]["sha256"]:
        raise ValueError("Conditional prediction hash mismatch")
    physical = keyed([
        {"station": row["station"], "target_date": row["target_date"],
         "forecast_full_load_hours": row["calibrated_physics_mwh"] / CAPACITY_MW[row["station"]]}
        for row in read_jsonl(physics_path)
    ], actual)
    conditional = keyed([
        {"station": row["station"], "target_date": row["target_date"],
         "forecast_full_load_hours": row["conditioned_physics_full_load_hours"]}
        for row in read_jsonl(conditional_path)
    ], actual)

    sources = {"physical_v1": physical, "conditional_v2": conditional}
    prediction_sources = {
        "physical_v1": {"sha256": sha256(physics_path)},
        "conditional_v2": {"sha256": sha256(conditional_path)},
    }
    for name, file_prefix in (
        ("adapter_v1", "mlx_adapter_valid"),
        ("base_v2", "mlx_v2_base_valid"),
        ("adapter_v2", "mlx_v2_adapter_valid"),
    ):
        path = OUT / f"{file_prefix}.jsonl"
        summary = json.loads((OUT / f"{file_prefix}_summary.json").read_text())
        if sha256(path) != summary["prediction_sha256"]:
            raise ValueError(f"{name}: prediction hash mismatch")
        if name.endswith("v2") and summary["experiment"] != "v2":
            raise ValueError(f"{name}: experiment mismatch")
        sources[name] = keyed(read_jsonl(path), actual)
        prediction_sources[name] = {"sha256": sha256(path), "run_id": summary["run_id"]}

    report = {
        "split": "valid",
        "rows": 184,
        "date_blocks": 46,
        "conditional_manifest_sha256": sha256(conditional_manifest_path),
        "prediction_sources": prediction_sources,
        "metrics": {name: metrics(list(rows.values()), actual) for name, rows in sources.items()},
        "per_station": {
            station: {
                name: metrics([row for row in rows.values() if row["station"] == station], actual)
                for name, rows in sources.items()
            }
            for station in CAPACITY_MW
        },
        "paired": {
            "conditional_v2_vs_physical_v1": paired_improvement(physical, conditional, actual),
            "adapter_v2_vs_conditional_v2": paired_improvement(conditional, sources["adapter_v2"], actual),
            "adapter_v2_vs_base_v2": paired_improvement(sources["base_v2"], sources["adapter_v2"], actual),
            "adapter_v2_vs_adapter_v1": paired_improvement(sources["adapter_v1"], sources["adapter_v2"], actual),
            "adapter_v2_vs_physical_v1": paired_improvement(physical, sources["adapter_v2"], actual),
        },
    }
    output = OUT / "mlx_v2_comparison_valid.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
