"""Compare class-probability LLM forecasts with the fixed conditional baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from audit_research_dataset import CAPACITY_MW, ROOT
from compare_research_mlx import metrics, paired_improvement


OUT = ROOT / "data/processed/research"
DAILY_SHA256 = "9b22a9b6c07bb0727ab18d7a616c96b0d1e59887b7e9fd9492e3941edc3affe1"
PHYSICS_SHA256 = "62aedeb0d51d9d92604f907fdbb37cf4a66c396fc97a00cf5f6785678b4a1881"
CONDITIONAL_SHA256 = "932af3a25e99f6aaf01d544e4259bc42c102bd7be33157d57381753b706b5501"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row: dict) -> tuple[str, str]:
    return row["station"], row["target_date"]


def keyed(rows: list[dict], actual: dict) -> dict:
    result = {key(row): row for row in rows}
    if len(result) != len(rows) or set(result) != set(actual):
        raise ValueError("Incomplete or duplicate validation predictions")
    return result


def main() -> None:
    for path, expected in (
        (OUT / "daily_valid.jsonl", DAILY_SHA256),
        (OUT / "physics_valid.jsonl", PHYSICS_SHA256),
        (OUT / "conditional_valid.jsonl", CONDITIONAL_SHA256),
    ):
        if sha256(path) != expected:
            raise ValueError(f"Source hash mismatch: {path}")
    daily = read_jsonl(OUT / "daily_valid.jsonl")
    actual = {key(row): row["target"]["actual_full_load_hours"] for row in daily}
    if len(daily) != 184 or len(actual) != 184:
        raise ValueError("Unexpected validation labels")
    physical = keyed([
        {"station": row["station"], "target_date": row["target_date"],
         "forecast_full_load_hours": row["calibrated_physics_mwh"] / CAPACITY_MW[row["station"]]}
        for row in read_jsonl(OUT / "physics_valid.jsonl")
    ], actual)
    conditional = keyed([
        {"station": row["station"], "target_date": row["target_date"],
         "forecast_full_load_hours": row["conditioned_physics_full_load_hours"]}
        for row in read_jsonl(OUT / "conditional_valid.jsonl")
    ], actual)
    sources = {"physical": physical, "conditional": conditional}
    input_hashes = {"daily_valid": DAILY_SHA256, "physical_valid": PHYSICS_SHA256,
                    "conditional_valid": CONDITIONAL_SHA256}
    for name, prefix in (
        ("adapter_v2", "mlx_v2_adapter_valid"),
        ("base_v3", "mlx_v3_base_valid"),
        ("adapter_v3", "mlx_v3_adapter_valid"),
    ):
        path = OUT / f"{prefix}.jsonl"
        summary = json.loads((OUT / f"{prefix}_summary.json").read_text())
        if sha256(path) != summary["prediction_sha256"] or summary["rows"] != 184:
            raise ValueError(f"{name}: prediction hash or count mismatch")
        sources[name] = keyed(read_jsonl(path), actual)
        input_hashes[name] = sha256(path)

    report = {
        "split": "valid",
        "rows": 184,
        "date_blocks": 46,
        "input_sha256": input_hashes,
        "metrics": {name: metrics(list(rows.values()), actual) for name, rows in sources.items()},
        "per_station": {
            station: {
                name: metrics([row for row in rows.values() if row["station"] == station], actual)
                for name, rows in sources.items()
            }
            for station in CAPACITY_MW
        },
        "paired": {
            "adapter_v3_vs_conditional": paired_improvement(conditional, sources["adapter_v3"], actual),
            "adapter_v3_vs_base_v3": paired_improvement(sources["base_v3"], sources["adapter_v3"], actual),
            "adapter_v3_vs_adapter_v2": paired_improvement(sources["adapter_v2"], sources["adapter_v3"], actual),
            "adapter_v3_vs_physical": paired_improvement(physical, sources["adapter_v3"], actual),
        },
    }
    output = OUT / "mlx_v3_comparison_valid.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
