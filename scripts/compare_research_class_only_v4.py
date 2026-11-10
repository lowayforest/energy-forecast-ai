"""Compare class-only v4 with frozen v3, including retrospective old-test diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from audit_research_dataset import CAPACITY_MW, ROOT
from compare_research_mlx import metrics, paired_improvement


OUT = ROOT / "data/processed/research"
EXPECTED = {
    "daily_valid.jsonl": "9b22a9b6c07bb0727ab18d7a616c96b0d1e59887b7e9fd9492e3941edc3affe1",
    "daily_test.jsonl": "7fd4970581a695f2ef497017aab5065e7e0d5db3e9e737b5f971c060b97a14fe",
    "conditional_valid.jsonl": "932af3a25e99f6aaf01d544e4259bc42c102bd7be33157d57381753b706b5501",
    "conditional_test.jsonl": "8e28da04ae859ddc511f1e2ec4336e1e1ed4c7a36106f0a30a0012b8f16feb8d",
    "mlx_v3_base_valid.jsonl": "628cafa33751848ed0324900366be338ad3b6663c429e26026479fc5bcb27555",
    "mlx_v3_adapter_valid.jsonl": "7db1ca7a694fae657727bca2fdfe65ae01982e19e17ba1ab02dc6009463bbc05",
    "mlx_v3_base_test.jsonl": "146fece212c4c24978c9d963a2d11c421f7cc6882e494c6a4b7b98b0c93441e5",
    "mlx_v3_adapter_test.jsonl": "9b1e9d6832c0422922892517f1b771707080cc75e420291b26af1c55ee161702",
    "sft_residual_v3/manifest.json": "4da10cf8c060aa5e2465db1eac8a62e69055f83f761ea71ebdab45a19ac96f51",
}
EXPOSED_TEST_DATE = "2019-04-16"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row: dict) -> tuple[str, str]:
    return row["station"], row["target_date"]


def class_label(residual: float, low: float, high: float) -> str:
    return "A" if residual <= low else "B" if residual <= high else "C"


def diagnostics(rows: dict, truth: dict, anchors: dict, low: float, high: float) -> dict:
    corrections = []
    residuals = []
    losses = []
    correct = 0
    for station_day, row in rows.items():
        residual = truth[station_day] - anchors[station_day]["forecast_full_load_hours"]
        label = class_label(residual, low, high)
        probabilities = row["class_probabilities"]
        losses.append(-math.log(max(probabilities[label], 1e-12)))
        correct += max(probabilities, key=probabilities.get) == label
        corrections.append(row["predicted_residual_hours"])
        residuals.append(residual)
    return {
        "class_cross_entropy": round(float(np.mean(losses)), 6),
        "hard_class_accuracy": round(correct / len(rows), 6),
        "mean_correction_hours": round(float(np.mean(corrections)), 6),
        "std_correction_hours": round(float(np.std(corrections)), 6),
        "true_residual_mean_hours": round(float(np.mean(residuals)), 6),
        "true_residual_std_hours": round(float(np.std(residuals)), 6),
        "correction_residual_pearson": round(float(np.corrcoef(corrections, residuals)[0, 1]), 6),
    }


def section(split: str, exclude_date: str | None, low: float, high: float) -> dict:
    daily = read_jsonl(OUT / f"daily_{split}.jsonl")
    if exclude_date:
        daily = [row for row in daily if row["target_date"] != exclude_date]
    truth = {key(row): row["target"]["actual_full_load_hours"] for row in daily}
    conditional = {key(row): {
        "station": row["station"], "target_date": row["target_date"],
        "forecast_full_load_hours": row["conditioned_physics_full_load_hours"],
    } for row in read_jsonl(OUT / f"conditional_{split}.jsonl") if key(row) in truth}
    if len(truth) != len(daily) or set(truth) != set(conditional):
        raise ValueError("Invalid labels or conditional predictions")
    sources = {"conditional": conditional}
    hashes = {}
    for name, filename in (
        ("base_v3", f"mlx_v3_base_{split}.jsonl"),
        ("adapter_v3", f"mlx_v3_adapter_{split}.jsonl"),
        ("adapter_v4", f"mlx_v4_adapter_{split}.jsonl"),
    ):
        path = OUT / filename
        if name == "adapter_v4":
            summary = json.loads((OUT / f"mlx_v4_adapter_{split}_summary.json").read_text())
            if summary["prediction_sha256"] != sha256(path) or summary["fallback_rows"] != 0:
                raise ValueError("Incomplete or changed v4 predictions")
        rows = [row for row in read_jsonl(path) if key(row) in truth]
        data = {key(row): row for row in rows}
        if len(rows) != len(truth) or set(data) != set(truth):
            raise ValueError(f"{name}: duplicate or missing predictions")
        if any("target" in row or "actual" in row for row in rows):
            raise ValueError("Labels entered prediction file")
        sources[name] = data
        hashes[name] = sha256(path)
    return {
        "rows": len(truth), "date_blocks": len({day for _, day in truth}),
        "prediction_sha256": hashes,
        "metrics": {name: metrics(list(rows.values()), truth) for name, rows in sources.items()},
        "per_station": {station: {
            name: metrics([row for k, row in rows.items() if k[0] == station], truth)
            for name, rows in sources.items()}
            for station in CAPACITY_MW},
        "paired": {
            "v4_vs_conditional": paired_improvement(conditional, sources["adapter_v4"], truth),
            "v4_vs_v3": paired_improvement(sources["adapter_v3"], sources["adapter_v4"], truth),
            "v4_vs_base": paired_improvement(sources["base_v3"], sources["adapter_v4"], truth),
        },
        "class_diagnostics": {name: diagnostics(rows, truth, conditional, low, high)
                              for name, rows in sources.items() if name != "conditional"},
    }


def main() -> None:
    for name, expected in EXPECTED.items():
        if sha256(OUT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    manifest = json.loads((OUT / "sft_residual_v3/manifest.json").read_text())
    low = manifest["class_boundary_hours"]["A_to_B"]
    high = manifest["class_boundary_hours"]["B_to_C"]
    report = {
        "purpose": "V4 objective ablation; old test is retrospective, not independent confirmation",
        "source_sha256": EXPECTED,
        "sections": {
            "valid": section("valid", None, low, high),
            "old_test_clean_58_dates": section("test", EXPOSED_TEST_DATE, low, high),
            "old_test_full_59_dates": section("test", None, low, high),
        },
    }
    path = OUT / "mlx_v4_comparison_valid_and_old_test.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
