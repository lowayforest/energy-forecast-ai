"""Diagnose validation errors without reading the sealed research test split."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_research_dataset import CAPACITY_MW, ROOT


OUT = ROOT / "data/processed/research"
INPUT_SHA256 = {
    "daily_train.jsonl": "9df5c2c495d3b79bb34906b105d52d7a4af871cd5a0f0a8fed2a307920d5fd80",
    "daily_valid.jsonl": "9b22a9b6c07bb0727ab18d7a616c96b0d1e59887b7e9fd9492e3941edc3affe1",
    "physics_train.jsonl": "63c3bf4972a505b6a9c97590224f52e5812b64024b30db6b469aa07e59502db4",
    "physics_valid.jsonl": "62aedeb0d51d9d92604f907fdbb37cf4a66c396fc97a00cf5f6785678b4a1881",
    "mlx_base_valid.jsonl": "2c9bfdca815fa6a5003e973287d76fa13efd780ced2545fa903ea1d582cad10f",
    "mlx_adapter_valid.jsonl": "08d1446a148c2a6b1deecce17dacb61808cba07ea992fcb9a7b58eb67c44599e",
}
WEATHER_FIELDS = ("nwp_ghi_kwh_m2", "nwp_temp_mean_c", "nwp_humidity_mean_pct")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(name: str) -> list[dict]:
    path = OUT / name
    if sha256(path) != INPUT_SHA256[name]:
        raise ValueError(f"Input hash mismatch: {name}")
    return [json.loads(line) for line in path.read_text().splitlines()]


def keyed(rows: list[dict]) -> dict[tuple[str, str], dict]:
    result = {(row["station"], row["target_date"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate station/date")
    return result


def mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 4)


def summary(rows: list[dict]) -> dict:
    actual = np.array([row["actual_h"] for row in rows])
    physical = np.array([row["physical_h"] for row in rows])
    base = np.array([row["base_h"] for row in rows])
    adapter = np.array([row["adapter_h"] for row in rows])
    correction = np.array([row["adapter_correction_h"] for row in rows])
    residual = actual - physical
    return {
        "rows": len(rows),
        "physical_mae_h": mean(list(np.abs(physical - actual))),
        "base_mae_h": mean(list(np.abs(base - actual))),
        "adapter_mae_h": mean(list(np.abs(adapter - actual))),
        "mean_true_residual_h": mean(list(residual)),
        "mean_adapter_correction_h": mean(list(correction)),
        "std_true_residual_h": round(float(np.std(residual)), 4),
        "std_adapter_correction_h": round(float(np.std(correction)), 4),
        "residual_correction_pearson": round(float(np.corrcoef(residual, correction)[0, 1]), 4),
        "adapter_better_than_physical_rows": int(np.sum(np.abs(adapter - actual) < np.abs(physical - actual))),
        "adapter_worse_than_physical_rows": int(np.sum(np.abs(adapter - actual) > np.abs(physical - actual))),
        "adapter_better_than_base_rows": int(np.sum(np.abs(adapter - actual) < np.abs(base - actual))),
        "adapter_worse_than_base_rows": int(np.sum(np.abs(adapter - actual) > np.abs(base - actual))),
    }


def build_joined(split: str, predictions: dict[str, dict] | None = None) -> list[dict]:
    daily = keyed(load_rows(f"daily_{split}.jsonl"))
    physics = keyed(load_rows(f"physics_{split}.jsonl"))
    if set(daily) != set(physics):
        raise ValueError(f"{split}: daily and physics keys differ")
    if predictions and any(set(rows) != set(daily) for rows in predictions.values()):
        raise ValueError("Validation prediction keys differ from labels")
    joined = []
    for station, day in sorted(daily):
        row = daily[station, day]
        if row["split"] != split or station not in CAPACITY_MW:
            raise ValueError("Unexpected split or station")
        actual_h = row["target"]["actual_full_load_hours"]
        physical_h = physics[station, day]["calibrated_physics_mwh"] / CAPACITY_MW[station]
        item = {
            "station": station,
            "date": day,
            "inputs": row["inputs"],
            "actual_h": actual_h,
            "physical_h": physical_h,
        }
        if predictions:
            item["base_h"] = predictions["base"][station, day]["forecast_full_load_hours"]
            item["adapter_h"] = predictions["adapter"][station, day]["forecast_full_load_hours"]
            item["adapter_correction_h"] = item["adapter_h"] - physical_h
        joined.append(item)
    return joined


def weather_groups(rows: list[dict], field: str) -> dict:
    values = np.array([row["inputs"][field] for row in rows])
    low, high = (float(x) for x in np.quantile(values, [1 / 3, 2 / 3]))
    groups = {
        "low": [row for row in rows if row["inputs"][field] <= low],
        "middle": [row for row in rows if low < row["inputs"][field] <= high],
        "high": [row for row in rows if row["inputs"][field] > high],
    }
    if sum(len(group) for group in groups.values()) != len(rows):
        raise ValueError("Weather bins do not partition validation rows")
    return {
        "cut_points_from_validation_inputs": [round(low, 4), round(high, 4)],
        "groups": {name: summary(group) for name, group in groups.items()},
    }


def case(row: dict) -> dict:
    actual, physical, adapter = row["actual_h"], row["physical_h"], row["adapter_h"]
    return {
        "station": row["station"],
        "date": row["date"],
        "nwp_ghi_kwh_m2": row["inputs"]["nwp_ghi_kwh_m2"],
        "actual_h": round(actual, 4),
        "physical_h": round(physical, 4),
        "adapter_h": round(adapter, 4),
        "physical_abs_error_h": round(abs(physical - actual), 4),
        "adapter_abs_error_h": round(abs(adapter - actual), 4),
        "mae_improvement_h": round(abs(physical - actual) - abs(adapter - actual), 4),
    }


def main() -> None:
    predictions = {
        name: keyed(load_rows(f"mlx_{name}_valid.jsonl")) for name in ("base", "adapter")
    }
    train = build_joined("train")
    valid = build_joined("valid", predictions)
    if len(train) != 890 or len(valid) != 184:
        raise ValueError("Unexpected train/validation sample counts")
    if any(row["date"] > "2019-02-28" for row in train):
        raise ValueError("Training dates exceed cutoff")
    if any(not ("2019-03-01" <= row["date"] <= "2019-04-15") for row in valid):
        raise ValueError("Validation dates outside cutoff")

    report = {
        "split": "train/valid only; test files not read",
        "input_sha256": INPUT_SHA256,
        "counts": {"train": len(train), "valid": len(valid)},
        "overall_valid": summary(valid),
        "per_station": {},
        "input_distribution": {},
        "valid_weather_terciles": {},
    }
    high_ghi_cutoff = float(
        np.quantile([row["inputs"]["nwp_ghi_kwh_m2"] for row in train], 2 / 3)
    )
    report["high_ghi_cutoff_from_train_kwh_m2"] = round(high_ghi_cutoff, 4)
    for station in CAPACITY_MW:
        train_rows = [row for row in train if row["station"] == station]
        valid_rows = [row for row in valid if row["station"] == station]
        station_summary = summary(valid_rows)
        station_summary["train_rows"] = len(train_rows)
        station_summary["train_mean_true_residual_h"] = mean(
            [row["actual_h"] - row["physical_h"] for row in train_rows]
        )
        station_summary["train_mean_nwp_ghi_kwh_m2"] = mean(
            [row["inputs"]["nwp_ghi_kwh_m2"] for row in train_rows]
        )
        station_summary["valid_mean_nwp_ghi_kwh_m2"] = mean(
            [row["inputs"]["nwp_ghi_kwh_m2"] for row in valid_rows]
        )
        for split, source in (("train", train_rows), ("valid", valid_rows)):
            high_ghi_rows = [
                row for row in source if row["inputs"]["nwp_ghi_kwh_m2"] > high_ghi_cutoff
            ]
            station_summary[f"{split}_high_ghi_rows"] = len(high_ghi_rows)
            station_summary[f"{split}_high_ghi_mean_true_residual_h"] = mean(
                [row["actual_h"] - row["physical_h"] for row in high_ghi_rows]
            )
        report["per_station"][station] = station_summary

    for field in WEATHER_FIELDS:
        train_values = np.array([row["inputs"][field] for row in train])
        valid_values = np.array([row["inputs"][field] for row in valid])
        train_upper_tercile = float(np.quantile(train_values, 2 / 3))
        report["input_distribution"][field] = {
            "train_mean": mean(list(train_values)),
            "valid_mean": mean(list(valid_values)),
            "train_upper_tercile": round(train_upper_tercile, 4),
            "valid_share_above_train_upper_tercile": round(
                float(np.mean(valid_values > train_upper_tercile)), 4
            ),
        }
        report["valid_weather_terciles"][field] = weather_groups(valid, field)

    ranked = sorted(
        valid,
        key=lambda row: abs(row["physical_h"] - row["actual_h"])
        - abs(row["adapter_h"] - row["actual_h"]),
    )
    report["largest_worsening_vs_physical"] = [case(row) for row in ranked[:5]]
    report["largest_improvement_vs_physical"] = [case(row) for row in ranked[-5:][::-1]]
    by_date = defaultdict(list)
    for row in valid:
        by_date[row["date"]].append(
            abs(row["physical_h"] - row["actual_h"])
            - abs(row["adapter_h"] - row["actual_h"])
        )
    if len(by_date) != 46 or any(len(values) != 4 for values in by_date.values()):
        raise ValueError("Expected 46 complete four-station date blocks")
    report["valid_dates_improved_vs_physical"] = int(
        sum(np.mean(values) > 0 for values in by_date.values())
    )

    path = OUT / "error_analysis_valid.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {path}")
    print(json.dumps({
        "overall_valid": report["overall_valid"],
        "per_station": report["per_station"],
        "input_distribution": report["input_distribution"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
