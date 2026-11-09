"""Compare paired base and adapted MLX forecasts on the fixed validation dates."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_research_dataset import CAPACITY_MW, ROOT


OUT = ROOT / "data/processed/research"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row: dict) -> tuple[str, str]:
    return row["station"], row["target_date"]


def metrics(rows: list[dict], actual: dict) -> dict:
    errors = np.array([row["forecast_full_load_hours"] - actual[key(row)] for row in rows])
    return {
        "rows": len(rows),
        "mae_hours": round(float(np.mean(np.abs(errors))), 4),
        "rmse_hours": round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "bias_hours": round(float(np.mean(errors)), 4),
    }


def paired_improvement(reference: dict, candidate: dict, actual: dict) -> dict:
    by_date: dict[str, list[float]] = defaultdict(list)
    row_diff = []
    for station_day, row in sorted(candidate.items()):
        improvement = (
            abs(reference[station_day]["forecast_full_load_hours"] - actual[station_day])
            - abs(row["forecast_full_load_hours"] - actual[station_day])
        )
        by_date[station_day[1]].append(improvement)
        row_diff.append(improvement)
    if any(len(values) != len(CAPACITY_MW) for values in by_date.values()):
        raise ValueError("Date block does not contain all four stations")
    date_diff = np.array([np.mean(values) for _, values in sorted(by_date.items())])
    rng = np.random.default_rng(42)
    draws = rng.choice(date_diff, size=(10_000, len(date_diff)), replace=True).mean(axis=1)
    return {
        "mean_mae_improvement_hours": round(float(np.mean(date_diff)), 6),
        "bootstrap_95pct_interval_hours": [round(float(x), 6) for x in np.quantile(draws, [0.025, 0.975])],
        "date_blocks": len(date_diff),
        "bootstrap_draws": 10_000,
        "bootstrap_seed": 42,
        "rows_better": sum(value > 0 for value in row_diff),
        "rows_worse": sum(value < 0 for value in row_diff),
        "dates_better": int(np.sum(date_diff > 0)),
        "dates_worse": int(np.sum(date_diff < 0)),
    }


def main() -> None:
    daily = read_jsonl(OUT / "daily_valid.jsonl")
    actual = {key(row): row["target"]["actual_full_load_hours"] for row in daily}
    if len(actual) != 184:
        raise ValueError("Unexpected validation label count")

    data = {}
    paths = {}
    for name in ("base", "adapter"):
        path = OUT / f"mlx_{name}_valid.jsonl"
        summary = json.loads((OUT / f"mlx_{name}_valid_summary.json").read_text())
        if sha256(path) != summary["prediction_sha256"]:
            raise ValueError(f"{name}: prediction hash mismatch")
        rows = read_jsonl(path)
        keyed = {key(row): row for row in rows}
        if len(keyed) != len(actual) or set(keyed) != set(actual):
            raise ValueError(f"{name}: incomplete or duplicate validation predictions")
        data[name] = keyed
        paths[name] = {"sha256": sha256(path), "run_id": summary["run_id"]}

    physics_rows = read_jsonl(OUT / "physics_valid.jsonl")
    physics = {
        key(row): {
            "forecast_full_load_hours": row["calibrated_physics_mwh"] / CAPACITY_MW[row["station"]]
        }
        for row in physics_rows
    }
    if set(physics) != set(actual):
        raise ValueError("Physical predictions do not align")

    report = {
        "split": "valid",
        "rows": len(actual),
        "date_blocks": 46,
        "prediction_sources": paths,
    }
    # Keep station/date attached to each forecast while scoring.
    report["metrics"] = {
        "physical": metrics(
            [{"station": station, "target_date": day, **row} for (station, day), row in physics.items()], actual
        ),
        "base": metrics(list(data["base"].values()), actual),
        "adapter": metrics(list(data["adapter"].values()), actual),
    }
    report["per_station"] = {
        station: {
            name: metrics(
                [
                    {"station": station, "target_date": day, **row}
                    for (s, day), row in source.items() if s == station
                ],
                actual,
            )
            for name, source in (("physical", physics), ("base", data["base"]), ("adapter", data["adapter"]))
        }
        for station in CAPACITY_MW
    }
    report["paired"] = {
        "adapter_vs_base": paired_improvement(data["base"], data["adapter"], actual),
        "adapter_vs_physical": paired_improvement(physics, data["adapter"], actual),
    }
    output = OUT / "mlx_comparison_valid.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
