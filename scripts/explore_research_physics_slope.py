"""Train-only rolling search for a station/GHI slope on the physical anchor."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_research_dataset import CAPACITY_MW, ROOT
from compare_research_mlx import metrics, paired_improvement


OUT = ROOT / "data/processed/research"
EXPECTED = {
    "daily_train.jsonl": "9df5c2c495d3b79bb34906b105d52d7a4af871cd5a0f0a8fed2a307920d5fd80",
    "daily_valid.jsonl": "9b22a9b6c07bb0727ab18d7a616c96b0d1e59887b7e9fd9492e3941edc3affe1",
    "physics_train.jsonl": "63c3bf4972a505b6a9c97590224f52e5812b64024b30db6b469aa07e59502db4",
    "physics_valid.jsonl": "62aedeb0d51d9d92604f907fdbb37cf4a66c396fc97a00cf5f6785678b4a1881",
    "conditional_valid.jsonl": "932af3a25e99f6aaf01d544e4259bc42c102bd7be33157d57381753b706b5501",
}
FOLDS = (
    ("2018-10-31", "2018-11-01", "2018-11-30"),
    ("2018-11-30", "2018-12-01", "2018-12-31"),
    ("2018-12-31", "2019-01-01", "2019-01-31"),
    ("2019-01-31", "2019-02-01", "2019-02-28"),
)
LAMBDAS = (None, 0.0, 20.0, 100.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row: dict) -> tuple[str, str]:
    return row["station"], row["target_date"]


def load(split: str) -> list[dict]:
    daily = [json.loads(line) for line in (OUT / f"daily_{split}.jsonl").read_text().splitlines()]
    physical = [json.loads(line) for line in (OUT / f"physics_{split}.jsonl").read_text().splitlines()]
    raw = {key(row): row["raw_physics_mwh"] for row in physical}
    if len(raw) != len(physical) or set(raw) != {key(row) for row in daily}:
        raise ValueError(f"Unmatched daily/physics rows: {split}")
    if len({key(row) for row in daily}) != len(daily):
        raise ValueError(f"Duplicate station/date: {split}")
    return [{"station": r["station"], "target_date": r["target_date"],
             "ghi": r["inputs"]["nwp_ghi_kwh_m2"],
             "actual_h": r["target"]["actual_full_load_hours"],
             "raw_mwh": raw[key(r)]} for r in daily]


def fit(rows: list[dict], ridge: float | None) -> dict:
    scales = {}
    for station, capacity in CAPACITY_MW.items():
        eligible = [r for r in rows if r["station"] == station and r["raw_mwh"] >= 0.1 * capacity]
        if len(eligible) < 10:
            raise ValueError(f"Insufficient calibration rows for {station}")
        scales[station] = round(float(np.median([
            r["actual_h"] * capacity / r["raw_mwh"] for r in eligible
        ])), 8)
    threshold = float(np.quantile([r["ghi"] for r in rows], 2 / 3))
    grouped = defaultdict(list)
    for row in rows:
        station = row["station"]
        physical_h = min(row["raw_mwh"] * scales[station] / CAPACITY_MW[station], 24)
        band = "high" if row["ghi"] > threshold else "low"
        grouped[station, band].append((row["ghi"], row["actual_h"] - physical_h))
    groups = {}
    for station in CAPACITY_MW:
        for band in ("low", "high"):
            arr = np.asarray(grouped[station, band], dtype=float)
            if len(arr) < 10:
                raise ValueError(f"Insufficient group rows for {station}/{band}")
            ghi_mean = float(np.mean(arr[:, 0]))
            offset = float(np.mean(arr[:, 1]))
            centered = arr[:, 0] - ghi_mean
            slope = 0.0 if ridge is None else float(np.clip(
                np.sum(centered * (arr[:, 1] - offset)) /
                (np.sum(centered ** 2) + ridge), -0.5, 0.5
            ))
            groups[station, band] = {"rows": len(arr), "ghi_mean": ghi_mean,
                                     "offset": offset, "slope": slope}
    return {"scales": scales, "threshold": threshold, "groups": groups}


def predict(model: dict, rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        station = row["station"]
        band = "high" if row["ghi"] > model["threshold"] else "low"
        group = model["groups"][station, band]
        capacity = CAPACITY_MW[station]
        physical_h = min(row["raw_mwh"] * model["scales"][station] / capacity, 24)
        hours = float(np.clip(physical_h + group["offset"] +
                              group["slope"] * (row["ghi"] - group["ghi_mean"]), 0, 24))
        result.append({"station": station, "target_date": row["target_date"],
                       "forecast_full_load_hours": round(hours, 6),
                       "forecast_energy_mwh": round(hours * capacity, 6)})
    return result


def mean_abs(predictions: list[dict], rows: list[dict]) -> float:
    truth = {key(r): r["actual_h"] for r in rows}
    if len(truth) != len(rows) or len(predictions) != len(rows):
        raise ValueError("Prediction/label count mismatch")
    return float(np.mean([abs(p["forecast_full_load_hours"] - truth[key(p)])
                          for p in predictions]))


def main() -> None:
    for name, expected in EXPECTED.items():
        if sha256(OUT / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    train, valid = load("train"), load("valid")
    if len(train) != 890 or len(valid) != 184:
        raise ValueError("Unexpected split size")
    fold_results = []
    for train_end, valid_start, valid_end in FOLDS:
        fit_rows = [r for r in train if r["target_date"] <= train_end]
        eval_rows = [r for r in train if valid_start <= r["target_date"] <= valid_end]
        if len(fit_rows) < 200 or len(eval_rows) < 100:
            raise ValueError("Rolling fold too small")
        scores = {}
        for ridge in LAMBDAS:
            model = fit(fit_rows, ridge)
            scores["baseline" if ridge is None else str(ridge)] = round(
                mean_abs(predict(model, eval_rows), eval_rows), 6)
        fold_results.append({"train_end": train_end, "valid_start": valid_start,
                             "valid_end": valid_end, "fit_rows": len(fit_rows),
                             "eval_rows": len(eval_rows), "mae_hours": scores})
    mean_scores = {name: round(float(np.mean([f["mae_hours"][name] for f in fold_results])), 6)
                   for name in fold_results[0]["mae_hours"]}
    best_name = min((n for n in mean_scores if n != "baseline"), key=mean_scores.get)
    selected_name = (best_name if mean_scores["baseline"] - mean_scores[best_name] >= 0.01
                     else "baseline")
    ridge = None if selected_name == "baseline" else float(selected_name)
    model = fit(train, ridge)
    selected = predict(model, valid)
    original = [json.loads(line) for line in (OUT / "conditional_valid.jsonl").read_text().splitlines()]
    original = [{"station": r["station"], "target_date": r["target_date"],
                 "forecast_full_load_hours": r["conditioned_physics_full_load_hours"]} for r in original]
    reference_model = fit(train, None)
    reference = predict(reference_model, valid)
    if abs(mean_abs(reference, valid) - mean_abs(original, valid)) > 0.00001:
        raise ValueError("Refitted reference does not reproduce original conditional MAE")
    if selected_name == "baseline":
        selected = original
    truth = {key(r): r["actual_h"] for r in valid}
    output = OUT / "physics_slope_selected_valid.jsonl"
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    report = {
        "status": "exploratory validation; old test not read or scored",
        "source_sha256": EXPECTED,
        "protocol_sha256": sha256(ROOT / "reports/research_physics_slope_protocol.md"),
        "folds": fold_results, "mean_fold_mae_hours": mean_scores,
        "selection_margin_hours": 0.01, "selected": selected_name,
        "model": {"scales": model["scales"], "threshold": model["threshold"],
                  "groups": {f"{s}/{b}": g for (s, b), g in model["groups"].items()}},
        "validation": {"conditional": metrics(original, truth),
                       "selected": metrics(selected, truth),
                       "paired_selected_vs_conditional": paired_improvement(
                           {key(r): r for r in original}, {key(r): r for r in selected}, truth),
                       "per_station": {s: {"conditional": metrics(
                           [r for r in original if r["station"] == s], truth),
                           "selected": metrics([r for r in selected if r["station"] == s], truth)}
                           for s in CAPACITY_MW}},
        "selected_prediction_sha256": sha256(output),
    }
    path = OUT / "physics_slope_exploration.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
