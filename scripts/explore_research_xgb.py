"""Train-only rolling XGBoost selection and exploratory validation comparison."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import xgboost as xgb

from audit_research_dataset import CAPACITY_MW, ROOT
from evaluate_research_mlx import metrics


OUT = ROOT / "data/processed/research"
EXPECTED = {
    "daily_train.jsonl": "9df5c2c495d3b79bb34906b105d52d7a4af871cd5a0f0a8fed2a307920d5fd80",
    "physics_train.jsonl": "63c3bf4972a505b6a9c97590224f52e5812b64024b30db6b469aa07e59502db4",
    "daily_valid.jsonl": "9b22a9b6c07bb0727ab18d7a616c96b0d1e59887b7e9fd9492e3941edc3affe1",
    "physics_valid.jsonl": "62aedeb0d51d9d92604f907fdbb37cf4a66c396fc97a00cf5f6785678b4a1881",
}
WEATHER = (
    "nwp_ghi_kwh_m2", "nwp_dni_kwh_m2", "nwp_ghi_peak_wm2",
    "nwp_temp_mean_c", "nwp_temp_max_c", "nwp_humidity_mean_pct", "nwp_wind_mean_mps",
)
FEATURES = (
    *[f"station_{s}" for s in CAPACITY_MW], "day_sin", "day_cos",
    *WEATHER, "rated_capacity_mw",
    *[f"history_h_{i}" for i in range(7)], "raw_physics_full_load_hours",
)
FOLDS = (
    ("2018-10-31", "2018-11-01", "2018-11-30"),
    ("2018-11-30", "2018-12-01", "2018-12-31"),
    ("2018-12-31", "2019-01-01", "2019-01-31"),
    ("2019-01-31", "2019-02-01", "2019-02-28"),
)
CANDIDATES = (
    {"name": "A", "max_depth": 2, "eta": 0.05, "num_boost_round": 150},
    {"name": "B", "max_depth": 3, "eta": 0.05, "num_boost_round": 200},
    {"name": "C", "max_depth": 4, "eta": 0.03, "num_boost_round": 300},
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row: dict) -> tuple[str, str]:
    return row["station"], row["target_date"]


def build(split: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    daily = read_jsonl(OUT / f"daily_{split}.jsonl")
    physical_rows = read_jsonl(OUT / f"physics_{split}.jsonl")
    physical = {key(r): r for r in physical_rows}
    if len(physical) != len(physical_rows) or {key(r) for r in daily} != set(physical):
        raise ValueError(f"{split}: duplicate or unmatched physics row")
    vectors = []
    labels = []
    for row in daily:
        station, day = key(row)
        if row["split"] != split or station not in CAPACITY_MW:
            raise ValueError("Unexpected station or split")
        data = row["inputs"]
        history = data["history_full_load_hours"]
        if len(history) != 7 or len(data["history_dates"]) != 7:
            raise ValueError("Unexpected history window")
        angle = 2 * math.pi * (data["day_of_year"] - 1) / 365.25
        features = (
            *[float(station == s) for s in CAPACITY_MW],
            math.sin(angle), math.cos(angle),
            *[data[name] for name in WEATHER], data["rated_capacity_mw"],
            *history, physical[station, day]["raw_physics_mwh"] / CAPACITY_MW[station],
        )
        if len(features) != len(FEATURES) or not np.isfinite(features).all():
            raise ValueError("Invalid or nonfinite feature vector")
        vectors.append(features)
        labels.append(row["target"]["actual_full_load_hours"])
    return daily, np.asarray(vectors, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def train(x: np.ndarray, y: np.ndarray, config: dict):
    params = {
        "objective": "reg:squarederror", "tree_method": "hist", "device": "cpu",
        "max_depth": config["max_depth"], "eta": config["eta"],
        "subsample": 0.9, "colsample_bytree": 0.9,
        "min_child_weight": 3, "reg_lambda": 10,
        "seed": 42, "nthread": 4,
    }
    return xgb.train(params, xgb.DMatrix(x, label=y, feature_names=list(FEATURES)),
                     num_boost_round=config["num_boost_round"], verbose_eval=False)


def predict(model, x: np.ndarray) -> np.ndarray:
    return np.clip(model.predict(xgb.DMatrix(x, feature_names=list(FEATURES))), 0, 24)


def main() -> None:
    if xgb.__version__ != "3.4.1":
        raise RuntimeError("Expected XGBoost 3.4.1")
    for name, wanted in EXPECTED.items():
        if sha256(OUT / name) != wanted:
            raise ValueError(f"Frozen input changed: {name}")
    train_rows, train_x, train_y = build("train")
    valid_rows, valid_x, valid_y = build("valid")
    if len(train_rows) != 890 or len(valid_rows) != 184:
        raise ValueError("Unexpected train/validation size")
    dates = np.asarray([r["target_date"] for r in train_rows])
    results = []
    for config in CANDIDATES:
        folds = []
        for train_end, valid_start, valid_end in FOLDS:
            fit_idx = np.where(dates <= train_end)[0]
            eval_idx = np.where((dates >= valid_start) & (dates <= valid_end))[0]
            if len(fit_idx) < 200 or len(eval_idx) < 100:
                raise ValueError("Rolling fold too small")
            model = train(train_x[fit_idx], train_y[fit_idx], config)
            forecast = predict(model, train_x[eval_idx])
            fold_mae = float(np.mean(np.abs(forecast - train_y[eval_idx])))
            folds.append({"train_end": train_end, "valid_start": valid_start,
                          "valid_end": valid_end, "train_rows": len(fit_idx),
                          "valid_rows": len(eval_idx), "mae_hours": round(fold_mae, 6)})
        results.append({"configuration": config, "folds": folds,
                        "mean_fold_mae_hours": round(float(np.mean([r["mae_hours"] for r in folds])), 6)})
    chosen = min(results, key=lambda r: r["mean_fold_mae_hours"])
    final_model = train(train_x, train_y, chosen["configuration"])
    model_path = OUT / "xgb_rolling_baseline_model.json"
    final_model.save_model(model_path)
    forecast = predict(final_model, valid_x)
    predictions = [
        {"station": r["station"], "target_date": r["target_date"],
         "forecast_full_load_hours": round(float(value), 6),
         "forecast_energy_mwh": round(float(value * CAPACITY_MW[r["station"]]), 6)}
        for r, value in zip(valid_rows, forecast)
    ]
    pred_path = OUT / "xgb_rolling_baseline_valid.jsonl"
    pred_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions))
    actual = {key(r): r["target"]["actual_full_load_hours"] for r in valid_rows}
    report = {
        "status": "exploratory; test split not read or scored",
        "xgboost_version": xgb.__version__, "source_sha256": EXPECTED,
        "protocol_sha256": sha256(ROOT / "reports/research_xgb_exploration_protocol.md"),
        "feature_names": FEATURES, "rolling_candidates": results,
        "selected_configuration": chosen["configuration"],
        "model_sha256": sha256(model_path), "prediction_sha256": sha256(pred_path),
        "validation": {"overall": metrics(predictions, actual),
                       "per_station": {s: metrics([r for r in predictions if r["station"] == s], actual)
                                       for s in CAPACITY_MW}},
    }
    manifest_path = OUT / "xgb_rolling_baseline_manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
