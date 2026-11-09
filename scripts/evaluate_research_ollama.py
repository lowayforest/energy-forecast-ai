"""Evaluate a frozen zero-shot Ollama prompt on the held-out validation dates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from audit_research_dataset import CAPACITY_MW, ROOT


OUT = ROOT / "data/processed/research"
PROMPT = ROOT / "prompts/research_pv_zero_shot_v1.md"
MODEL = "qwen3.5:9b"
MODEL_DIGEST = "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
API = "http://127.0.0.1:11434/api/generate"
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 80}
SCHEMA = {
    "type": "object",
    "properties": {"forecast_full_load_hours": {"type": "number"}},
    "required": ["forecast_full_load_hours"],
    "additionalProperties": False,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row: dict) -> tuple[str, str]:
    return row["station"], row["target_date"]


def make_input(daily: dict, physics: dict) -> dict:
    x = daily["inputs"]
    capacity = x["rated_capacity_mw"]
    return {
        "station": daily["station"],
        "target_date": daily["target_date"],
        "rated_capacity_mw": capacity,
        "day_of_year": x["day_of_year"],
        "nwp_ghi_kwh_m2": x["nwp_ghi_kwh_m2"],
        "nwp_dni_kwh_m2": x["nwp_dni_kwh_m2"],
        "nwp_ghi_peak_wm2": x["nwp_ghi_peak_wm2"],
        "nwp_temp_mean_c": x["nwp_temp_mean_c"],
        "nwp_temp_max_c": x["nwp_temp_max_c"],
        "nwp_humidity_mean_pct": x["nwp_humidity_mean_pct"],
        "nwp_wind_mean_mps": x["nwp_wind_mean_mps"],
        "history_dates": x["history_dates"],
        "history_full_load_hours": x["history_full_load_hours"],
        "physics_full_load_hours": round(physics["calibrated_physics_mwh"] / capacity, 5),
    }


def invoke(prompt: str, features: dict) -> dict:
    body = {
        "model": MODEL,
        "system": prompt,
        "prompt": json.dumps(features, ensure_ascii=False, separators=(",", ":")),
        "format": SCHEMA,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": OPTIONS,
    }
    request = urllib.request.Request(
        API,
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    result["client_elapsed_seconds"] = round(time.monotonic() - start, 3)
    if not result.get("done"):
        raise RuntimeError("Ollama returned an incomplete response")
    return result


def verify_local_model() -> None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as response:
            models = json.load(response)["models"]
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Cannot inspect local Ollama models: {exc}") from exc
    matching = [model for model in models if model["name"] == MODEL]
    if len(matching) != 1 or matching[0]["digest"] != MODEL_DIGEST:
        raise RuntimeError(f"Expected local {MODEL} with digest {MODEL_DIGEST}")


def decode_forecast(response: str, anchor_hours: float) -> tuple[float, str | None]:
    try:
        result = json.loads(response)
        if set(result) != {"forecast_full_load_hours"}:
            raise ValueError("unexpected fields")
        value = result["forecast_full_load_hours"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("nonfinite or nonnumeric value")
        if not 0 <= value <= 24:
            raise ValueError("outside 0–24 h")
        return float(value), None
    except (json.JSONDecodeError, TypeError, ValueError):
        return anchor_hours, "invalid_model_output"


def score(rows: list[dict], daily_by_key: dict, physics_by_key: dict) -> dict:
    actual = np.array([daily_by_key[key(row)]["target"]["actual_full_load_hours"] for row in rows])
    model = np.array([row["forecast_full_load_hours"] for row in rows])
    physical = np.array([
        physics_by_key[key(row)]["calibrated_physics_mwh"] / CAPACITY_MW[row["station"]]
        for row in rows
    ])

    def metrics(pred: np.ndarray) -> dict:
        error = pred - actual
        return {
            "mae_hours": round(float(np.mean(np.abs(error))), 4),
            "rmse_hours": round(float(np.sqrt(np.mean(error ** 2))), 4),
            "bias_hours": round(float(np.mean(error)), 4),
        }

    return {"model": metrics(model), "physical": metrics(physical)}


def paired_diagnostics(rows: list[dict], daily_by_key: dict, physics_by_key: dict) -> dict:
    """Resample dates, preserving four-station dependence within each date."""
    by_date: dict[str, list[float]] = {}
    corrections = []
    row_improvements = []
    for row in rows:
        station_day = key(row)
        capacity = CAPACITY_MW[row["station"]]
        actual = daily_by_key[station_day]["target"]["actual_full_load_hours"]
        physical = physics_by_key[station_day]["calibrated_physics_mwh"] / capacity
        model = row["forecast_full_load_hours"]
        improvement = abs(physical - actual) - abs(model - actual)
        by_date.setdefault(row["target_date"], []).append(improvement)
        row_improvements.append(improvement)
        corrections.append(model - physical)

    date_improvements = np.array([np.mean(values) for _, values in sorted(by_date.items())])
    rng = np.random.default_rng(42)
    bootstrap = rng.choice(date_improvements, size=(10_000, len(date_improvements)), replace=True).mean(axis=1)
    return {
        "date_count": len(date_improvements),
        "mean_mae_improvement_hours": round(float(np.mean(date_improvements)), 6),
        "bootstrap_95pct_interval_hours": [round(float(x), 6) for x in np.quantile(bootstrap, [0.025, 0.975])],
        "bootstrap_method": "10,000 date-block resamples, seed 42; positive means model improves MAE",
        "rows_better": sum(value > 0 for value in row_improvements),
        "rows_worse": sum(value < 0 for value in row_improvements),
        "dates_better": int(np.sum(date_improvements > 0)),
        "dates_worse": int(np.sum(date_improvements < 0)),
        "mean_signed_correction_hours": round(float(np.mean(corrections)), 6),
        "mean_absolute_correction_hours": round(float(np.mean(np.abs(corrections))), 6),
        "p95_absolute_correction_hours": round(float(np.quantile(np.abs(corrections), 0.95)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Run only the first N validation rows for a smoke test")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    verify_local_model()

    split = "valid"  # Test labels remain sealed until the final plan is frozen.
    daily_path = OUT / f"daily_{split}.jsonl"
    physics_path = OUT / f"physics_{split}.jsonl"
    manifest_path = OUT / "physics_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256(physics_path) != manifest["predictions"][split]["sha256"]:
        raise ValueError("Physics predictions have changed")
    daily = read_jsonl(daily_path)
    physics = read_jsonl(physics_path)
    if len(daily) != len(physics):
        raise ValueError("Daily/physics row count mismatch")
    daily_by_key = {key(row): row for row in daily}
    physics_by_key = {key(row): row for row in physics}
    if len(daily_by_key) != len(daily) or set(daily_by_key) != set(physics_by_key):
        raise ValueError("Daily/physics row keys mismatch")

    prompt = PROMPT.read_text().strip()
    run_config = {
        "model": MODEL,
        "prompt_sha256": sha256(PROMPT),
        "daily_sha256": sha256(daily_path),
        "physics_sha256": sha256(physics_path),
        "physics_manifest_sha256": sha256(manifest_path),
        "schema": SCHEMA,
        "options": OPTIONS,
        "think": False,
        "stream": False,
        "split": split,
    }
    run_id = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
    output_path = OUT / "ollama_zero_shot_valid.jsonl"
    config_path = OUT / "ollama_zero_shot_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != {**run_config, "run_id": run_id}:
            raise ValueError("Existing run uses another model, prompt or dataset")
    else:
        if output_path.exists():
            raise ValueError("Prediction file exists without its run configuration")
        config_path.write_text(json.dumps({**run_config, "run_id": run_id}, ensure_ascii=False, indent=2) + "\n")

    completed = read_jsonl(output_path) if output_path.exists() else []
    if len({key(row) for row in completed}) != len(completed):
        raise ValueError("Duplicate predictions in existing output")
    if any(row.get("run_id") != run_id for row in completed):
        raise ValueError("Prediction run ID mismatch")
    done = {key(row) for row in completed}
    expected = daily[: args.limit] if args.limit else daily
    if not done.issubset(set(map(key, daily))):
        raise ValueError("Unexpected prediction row")

    with output_path.open("a") as file:
        for row in expected:
            station_day = key(row)
            if station_day in done:
                continue
            anchor = physics_by_key[station_day]["calibrated_physics_mwh"] / CAPACITY_MW[row["station"]]
            features = make_input(row, physics_by_key[station_day])
            result = invoke(prompt, features)
            forecast_hours, fallback = decode_forecast(result.get("response", ""), anchor)
            record = {
                "run_id": run_id,
                "station": row["station"],
                "target_date": row["target_date"],
                "forecast_full_load_hours": round(forecast_hours, 6),
                "forecast_energy_mwh": round(forecast_hours * CAPACITY_MW[row["station"]], 6),
                "fallback": fallback,
                "raw_response": result.get("response", ""),
                "client_elapsed_seconds": result["client_elapsed_seconds"],
                "prompt_eval_count": result.get("prompt_eval_count"),
                "eval_count": result.get("eval_count"),
                "done_reason": result.get("done_reason"),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            done.add(station_day)
            print(
                f"{len(done)}/{len(daily)} {row['station']} {row['target_date']} "
                f"forecast={forecast_hours:.4f}h elapsed={result['client_elapsed_seconds']:.1f}s"
                + (f" fallback={fallback}" if fallback else ""),
                flush=True,
            )

    predictions = read_jsonl(output_path)
    selected = [row for row in predictions if key(row) in set(map(key, expected))]
    selected.sort(key=lambda row: (row["target_date"], row["station"]))
    summary = {
        "run_id": run_id,
        "completed": len(predictions),
        "validation_total": len(daily),
        "selected_rows": len(selected),
        "fallback_rows": sum(row["fallback"] is not None for row in selected),
        "mean_client_elapsed_seconds": round(float(np.mean([row["client_elapsed_seconds"] for row in selected])), 3),
        "scores": score(selected, daily_by_key, physics_by_key),
        "per_station": {
            station: score([row for row in selected if row["station"] == station], daily_by_key, physics_by_key)
            for station in CAPACITY_MW if any(row["station"] == station for row in selected)
        },
        "paired_diagnostics": paired_diagnostics(selected, daily_by_key, physics_by_key),
        "output_sha256": sha256(output_path),
    }
    (OUT / "ollama_zero_shot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
