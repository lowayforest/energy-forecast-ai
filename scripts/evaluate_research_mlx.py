"""Evaluate base/adapted MLX residual forecasts on the fixed validation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from audit_research_dataset import CAPACITY_MW, ROOT


OUT = ROOT / "data/processed/research"
MODEL_DIR = ROOT / "data/cache/research/mlx_qwen35_9b_4bit"
ADAPTER_DIR = OUT / "adapters/qwen35_9b_residual_v1"
SFT_VALID = OUT / "sft_residual_v1/valid.jsonl"
DAILY_VALID = OUT / "daily_valid.jsonl"
PHYSICS_VALID = OUT / "physics_valid.jsonl"
MODEL_REVISION = "8b2b98c00a6b4d291155e4890773ca8f769aee53"
WEIGHT_SHA256 = {
    "model-00001-of-00002.safetensors": "a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd",
    "model-00002-of-00002.safetensors": "b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d",
}
EXPECTED_SFT_SHA256 = "aa326ce05e5d8631e80a5ef39d3bfca7b9dea471f53a025c1a67d8c0d0ac3da5"
EXPECTED_DAILY_SHA256 = "9b22a9b6c07bb0727ab18d7a616c96b0d1e59887b7e9fd9492e3941edc3affe1"
EXPECTED_PHYSICS_SHA256 = "62aedeb0d51d9d92604f907fdbb37cf4a66c396fc97a00cf5f6785678b4a1881"
MAX_TOKENS = 64
MAX_ABS_RESIDUAL_HOURS = 24.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row: dict) -> tuple[str, str]:
    return row["station"], row["target_date"]


def parse_response(response: str) -> tuple[float, str | None]:
    try:
        decoded = json.loads(response.strip())
        if not isinstance(decoded, dict) or set(decoded) != {"correction_full_load_hours"}:
            raise ValueError("wrong JSON shape")
        value = decoded["correction_full_load_hours"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("not a finite number")
        if abs(value) > MAX_ABS_RESIDUAL_HOURS:
            raise ValueError("residual outside ±24 h")
        return float(value), None
    except (ValueError, TypeError):
        return 0.0, "invalid_model_output"


def metrics(rows: list[dict], actual_by_key: dict) -> dict:
    actual = np.array([actual_by_key[key(row)]["target"]["actual_full_load_hours"] for row in rows])
    predicted = np.array([row["forecast_full_load_hours"] for row in rows])
    error = predicted - actual
    return {
        "rows": len(rows),
        "mae_hours": round(float(np.mean(np.abs(error))), 4),
        "rmse_hours": round(float(np.sqrt(np.mean(error ** 2))), 4),
        "bias_hours": round(float(np.mean(error)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("base", "adapter"), required=True)
    parser.add_argument("--limit", type=int, help="Only generate first N rows; no metrics until complete")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if version("mlx-lm") != "0.32.2":
        raise RuntimeError("Expected mlx-lm 0.32.2")
    for name, expected in WEIGHT_SHA256.items():
        if sha256(MODEL_DIR / name) != expected:
            raise ValueError(f"Model weight hash mismatch: {name}")
    for path, expected in (
        (SFT_VALID, EXPECTED_SFT_SHA256),
        (DAILY_VALID, EXPECTED_DAILY_SHA256),
        (PHYSICS_VALID, EXPECTED_PHYSICS_SHA256),
    ):
        if sha256(path) != expected:
            raise ValueError(f"Input hash mismatch: {path}")

    adapter_path = ADAPTER_DIR if args.variant == "adapter" else None
    adapter_hash = sha256(adapter_path / "adapters.safetensors") if adapter_path else None
    config = {
        "variant": args.variant,
        "model_revision": MODEL_REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "sft_valid_sha256": EXPECTED_SFT_SHA256,
        "daily_valid_sha256": EXPECTED_DAILY_SHA256,
        "physics_valid_sha256": EXPECTED_PHYSICS_SHA256,
        "adapter_sha256": adapter_hash,
        "mlx_lm_version": version("mlx-lm"),
        "generation": {"temperature": 0, "max_tokens": MAX_TOKENS, "enable_thinking": False},
        "invalid_output_policy": "fallback to zero residual (calibrated physical prediction)",
    }
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = OUT / f"mlx_{args.variant}_valid_config.json"
    prediction_path = OUT / f"mlx_{args.variant}_valid.jsonl"
    expected_config = {**config, "run_id": run_id}
    if config_path.exists():
        if json.loads(config_path.read_text()) != expected_config:
            raise ValueError("Existing run has another model, adapter or input")
    else:
        if prediction_path.exists():
            raise ValueError("Predictions exist without run configuration")
        config_path.write_text(json.dumps(expected_config, ensure_ascii=False, indent=2) + "\n")

    daily = read_jsonl(DAILY_VALID)
    physics = {key(row): row for row in read_jsonl(PHYSICS_VALID)}
    sft = read_jsonl(SFT_VALID)
    if len(daily) != len(sft) or len(physics) != len(daily):
        raise ValueError("Validation row count mismatch")
    actual_by_key = {key(row): row for row in daily}
    examples = []
    for record in sft:
        messages = record["messages"]
        if [message["role"] for message in messages] != ["system", "user", "assistant"]:
            raise ValueError("Unexpected SFT message order")
        features = json.loads(messages[1]["content"])
        station_day = features["station"], features["target_date"]
        if station_day not in actual_by_key or station_day not in physics:
            raise ValueError("SFT/validation key mismatch")
        if "target" in features or any(name.startswith("lmd_") for name in features):
            raise ValueError("Target-day observations entered prompt")
        examples.append((features, messages[:2]))  # Never send assistant label to generation.
    if len({(features["station"], features["target_date"]) for features, _ in examples}) != len(examples):
        raise ValueError("Duplicate SFT examples")

    completed = read_jsonl(prediction_path) if prediction_path.exists() else []
    if len({key(row) for row in completed}) != len(completed):
        raise ValueError("Duplicate existing predictions")
    if any(row.get("run_id") != run_id for row in completed):
        raise ValueError("Existing predictions use another run ID")
    completed_keys = {key(row) for row in completed}
    selected = examples[: args.limit] if args.limit else examples
    if not completed_keys.issubset({(features["station"], features["target_date"]) for features, _ in examples}):
        raise ValueError("Unknown existing prediction key")

    if len(completed_keys) < len(selected):
        model, tokenizer = load(str(MODEL_DIR), adapter_path=str(adapter_path) if adapter_path else None)
        mx.random.seed(42)
        sampler = make_sampler(temp=0)
        with prediction_path.open("a") as file:
            for features, messages in selected:
                station_day = features["station"], features["target_date"]
                if station_day in completed_keys:
                    continue
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                started = time.monotonic()
                response = generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler)
                elapsed = round(time.monotonic() - started, 3)
                residual, fallback = parse_response(response)
                physical_h = features["physics_full_load_hours"]
                forecast_h = float(np.clip(physical_h + residual, 0, 24))
                record = {
                    "run_id": run_id,
                    "station": features["station"],
                    "target_date": features["target_date"],
                    "predicted_residual_hours": round(residual, 6),
                    "forecast_full_load_hours": round(forecast_h, 6),
                    "forecast_energy_mwh": round(forecast_h * CAPACITY_MW[features["station"]], 6),
                    "fallback": fallback,
                    "raw_response": response,
                    "generation_seconds": elapsed,
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
                completed_keys.add(station_day)
                print(
                    f"{args.variant} {len(completed_keys)}/{len(examples)} "
                    f"{features['station']} {features['target_date']} residual={residual:.4f}h "
                    f"time={elapsed:.2f}s" + (f" fallback={fallback}" if fallback else ""),
                    flush=True,
                )

    predictions = read_jsonl(prediction_path) if prediction_path.exists() else []
    if len(predictions) == len(examples):
        predictions.sort(key=lambda row: (row["target_date"], row["station"]))
        summary = {
            "run_id": run_id,
            "variant": args.variant,
            "rows": len(predictions),
            "fallback_rows": sum(row["fallback"] is not None for row in predictions),
            "mean_generation_seconds": round(float(np.mean([row["generation_seconds"] for row in predictions])), 3),
            "overall": metrics(predictions, actual_by_key),
            "per_station": {
                station: metrics([row for row in predictions if row["station"] == station], actual_by_key)
                for station in CAPACITY_MW
            },
            "prediction_sha256": sha256(prediction_path),
        }
        (OUT / f"mlx_{args.variant}_valid_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Pilot complete: {len(predictions)}/{len(examples)} rows; metrics withheld", flush=True)


if __name__ == "__main__":
    main()
