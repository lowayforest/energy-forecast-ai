"""Forecast residuals from A/B/C token probabilities on sealed-off validation only."""

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
from mlx_lm import load

from audit_research_dataset import CAPACITY_MW, ROOT
from evaluate_research_mlx import metrics


OUT = ROOT / "data/processed/research"
MODEL_DIR = ROOT / "data/cache/research/mlx_qwen35_9b_4bit"
ADAPTER_DIR = OUT / "adapters/qwen35_9b_residual_v3"
SFT_DIR = OUT / "sft_residual_v3"
DAILY_VALID = OUT / "daily_valid.jsonl"
CONDITIONAL_VALID = OUT / "conditional_valid.jsonl"
MODEL_REVISION = "8b2b98c00a6b4d291155e4890773ca8f769aee53"
WEIGHT_SHA256 = {
    "model-00001-of-00002.safetensors": "a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd",
    "model-00002-of-00002.safetensors": "b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d",
}
DAILY_VALID_SHA256 = "9b22a9b6c07bb0727ab18d7a616c96b0d1e59887b7e9fd9492e3941edc3affe1"
CONDITIONAL_VALID_SHA256 = "932af3a25e99f6aaf01d544e4259bc42c102bd7be33157d57381753b706b5501"
PROMPT_SHA256 = "c0b9cc042eb7ef691b4146100462a0b871a0bcf8049a924ddae5fb0c76827b16"
LABELS = ("A", "B", "C")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("base", "adapter"), required=True)
    parser.add_argument("--limit", type=int, help="Generate first N examples only; no metrics until all 184")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if version("mlx-lm") != "0.32.2":
        raise RuntimeError("Expected mlx-lm 0.32.2")
    for name, expected in WEIGHT_SHA256.items():
        if sha256(MODEL_DIR / name) != expected:
            raise ValueError(f"Model weight changed: {name}")
    for path, expected in ((DAILY_VALID, DAILY_VALID_SHA256), (CONDITIONAL_VALID, CONDITIONAL_VALID_SHA256)):
        if sha256(path) != expected:
            raise ValueError(f"Input hash mismatch: {path}")

    manifest_path = SFT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["test_status"] != "no test SFT file built":
        raise ValueError("Unexpected SFT test status")
    if manifest["prompt_sha256"] != PROMPT_SHA256:
        raise ValueError("V3 prompt hash mismatch")
    sft_valid = SFT_DIR / "valid.jsonl"
    if sha256(sft_valid) != manifest["splits"]["valid"]["sha256"]:
        raise ValueError("V3 SFT validation hash mismatch")
    medians = manifest["class_representative_median_hours"]
    if set(medians) != set(LABELS) or not medians["A"] < medians["B"] < medians["C"]:
        raise ValueError("Invalid class representatives")

    adapter_path = ADAPTER_DIR if args.variant == "adapter" else None
    adapter_sha256 = sha256(adapter_path / "adapters.safetensors") if adapter_path else None
    provenance_sha256 = None
    if adapter_path:
        provenance_path = adapter_path / "training_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        if provenance["status"] != "890 training updates completed successfully":
            raise ValueError("V3 training did not complete")
        if provenance["sft_manifest_sha256"] != sha256(manifest_path):
            raise ValueError("V3 adapter was trained on another SFT file")
        if provenance["adapter_sha256"] != adapter_sha256:
            raise ValueError("V3 adapter hash mismatch")
        if provenance["adapter_config_sha256"] != sha256(adapter_path / "adapter_config.json"):
            raise ValueError("V3 adapter configuration changed")
        provenance_sha256 = sha256(provenance_path)

    model, tokenizer = load(str(MODEL_DIR), adapter_path=str(adapter_path) if adapter_path else None)
    model.eval()
    mx.random.seed(42)
    class_token_ids = []
    for label in LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Class {label} is not a single token")
        class_token_ids.append(ids[0])
    config = {
        "experiment": "v3_residual_class",
        "variant": args.variant,
        "model_revision": MODEL_REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "daily_valid_sha256": DAILY_VALID_SHA256,
        "conditional_valid_sha256": CONDITIONAL_VALID_SHA256,
        "sft_manifest_sha256": sha256(manifest_path),
        "sft_valid_sha256": sha256(sft_valid),
        "prompt_sha256": PROMPT_SHA256,
        "class_token_ids": dict(zip(LABELS, class_token_ids)),
        "class_representative_median_hours": medians,
        "adapter_sha256": adapter_sha256,
        "training_provenance_sha256": provenance_sha256,
        "mlx_lm_version": version("mlx-lm"),
        "decoding": "softmax over A/B/C next-token logits; probability-weighted training class medians",
        "invalid_output_policy": "zero residual (conditional physical forecast)",
    }
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    prefix = f"mlx_v3_{args.variant}_valid"
    config_path = OUT / f"{prefix}_config.json"
    prediction_path = OUT / f"{prefix}.jsonl"
    expected_config = {**config, "run_id": run_id}
    if config_path.exists():
        if json.loads(config_path.read_text()) != expected_config:
            raise ValueError("Existing v3 run has different inputs or model")
    else:
        if prediction_path.exists():
            raise ValueError("Predictions exist without v3 run configuration")
        config_path.write_text(json.dumps(expected_config, ensure_ascii=False, indent=2) + "\n")

    daily = read_jsonl(DAILY_VALID)
    conditional = {key(row): row for row in read_jsonl(CONDITIONAL_VALID)}
    sft = read_jsonl(sft_valid)
    if len(daily) != 184 or len(conditional) != 184 or len(sft) != 184:
        raise ValueError("Unexpected validation count")
    actual = {key(row): row for row in daily}
    if len(actual) != 184:
        raise ValueError("Duplicate validation labels")
    examples = []
    for record in sft:
        messages = record["messages"]
        if [message["role"] for message in messages] != ["system", "user", "assistant"]:
            raise ValueError("Unexpected SFT roles")
        features = json.loads(messages[1]["content"])
        station_day = features["station"], features["target_date"]
        if station_day not in actual or station_day not in conditional:
            raise ValueError("SFT station/date mismatch")
        if abs(features["physics_full_load_hours"] - conditional[station_day]["conditioned_physics_full_load_hours"]) > 1e-5:
            raise ValueError("SFT anchor differs from conditional physical forecast")
        if "target" in features or any(name.startswith("lmd_") for name in features):
            raise ValueError("Target-day observation entered prompt")
        examples.append((features, messages[:2]))
    if len({(x["station"], x["target_date"]) for x, _ in examples}) != 184:
        raise ValueError("Duplicate SFT examples")

    completed = read_jsonl(prediction_path) if prediction_path.exists() else []
    if len({key(row) for row in completed}) != len(completed):
        raise ValueError("Duplicate existing predictions")
    if any(row.get("run_id") != run_id for row in completed):
        raise ValueError("Existing predictions use another run")
    selected = examples[: args.limit] if args.limit else examples
    completed_keys = {key(row) for row in completed}
    if not completed_keys.issubset({(x["station"], x["target_date"]) for x, _ in examples}):
        raise ValueError("Unknown existing prediction")
    with prediction_path.open("a") as file:
        for features, messages in selected:
            station_day = features["station"], features["target_date"]
            if station_day in completed_keys:
                continue
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            started = time.monotonic()
            logits = model(mx.array([token_ids]))
            if hasattr(logits, "logits"):
                logits = logits.logits
            elif isinstance(logits, tuple):
                logits = logits[0]
            class_logits = np.array([float(logits[0, -1, token_id]) for token_id in class_token_ids])
            if not np.isfinite(class_logits).all():
                probabilities = np.array([0.0, 1.0, 0.0])
                correction, fallback = 0.0, "nonfinite_class_logits"
                allowed_token_mass = 0.0
            else:
                weights = np.exp(class_logits - class_logits.max())
                probabilities = weights / weights.sum()
                correction = float(sum(probabilities[i] * medians[label] for i, label in enumerate(LABELS)))
                fallback = None
                full_logsumexp = float(mx.logsumexp(logits[0, -1, :]))
                if not math.isfinite(full_logsumexp):
                    correction, fallback, allowed_token_mass = 0.0, "nonfinite_full_logits", 0.0
                else:
                    allowed_token_mass = float(np.clip(
                        np.exp(class_logits.max() + np.log(weights.sum()) - full_logsumexp), 0, 1
                    ))
            anchor = features["physics_full_load_hours"]
            forecast_h = float(np.clip(anchor + correction, 0, 24))
            record = {
                "run_id": run_id,
                "station": features["station"],
                "target_date": features["target_date"],
                "class_probabilities": {label: round(float(probabilities[i]), 6) for i, label in enumerate(LABELS)},
                "allowed_token_mass": round(allowed_token_mass, 6),
                "predicted_residual_hours": round(correction, 6),
                "forecast_full_load_hours": round(forecast_h, 6),
                "forecast_energy_mwh": round(forecast_h * CAPACITY_MW[features["station"]], 6),
                "fallback": fallback,
                "inference_seconds": round(time.monotonic() - started, 3),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            completed_keys.add(station_day)
            print(f"v3/{args.variant} {len(completed_keys)}/184 {station_day} correction={correction:.4f}h", flush=True)

    predictions = read_jsonl(prediction_path)
    if len(predictions) == 184:
        predictions.sort(key=lambda row: (row["target_date"], row["station"]))
        summary = {
            "run_id": run_id,
            "variant": args.variant,
            "rows": len(predictions),
            "fallback_rows": sum(row["fallback"] is not None for row in predictions),
            "mean_inference_seconds": round(float(np.mean([row["inference_seconds"] for row in predictions])), 3),
            "mean_allowed_token_mass": round(float(np.mean([row["allowed_token_mass"] for row in predictions])), 4),
            "overall": metrics(predictions, actual),
            "per_station": {
                station: metrics([row for row in predictions if row["station"] == station], actual)
                for station in CAPACITY_MW
            },
            "prediction_sha256": sha256(prediction_path),
        }
        (OUT / f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Pilot complete: {len(predictions)}/184; metrics withheld", flush=True)


if __name__ == "__main__":
    main()
