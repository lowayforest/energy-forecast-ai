"""Run the frozen v3 base or adapter on label-free test prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version

import mlx.core as mx
import numpy as np
from mlx_lm import load

from audit_research_dataset import CAPACITY_MW, ROOT
from evaluate_research_class_v3 import MODEL_DIR, WEIGHT_SHA256, LABELS, sha256


OUT = ROOT / "data/processed/research"
ADAPTER_DIR = OUT / "adapters/qwen35_9b_residual_v3"
INPUT_MANIFEST = OUT / "v3_final_test_inputs_manifest.json"
ADAPTER_SHA256 = "ccfaef49cd75e4398770ab31f33a89a551b0f7610f988d27a6911985b8c8603e"
SFT_MANIFEST_SHA256 = "4da10cf8c060aa5e2465db1eac8a62e69055f83f761ea71ebdab45a19ac96f51"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("base", "adapter"), required=True)
    args = parser.parse_args()
    if version("mlx-lm") != "0.32.2":
        raise RuntimeError("Expected mlx-lm 0.32.2")
    for name, expected in WEIGHT_SHA256.items():
        if sha256(MODEL_DIR / name) != expected:
            raise ValueError(f"Model weight changed: {name}")
    if sha256(OUT / "sft_residual_v3/manifest.json") != SFT_MANIFEST_SHA256:
        raise ValueError("Training manifest changed")
    if sha256(ADAPTER_DIR / "adapters.safetensors") != ADAPTER_SHA256:
        raise ValueError("Final adapter changed")
    provenance = json.loads((ADAPTER_DIR / "training_provenance.json").read_text())
    if provenance["adapter_sha256"] != ADAPTER_SHA256 or provenance["sft_manifest_sha256"] != SFT_MANIFEST_SHA256:
        raise ValueError("Adapter provenance mismatch")
    manifest = json.loads(INPUT_MANIFEST.read_text())
    examples_path = OUT / "v3_final_test_inputs.jsonl"
    conditional_path = OUT / "conditional_test.jsonl"
    if sha256(examples_path) != manifest["test_inputs_sha256"]:
        raise ValueError("Changed test prompts")
    if sha256(conditional_path) != manifest["conditional_test_sha256"]:
        raise ValueError("Changed conditional test predictions")
    examples = read_jsonl(examples_path)
    conditional = {(r["station"], r["target_date"]): r for r in read_jsonl(conditional_path)}
    if len(examples) != 236 or len(conditional) != 236:
        raise ValueError("Incomplete test inputs")
    medians = json.loads((OUT / "sft_residual_v3/manifest.json").read_text())["class_representative_median_hours"]

    adapter_path = str(ADAPTER_DIR) if args.variant == "adapter" else None
    model, tokenizer = load(str(MODEL_DIR), adapter_path=adapter_path)
    model.eval()
    mx.random.seed(42)
    ids = [tokenizer.encode(k, add_special_tokens=False) for k in LABELS]
    if any(len(item) != 1 for item in ids):
        raise ValueError("Class labels are not single tokens")
    token_ids = [item[0] for item in ids]
    config = {
        "experiment": "v3_frozen_final_test", "variant": args.variant,
        "input_manifest_sha256": sha256(INPUT_MANIFEST),
        "adapter_sha256": ADAPTER_SHA256 if adapter_path else None,
        "model_weight_sha256": WEIGHT_SHA256,
        "class_token_ids": dict(zip(LABELS, token_ids)),
        "class_representative_median_hours": medians,
        "mlx_lm_version": version("mlx-lm"),
        "decoding": "same A/B/C restricted-softmax weighted medians as frozen validation evaluator",
    }
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    path = OUT / f"mlx_v3_{args.variant}_test.jsonl"
    config_path = OUT / f"mlx_v3_{args.variant}_test_config.json"
    expected_config = {**config, "run_id": run_id}
    if config_path.exists():
        if json.loads(config_path.read_text()) != expected_config:
            raise ValueError("Existing test run has different inputs")
    else:
        if path.exists():
            raise ValueError("Test predictions exist without matching config")
        config_path.write_text(json.dumps(expected_config, ensure_ascii=False, indent=2) + "\n")
    completed = read_jsonl(path) if path.exists() else []
    keys = {(r["station"], r["target_date"]) for r in completed}
    if len(completed) != len(keys) or any(r["run_id"] != run_id for r in completed):
        raise ValueError("Invalid existing test predictions")
    with path.open("a") as file:
        for example in examples:
            messages = example["messages"]
            if [m["role"] for m in messages] != ["system", "user"]:
                raise ValueError("Assistant label entered test prompt")
            features = json.loads(messages[1]["content"])
            station_day = features["station"], features["target_date"]
            if station_day not in conditional or "target" in features or any(k.startswith("lmd_") for k in features):
                raise ValueError("Invalid or leaked test prompt")
            if abs(features["physics_full_load_hours"] - conditional[station_day]["conditioned_physics_full_load_hours"]) > 1e-5:
                raise ValueError("Test physical anchor mismatch")
            if station_day in keys:
                continue
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(input_ids) > 512:
                raise ValueError("Test input exceeds frozen context length")
            logits = model(mx.array([input_ids]))
            if hasattr(logits, "logits"):
                logits = logits.logits
            elif isinstance(logits, tuple):
                logits = logits[0]
            class_logits = np.array([float(logits[0, -1, token_id]) for token_id in token_ids])
            if not np.isfinite(class_logits).all():
                probabilities = np.array([0.0, 1.0, 0.0])
                correction, fallback, allowed_mass = 0.0, "nonfinite_class_logits", 0.0
            else:
                weights = np.exp(class_logits - class_logits.max())
                probabilities = weights / weights.sum()
                correction = float(sum(probabilities[i] * medians[label] for i, label in enumerate(LABELS)))
                fallback = None
                full_logsumexp = float(mx.logsumexp(logits[0, -1, :]))
                if not math.isfinite(full_logsumexp):
                    correction, fallback, allowed_mass = 0.0, "nonfinite_full_logits", 0.0
                else:
                    allowed_mass = float(np.clip(np.exp(class_logits.max() + np.log(weights.sum()) - full_logsumexp), 0, 1))
            anchor = features["physics_full_load_hours"]
            forecast_h = float(np.clip(anchor + correction, 0, 24))
            record = {
                "run_id": run_id, "station": station_day[0], "target_date": station_day[1],
                "class_probabilities": {label: round(float(probabilities[i]), 6) for i, label in enumerate(LABELS)},
                "allowed_token_mass": round(allowed_mass, 6),
                "predicted_residual_hours": round(correction, 6),
                "forecast_full_load_hours": round(forecast_h, 6),
                "forecast_energy_mwh": round(forecast_h * CAPACITY_MW[station_day[0]], 6),
                "fallback": fallback,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            keys.add(station_day)
            print(f"v3/{args.variant} test {len(keys)}/236 {station_day}", flush=True)
    if len(keys) != 236 or keys != set(conditional):
        raise ValueError("Incomplete final test predictions")
    summary = {"run_id": run_id, "variant": args.variant, "rows": 236,
               "fallback_rows": sum(r["fallback"] is not None for r in read_jsonl(path)),
               "prediction_sha256": sha256(path)}
    (OUT / f"mlx_v3_{args.variant}_test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
