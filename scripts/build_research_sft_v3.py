"""Convert the fixed v2 numeric residual task into balanced A/B/C SFT labels."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from audit_research_dataset import ROOT


OUT = ROOT / "data/processed/research"
SOURCE_DIR = OUT / "sft_residual_v2"
SFT_DIR = OUT / "sft_residual_v3"
PROMPT = ROOT / "prompts/research_pv_residual_class_v3.md"
PROMPT_SHA256 = "c0b9cc042eb7ef691b4146100462a0b871a0bcf8049a924ddae5fb0c76827b16"
SOURCE_SHA256 = {
    "manifest.json": "af247b7a720bb438cfcd28ac191b52e59d2854f69698653c89bf989110f6837b",
    "train.jsonl": "c5d7b572f5b7bc91922aa15729f103fe3fa72bb83d98f025489f4fa2877ac542",
    "valid.jsonl": "cdef18c9caa2da110f9e0cea6ff61b1fb59557e6bcbbb8c762dd08554041108c",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def residual(row: dict) -> float:
    return float(json.loads(row["messages"][2]["content"])["correction_full_load_hours"])


def category(value: float, low: float, high: float) -> str:
    return "A" if value <= low else "B" if value <= high else "C"


def main() -> None:
    for name, wanted in SOURCE_SHA256.items():
        if sha256(SOURCE_DIR / name) != wanted:
            raise ValueError(f"V2 SFT source changed: {name}")
    if sha256(PROMPT) != PROMPT_SHA256:
        raise ValueError("V3 class prompt changed")
    source = {split: read_jsonl(SOURCE_DIR / f"{split}.jsonl") for split in ("train", "valid")}
    if len(source["train"]) != 890 or len(source["valid"]) != 184:
        raise ValueError("Unexpected train/validation count")
    train_residuals = np.array([residual(row) for row in source["train"]])
    low, high = (float(x) for x in np.quantile(train_residuals, [1 / 3, 2 / 3]))
    if abs(low - (-0.39436666666666664)) > 1e-9 or abs(high - 0.3249666666666667) > 1e-9:
        raise ValueError("Class boundaries changed from preregistered train quantiles")
    prompt = PROMPT.read_text().strip()
    if "-0.3944" not in prompt or "0.3250" not in prompt:
        raise ValueError("Prompt class boundaries do not match training quantiles")

    train_categories = np.array([category(value, low, high) for value in train_residuals])
    medians = {
        label: round(float(np.median(train_residuals[train_categories == label])), 6)
        for label in ("A", "B", "C")
    }
    if not medians["A"] < medians["B"] < medians["C"]:
        raise ValueError("Invalid class representatives")
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": "predict one-token residual class; probability-weighted train class medians give numeric correction",
        "source_v2_sha256": SOURCE_SHA256,
        "prompt_sha256": sha256(PROMPT),
        "class_boundary_hours": {"A_to_B": low, "B_to_C": high},
        "class_representative_median_hours": medians,
        "splits": {},
        "test_status": "no test SFT file built",
    }
    for split in ("train", "valid"):
        examples = []
        labels = []
        for row in source[split]:
            messages = row["messages"]
            if [message["role"] for message in messages] != ["system", "user", "assistant"]:
                raise ValueError("Unexpected source SFT roles")
            features = json.loads(messages[1]["content"])
            if "target" in features or any(name.startswith("lmd_") for name in features):
                raise ValueError("Target-day observations entered prompt")
            label = category(residual(row), low, high)
            examples.append({"messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": messages[1]["content"]},
                {"role": "assistant", "content": label},
            ]})
            labels.append(label)
        path = SFT_DIR / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples))
        manifest["splits"][split] = {
            "rows": len(examples),
            "sha256": sha256(path),
            "class_counts": dict(Counter(labels)),
        }
    (SFT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
