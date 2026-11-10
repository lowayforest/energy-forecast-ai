"""Run MLX-LM LoRA with loss on the A/B/C token only."""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from mlx_lm import lora as lora_cli
from mlx_lm.tuner.datasets import ChatDataset
from mlx_lm.utils import load_tokenizer

from audit_research_dataset import ROOT


OUT = ROOT / "data/processed/research"
DATA = OUT / "sft_residual_v3"
MODEL = ROOT / "data/cache/research/mlx_qwen35_9b_4bit"
MANIFEST_SHA256 = "4da10cf8c060aa5e2465db1eac8a62e69055f83f761ea71ebdab45a19ac96f51"
LABELS = ("A", "B", "C")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def class_only_process(self: ChatDataset, item: dict) -> tuple[list[int], int]:
    messages = item[self.chat_key]
    if not self.mask_prompt or [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Class-only training requires masked 3-message chat data")
    label = messages[-1]["content"]
    if label not in LABELS:
        raise ValueError("Expected exactly one A/B/C class")
    prefix = self.tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, enable_thinking=False,
        return_dict=False,
    )
    full = self.tokenizer.apply_chat_template(
        messages, enable_thinking=False, return_dict=False,
    )
    label_ids = self.tokenizer.encode(label, add_special_tokens=False)
    if len(label_ids) != 1 or full[: len(prefix)] != prefix or full[len(prefix)] != label_ids[0]:
        raise ValueError("Assistant class is not the first token after the exact inference prefix")
    tokens = prefix + label_ids
    if len(tokens) != len(prefix) + 1 or len(tokens) > 400:
        raise ValueError("Class-only target or memory guard violated")
    return tokens, len(prefix)


ChatDataset.process = class_only_process


def audit() -> None:
    manifest_path = DATA / "manifest.json"
    if sha256(manifest_path) != MANIFEST_SHA256:
        raise ValueError("V3 training manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest["test_status"] != "no test SFT file built":
        raise ValueError("Test samples entered training SFT")
    tokenizer = load_tokenizer(str(MODEL))
    report = {
        "objective": "exactly one A/B/C token after enable_thinking=False inference prefix",
        "sft_manifest_sha256": MANIFEST_SHA256,
        "splits": {},
    }
    for split, expected in (("train", 890), ("valid", 184)):
        rows = [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines()]
        dataset = ChatDataset(rows, tokenizer, mask_prompt=True)
        processed = [dataset.process(row) for row in rows]
        if len(rows) != expected or any(len(tokens) - offset != 1 for tokens, offset in processed):
            raise ValueError("Incorrect class-only training count")
        report["splits"][split] = {
            "rows": len(rows),
            "sft_sha256": sha256(DATA / f"{split}.jsonl"),
            "max_tokens": max(len(tokens) for tokens, _ in processed),
            "min_supervised_tokens": 1,
            "max_supervised_tokens": 1,
            "class_counts": dict(Counter(row["messages"][-1]["content"] for row in rows)),
        }
    path = OUT / "v4_class_only_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    del tokenizer
    gc.collect()


if __name__ == "__main__":
    audit()
    if "--audit-only" not in sys.argv:
        lora_cli.main()
