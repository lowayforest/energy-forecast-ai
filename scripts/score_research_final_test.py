"""Score the frozen test forecasts once, including the clean 58-date subset."""

from __future__ import annotations

import hashlib
import json

from audit_research_dataset import CAPACITY_MW, ROOT
from compare_research_mlx import metrics, paired_improvement


OUT = ROOT / "data/processed/research"
DAILY_SHA256 = "7fd4970581a695f2ef497017aab5065e7e0d5db3e9e737b5f971c060b97a14fe"
PHYSICS_SHA256 = "097c7bb40d52609e3223380ebc84c3320f5edd542f36ffd50a6864dddc121d57"
EXPOSED_DATE = "2019-04-16"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row):
    return row["station"], row["target_date"]


def main() -> None:
    if sha256(OUT / "daily_test.jsonl") != DAILY_SHA256 or sha256(OUT / "physics_test.jsonl") != PHYSICS_SHA256:
        raise ValueError("Test source changed")
    input_manifest = json.loads((OUT / "v3_final_test_inputs_manifest.json").read_text())
    if sha256(OUT / "conditional_test.jsonl") != input_manifest["conditional_test_sha256"]:
        raise ValueError("Conditional test predictions changed")
    daily = read_jsonl(OUT / "daily_test.jsonl")
    actual = {key(r): r["target"]["actual_full_load_hours"] for r in daily}
    if len(daily) != 236 or len(actual) != 236 or len({day for _, day in actual}) != 59:
        raise ValueError("Unexpected test labels")
    sources = {}
    physics = read_jsonl(OUT / "physics_test.jsonl")
    sources["physical"] = {key(r): {"station": r["station"], "target_date": r["target_date"],
        "forecast_full_load_hours": r["calibrated_physics_mwh"] / CAPACITY_MW[r["station"]]} for r in physics}
    conditional = read_jsonl(OUT / "conditional_test.jsonl")
    sources["conditional"] = {key(r): {"station": r["station"], "target_date": r["target_date"],
        "forecast_full_load_hours": r["conditioned_physics_full_load_hours"]} for r in conditional}
    hashes = {"daily_test": DAILY_SHA256, "physics_test": PHYSICS_SHA256,
              "conditional_test": sha256(OUT / "conditional_test.jsonl")}
    for variant in ("base", "adapter"):
        prefix = f"mlx_v3_{variant}_test"
        path = OUT / f"{prefix}.jsonl"
        summary = json.loads((OUT / f"{prefix}_summary.json").read_text())
        if summary["rows"] != 236 or sha256(path) != summary["prediction_sha256"]:
            raise ValueError(f"{variant}: incomplete test predictions")
        rows = read_jsonl(path)
        if len(rows) != 236 or len({key(r) for r in rows}) != 236 or any("target" in r or "actual" in r for r in rows):
            raise ValueError(f"{variant}: label or duplicate in test predictions")
        sources[variant] = {key(r): r for r in rows}
        hashes[variant] = sha256(path)
    if any(set(rows) != set(actual) for rows in sources.values()):
        raise ValueError("Test predictions do not align")

    report = {"split": "test", "input_sha256": hashes,
              "exposed_before_freeze": {"station": "S-1", "date": EXPOSED_DATE,
                 "handling": "First date excluded from primary; full test shown separately."},
              "sections": {}}
    for name, selected in (
        ("primary_clean_58_dates", {k for k in actual if k[1] != EXPOSED_DATE}),
        ("secondary_full_59_dates", set(actual)),
    ):
        if len(selected) != (232 if name.startswith("primary") else 236):
            raise ValueError("Unexpected test subset size")
        truth = {k: actual[k] for k in selected}
        data = {method: {k: row for k, row in rows.items() if k in selected}
                for method, rows in sources.items()}
        report["sections"][name] = {
            "rows": len(selected), "date_blocks": len({k[1] for k in selected}),
            "metrics": {method: metrics(list(rows.values()), truth) for method, rows in data.items()},
            "per_station": {station: {
                method: metrics([r for k, r in rows.items() if k[0] == station], truth)
                for method, rows in data.items()}
                for station in CAPACITY_MW},
            "paired": {
                "adapter_vs_conditional": paired_improvement(data["conditional"], data["adapter"], truth),
                "adapter_vs_base": paired_improvement(data["base"], data["adapter"], truth),
                "adapter_vs_physical": paired_improvement(data["physical"], data["adapter"], truth),
            },
        }
    path = OUT / "v3_final_test_comparison.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
