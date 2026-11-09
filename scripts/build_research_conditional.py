"""Fit a train-only station/GHI correction to the existing physical PV forecast."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_research_dataset import CAPACITY_MW, ROOT


OUT = ROOT / "data/processed/research"
INPUT_SHA256 = {
    "daily_train.jsonl": "9df5c2c495d3b79bb34906b105d52d7a4af871cd5a0f0a8fed2a307920d5fd80",
    "daily_valid.jsonl": "9b22a9b6c07bb0727ab18d7a616c96b0d1e59887b7e9fd9492e3941edc3affe1",
    "physics_train.jsonl": "63c3bf4972a505b6a9c97590224f52e5812b64024b30db6b469aa07e59502db4",
    "physics_valid.jsonl": "62aedeb0d51d9d92604f907fdbb37cf4a66c396fc97a00cf5f6785678b4a1881",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(name: str) -> list[dict]:
    path = OUT / name
    if sha256(path) != INPUT_SHA256[name]:
        raise ValueError(f"Input SHA-256 mismatch: {name}")
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row: dict) -> tuple[str, str]:
    return row["station"], row["target_date"]


def join(split: str) -> list[dict]:
    daily = read_rows(f"daily_{split}.jsonl")
    physical = read_rows(f"physics_{split}.jsonl")
    keyed_physical = {key(row): row for row in physical}
    if len(keyed_physical) != len(physical) or {key(row) for row in daily} != set(keyed_physical):
        raise ValueError(f"{split}: duplicate or unmatched station/date")
    if len({key(row) for row in daily}) != len(daily):
        raise ValueError(f"{split}: duplicate daily rows")
    result = []
    for row in daily:
        station = row["station"]
        if station not in CAPACITY_MW or row["split"] != split:
            raise ValueError("Unexpected station or split")
        if row["inputs"]["rated_capacity_mw"] != CAPACITY_MW[station]:
            raise ValueError("Capacity mismatch")
        if split == "train" and row["target_date"] > "2019-02-28":
            raise ValueError("Training date after cutoff")
        if split == "valid" and not "2019-03-01" <= row["target_date"] <= "2019-04-15":
            raise ValueError("Validation date outside cutoff")
        result.append({"daily": row, "physics": keyed_physical[key(row)]})
    return result


def score(rows: list[dict], predictions: list[dict]) -> dict:
    actual = np.array([row["daily"]["target"]["actual_full_load_hours"] for row in rows])
    original = np.array([
        row["physics"]["calibrated_physics_mwh"] / CAPACITY_MW[row["daily"]["station"]]
        for row in rows
    ])
    conditioned = np.array([row["conditioned_physics_full_load_hours"] for row in predictions])
    return {
        "rows": len(rows),
        "original_mae_h": round(float(np.mean(np.abs(original - actual))), 4),
        "conditioned_mae_h": round(float(np.mean(np.abs(conditioned - actual))), 4),
        "original_rmse_h": round(float(np.sqrt(np.mean((original - actual) ** 2))), 4),
        "conditioned_rmse_h": round(float(np.sqrt(np.mean((conditioned - actual) ** 2))), 4),
        "conditioned_bias_h": round(float(np.mean(conditioned - actual)), 4),
    }


def main() -> None:
    train = join("train")
    valid = join("valid")
    if len(train) != 890 or len(valid) != 184:
        raise ValueError("Unexpected train/validation size")

    threshold = float(np.quantile(
        [row["daily"]["inputs"]["nwp_ghi_kwh_m2"] for row in train], 2 / 3
    ))

    def group(row: dict) -> tuple[str, str]:
        station = row["daily"]["station"]
        ghi = row["daily"]["inputs"]["nwp_ghi_kwh_m2"]
        return station, "high" if ghi > threshold else "low"

    residuals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in train:
        station = row["daily"]["station"]
        physical_h = row["physics"]["calibrated_physics_mwh"] / CAPACITY_MW[station]
        residuals[group(row)].append(row["daily"]["target"]["actual_full_load_hours"] - physical_h)
    if set(residuals) != {(station, band) for station in CAPACITY_MW for band in ("low", "high")}:
        raise ValueError("Missing station/GHI group")
    totals = {name: (len(values), float(sum(values))) for name, values in residuals.items()}
    if any(count < 2 for count, _ in totals.values()):
        raise ValueError("Leave-one-out group must have at least two training rows")

    manifest = {
        "protocol": "station x global train GHI upper-tercile mean physical residual; train leave-one-out; valid full train",
        "threshold_nwp_ghi_kwh_m2": round(threshold, 8),
        "source_sha256": INPUT_SHA256,
        "groups": {
            station: {
                band: {
                    "train_rows": totals[station, band][0],
                    "train_mean_residual_h": round(totals[station, band][1] / totals[station, band][0], 8),
                }
                for band in ("low", "high")
            }
            for station in CAPACITY_MW
        },
        "predictions": {},
        "scores": {},
        "test_status": "not read or scored",
    }
    for split, rows in (("train", train), ("valid", valid)):
        predictions = []
        for row in rows:
            daily, physical = row["daily"], row["physics"]
            station, band = group(row)
            capacity = CAPACITY_MW[station]
            physical_h = physical["calibrated_physics_mwh"] / capacity
            count, total = totals[station, band]
            if split == "train":
                own_residual = daily["target"]["actual_full_load_hours"] - physical_h
                offset = (total - own_residual) / (count - 1)
            else:
                offset = total / count
            conditioned_h = round(float(np.clip(physical_h + offset, 0, 24)), 5)
            predictions.append({
                "station": station,
                "target_date": daily["target_date"],
                "ghi_band": band,
                "conditional_offset_hours": round(offset, 6),
                "conditioned_physics_full_load_hours": conditioned_h,
                "conditioned_physics_mwh": round(conditioned_h * capacity, 5),
            })
        path = OUT / f"conditional_{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
        manifest["predictions"][split] = {"rows": len(predictions), "sha256": sha256(path)}
        manifest["scores"][split] = {
            "all_stations": score(rows, predictions),
            "per_station": {
                station: score(
                    [row for row in rows if row["daily"]["station"] == station],
                    [row for row in predictions if row["station"] == station],
                )
                for station in CAPACITY_MW
            },
        }
    manifest_path = OUT / "conditional_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
