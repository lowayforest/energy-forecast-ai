"""Build a train-calibrated PV physics baseline without scoring the test set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pvlib
from pvlib.pvsystem import pvwatts_dc
from pvlib.temperature import ross

from audit_research_dataset import ARCHIVE, CAPACITY_MW, ROOT


OUT = ROOT / "data/processed/research"
GAMMA_PER_C = -0.004
NOCT_C = 45.0
INVERTER_EFFICIENCY = 0.96
INTERVAL_HOURS = 0.25
MIN_CALIBRATION_ENERGY_MWH_PER_MW = 0.1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_daily(split: str) -> list[dict]:
    path = OUT / f"daily_{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if split == "test":
        # Keep the held-out targets inaccessible to this baseline run.
        rows = [{key: value for key, value in row.items() if key != "target"} for row in rows]
    return rows


def raw_daily_energy(source: ZipFile, station: str) -> dict[str, float]:
    with source.open(f"All_dataset/{station}.csv") as file:
        frame = pd.read_csv(file, usecols=["Time", "nwp_globalirrad", "nwp_temperature"])
    local_day = pd.to_datetime(frame["Time"]).dt.strftime("%Y-%m-%d")
    ghi = frame["nwp_globalirrad"].to_numpy(dtype=float)
    air = frame["nwp_temperature"].to_numpy(dtype=float)
    if not (np.isfinite(ghi).all() and np.isfinite(air).all()):
        raise ValueError(f"{station}: nonfinite forecast weather")
    if (ghi < 0).any():
        raise ValueError(f"{station}: negative forecast GHI")

    # GHI is a documented proxy for the unavailable plane-of-array irradiance.
    cell_c = ross(ghi, air, noct=NOCT_C)
    dc_w = pvwatts_dc(
        ghi, cell_c, pdc0=CAPACITY_MW[station] * 1_000_000,
        gamma_pdc=GAMMA_PER_C,
    )
    raw_ac_mw = np.clip(dc_w * INVERTER_EFFICIENCY / 1_000_000, 0, CAPACITY_MW[station])
    raw_mwh = pd.Series(raw_ac_mw * INTERVAL_HOURS).groupby(local_day).sum()
    return {str(day): float(energy) for day, energy in raw_mwh.items()}


def metrics(rows: list[dict], field: str) -> dict:
    actual = np.array([row["target"]["actual_energy_mwh"] for row in rows], dtype=float)
    predicted = np.array([row[field] for row in rows], dtype=float)
    capacities = np.array([CAPACITY_MW[row["station"]] for row in rows], dtype=float)
    error_mwh = predicted - actual
    error_hours = error_mwh / capacities
    return {
        "rows": len(rows),
        "mae_mwh": round(float(np.mean(np.abs(error_mwh))), 4),
        "rmse_mwh": round(float(np.sqrt(np.mean(error_mwh ** 2))), 4),
        "mae_full_load_hours": round(float(np.mean(np.abs(error_hours))), 4),
        "rmse_full_load_hours": round(float(np.sqrt(np.mean(error_hours ** 2))), 4),
        "bias_full_load_hours": round(float(np.mean(error_hours)), 4),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily_manifest_path = OUT / "daily_manifest.json"
    daily_manifest = json.loads(daily_manifest_path.read_text())
    if daily_manifest["source_sha256"] != sha256(ARCHIVE):
        raise ValueError("Source archive changed after daily dataset construction")

    splits = {split: read_daily(split) for split in ("train", "valid", "test")}
    for split, rows in splits.items():
        if sha256(OUT / f"daily_{split}.jsonl") != daily_manifest["splits"][split]["sha256"]:
            raise ValueError(f"{split}: daily dataset changed")
        if len(rows) != daily_manifest["splits"][split]["rows"]:
            raise ValueError(f"{split}: daily row count changed")

    with ZipFile(ARCHIVE) as source:
        raw_by_station = {station: raw_daily_energy(source, station) for station in CAPACITY_MW}

    for rows in splits.values():
        for row in rows:
            row["raw_physics_mwh"] = raw_by_station[row["station"]][row["target_date"]]

    calibration = {}
    for station, capacity in CAPACITY_MW.items():
        train_rows = [row for row in splits["train"] if row["station"] == station]
        eligible = [
            row for row in train_rows
            if row["raw_physics_mwh"] >= MIN_CALIBRATION_ENERGY_MWH_PER_MW * capacity
        ]
        if not eligible:
            raise ValueError(f"{station}: no valid calibration days")
        ratios = [
            row["target"]["actual_energy_mwh"] / row["raw_physics_mwh"]
            for row in eligible
        ]
        scale = float(np.median(ratios))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{station}: invalid training scale {scale}")
        calibration[station] = {
            "scale": round(scale, 8),
            "train_days": len(train_rows),
            "eligible_train_days": len(eligible),
        }

    for rows in splits.values():
        for row in rows:
            station = row["station"]
            row["calibrated_physics_mwh"] = min(
                row["raw_physics_mwh"] * calibration[station]["scale"],
                CAPACITY_MW[station] * 24,
            )

    output = {}
    for split, rows in splits.items():
        path = OUT / f"physics_{split}.jsonl"
        # The predictions are features. Never copy target labels into these files.
        predictions = [
            {
                "station": row["station"],
                "target_date": row["target_date"],
                "raw_physics_mwh": round(row["raw_physics_mwh"], 5),
                "calibrated_physics_mwh": round(row["calibrated_physics_mwh"], 5),
            }
            for row in rows
        ]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
        output[split] = {"rows": len(rows), "sha256": sha256(path)}

    scores = {}
    for split in ("train", "valid"):
        rows = splits[split]
        scores[split] = {
            "all_stations": {
                "raw": metrics(rows, "raw_physics_mwh"),
                "calibrated": metrics(rows, "calibrated_physics_mwh"),
            },
            "per_station": {
                station: {
                    "raw": metrics([r for r in rows if r["station"] == station], "raw_physics_mwh"),
                    "calibrated": metrics(
                        [r for r in rows if r["station"] == station], "calibrated_physics_mwh"
                    ),
                }
                for station in CAPACITY_MW
            },
        }

    manifest = {
        "source_sha256": sha256(ARCHIVE),
        "daily_manifest_sha256": sha256(daily_manifest_path),
        "pvlib_version": pvlib.__version__,
        "model": "Ross cell temperature + PVWatts DC + fixed-efficiency AC approximation",
        "irradiance_proxy": "nwp_globalirrad (GHI), not measured/estimated plane-of-array irradiance",
        "assumptions": {
            "noct_c": NOCT_C,
            "gamma_pdc_per_c": GAMMA_PER_C,
            "inverter_efficiency": INVERTER_EFFICIENCY,
            "interval_hours": INTERVAL_HOURS,
            "ac_power_cap_mw": CAPACITY_MW,
            "minimum_calibration_energy_mwh_per_mw": MIN_CALIBRATION_ENERGY_MWH_PER_MW,
        },
        "calibration_method": "Per-station median(actual daily MWh / raw physics daily MWh) on eligible training days only",
        "calibration": calibration,
        "predictions": output,
        "scores": scores,
        "test_score_status": "withheld until the final model and evaluation plan are frozen",
    }
    manifest_path = OUT / "physics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
