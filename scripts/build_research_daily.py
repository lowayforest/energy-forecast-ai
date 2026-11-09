"""Build leakage-aware daily PV examples from the audited AI-PVOD archive."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from audit_research_dataset import ARCHIVE, CAPACITY_MW, ROOT, audit_station


OUT = ROOT / "data/processed/research"
HISTORY_DAYS = 7
HISTORY_LAG_DAYS = 3  # No complete actual day later than target day minus 3.
TRAIN_END = "2019-02-28"
VALID_END = "2019-04-15"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_for(day: str) -> str:
    if day <= TRAIN_END:
        return "train"
    if day <= VALID_END:
        return "valid"
    return "test"


def station_examples(source: ZipFile, station: str) -> list[dict]:
    audit_station(source, station)
    with source.open(f"All_dataset/{station}.csv") as file:
        frame = pd.read_csv(file)
    frame["local_day"] = pd.to_datetime(frame["Time"]).dt.date

    daily = frame.groupby("local_day", sort=True).agg(
        actual_energy_mwh=("power", lambda x: float(x.sum()) * 0.25),
        nwp_ghi_kwh_m2=("nwp_globalirrad", lambda x: float(x.sum()) * 0.25 / 1000),
        nwp_dni_kwh_m2=("nwp_directirrad", lambda x: float(x.sum()) * 0.25 / 1000),
        nwp_ghi_peak_wm2=("nwp_globalirrad", "max"),
        nwp_temp_mean_c=("nwp_temperature", "mean"),
        nwp_temp_max_c=("nwp_temperature", "max"),
        nwp_humidity_mean_pct=("nwp_humidity", "mean"),
        nwp_wind_mean_mps=("nwp_windspeed", "mean"),
    )
    daily["actual_full_load_hours"] = daily["actual_energy_mwh"] / CAPACITY_MW[station]

    examples = []
    for day, row in daily.iterrows():
        history_end = day - timedelta(days=HISTORY_LAG_DAYS)
        history_dates = [history_end - timedelta(days=n) for n in range(HISTORY_DAYS - 1, -1, -1)]
        if not all(history_day in daily.index for history_day in history_dates):
            continue
        if any(history_day >= day for history_day in history_dates):
            raise AssertionError("Future history entered an example")

        inputs = {
            "rated_capacity_mw": CAPACITY_MW[station],
            "day_of_year": day.timetuple().tm_yday,
            "nwp_ghi_kwh_m2": round(float(row["nwp_ghi_kwh_m2"]), 4),
            "nwp_dni_kwh_m2": round(float(row["nwp_dni_kwh_m2"]), 4),
            "nwp_ghi_peak_wm2": round(float(row["nwp_ghi_peak_wm2"]), 2),
            "nwp_temp_mean_c": round(float(row["nwp_temp_mean_c"]), 2),
            "nwp_temp_max_c": round(float(row["nwp_temp_max_c"]), 2),
            "nwp_humidity_mean_pct": round(float(row["nwp_humidity_mean_pct"]), 2),
            "nwp_wind_mean_mps": round(float(row["nwp_wind_mean_mps"]), 2),
            "history_dates": [str(history_day) for history_day in history_dates],
            "history_full_load_hours": [
                round(float(daily.loc[history_day, "actual_full_load_hours"]), 4)
                for history_day in history_dates
            ],
        }
        if "power" in inputs or any(key.startswith("lmd_") for key in inputs):
            raise AssertionError("Target-day observation entered model inputs")
        examples.append({
            "station": station,
            "target_date": str(day),
            "split": split_for(str(day)),
            "inputs": inputs,
            "target": {
                "actual_energy_mwh": round(float(row["actual_energy_mwh"]), 5),
                "actual_full_load_hours": round(float(row["actual_full_load_hours"]), 5),
            },
        })
    return examples


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with ZipFile(ARCHIVE) as source:
        examples = [example for station in CAPACITY_MW for example in station_examples(source, station)]

    examples.sort(key=lambda example: (example["target_date"], example["station"]))
    if len({(example["station"], example["target_date"]) for example in examples}) != len(examples):
        raise AssertionError("Duplicate station/date examples")

    manifest = {
        "source_url": "https://huggingface.co/datasets/yujiaA/AI-PVOD",
        "source_sha256": sha256(ARCHIVE),
        "timezone": "Asia/Shanghai",
        "history_days": HISTORY_DAYS,
        "latest_complete_history_day": "target_date - 3 days",
        "split_boundaries": {"train_end": TRAIN_END, "valid_end": VALID_END, "test_end": "2019-06-13"},
        "daily_energy_formula": "sum(power_mw * 0.25 h) = MWh",
        "forecast_issue_note": "PVOD paper describes daily WRF NWP at 28-54 h lead; per-row issuance time is absent from CSV.",
        "splits": {},
    }
    for split in ("train", "valid", "test"):
        rows = [example for example in examples if example["split"] == split]
        path = OUT / f"daily_{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["splits"][split] = {
            "rows": len(rows),
            "first_target_date": rows[0]["target_date"],
            "last_target_date": rows[-1]["target_date"],
            "sha256": sha256(path),
        }
    (OUT / "daily_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
