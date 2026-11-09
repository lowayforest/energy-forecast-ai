"""Audit AI-PVOD station data before building day-ahead research examples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/research/ai_pvod_full_dataset.zip"

# Rated capacities from Table 1 of doi:10.1038/s41467-026-73817-3.
CAPACITY_MW = {"S-1": 6.6, "S-2": 20.0, "S-3": 20.0, "S-4": 20.0}
NWP_FIELDS = (
    "nwp_globalirrad",
    "nwp_directirrad",
    "nwp_temperature",
    "nwp_humidity",
    "nwp_windspeed",
    "nwp_winddirection",
    "nwp_pressure",
)


def audit_station(source: ZipFile, station: str) -> dict:
    with source.open(f"All_dataset/{station}.csv") as file:
        frame = pd.read_csv(file)

    required = {"Timestamp", "Time", "power", *NWP_FIELDS}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{station}: missing columns {sorted(missing_columns)}")

    utc = pd.to_datetime(frame["Timestamp"], errors="raise")
    local = pd.to_datetime(frame["Time"], errors="raise")
    if not utc.is_monotonic_increasing or utc.duplicated().any():
        raise ValueError(f"{station}: UTC timestamps are unsorted or duplicated")
    if not (utc.diff().iloc[1:] == pd.Timedelta(minutes=15)).all():
        raise ValueError(f"{station}: UTC timestamps are not continuous at 15 minutes")
    if not ((local - utc) == pd.Timedelta(hours=8)).all():
        raise ValueError(f"{station}: local timestamps are not UTC+8")

    numeric = frame[["power", *NWP_FIELDS]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{station}: power or NWP contains missing/nonfinite values")
    if (numeric["power"] < 0).any() or (numeric["power"] > CAPACITY_MW[station] + 0.01).any():
        raise ValueError(f"{station}: power falls outside 0 to rated capacity")

    dates = local.dt.date
    per_day = dates.value_counts()
    if not (per_day == 96).all():
        raise ValueError(f"{station}: a local day does not have 96 records")
    daily_mwh = numeric["power"].groupby(dates).sum() * 0.25
    return {
        "station": station,
        "rated_capacity_mw": CAPACITY_MW[station],
        "first_local_day": str(dates.min()),
        "last_local_day": str(dates.max()),
        "complete_days": int(len(per_day)),
        "records": int(len(frame)),
        "max_power_mw": round(float(numeric["power"].max()), 3),
        "mean_daily_energy_mwh": round(float(daily_mwh.mean()), 3),
        "max_daily_energy_mwh": round(float(daily_mwh.max()), 3),
    }


def main() -> None:
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    with ZipFile(ARCHIVE) as source:
        stations = [audit_station(source, station) for station in CAPACITY_MW]
    print(json.dumps({"source": str(ARCHIVE), "sha256": digest, "stations": stations}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
