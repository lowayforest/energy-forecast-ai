"""Fetch issued GFS 0.25-degree forecasts for SQ1 August 2023 backtest.

For target local day D, use the GFS 00Z run dated D-1 (released later on
the previous local day). Only the target day's approximate 05:00-20:00
local daylight is needed for a daily energy estimate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from datetime import date, timedelta
from pathlib import Path

import eccodes
import httpx
import numpy as np
import pandas as pd

from prepare_data import OUT, physical_energy


BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
HOURS = (21, 24, 27, 30, 33, 36)
LAT, LON = 22.33, 114.18
SEMAPHORE = asyncio.Semaphore(5)
CACHE = Path(__file__).resolve().parents[1] / "data/cache/gfs"


async def get_retry(client, url, headers=None):
    for attempt in range(6):
        try:
            async with SEMAPHORE:
                response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.content
        except (httpx.HTTPError, httpx.TimeoutException):
            if attempt == 5:
                raise
            await asyncio.sleep(min(12, (2**attempt) * 0.5 + random.random()))


def base_path(run_day: date, hour: int):
    return f"{BASE}/gfs.{run_day:%Y%m%d}/00/atmos/gfs.t00z.pgrb2.0p25.f{hour:03d}"


def byte_range(index: str, name: str):
    rows = [line.split(":", 4) for line in index.splitlines() if line]
    for i, row in enumerate(rows):
        wanted_level = "2 m above ground" if name == "TMP" else "surface"
        if len(row) >= 5 and row[3] == name and row[4].startswith(wanted_level + ":"):
            start = int(row[1])
            stop = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else None
            if stop is None:
                raise ValueError("GRIB field unexpectedly last in index")
            return start, stop
    raise ValueError(f"{name} absent from GFS index")


def decode_nearest(message: bytes):
    handle = eccodes.codes_new_from_message(message)
    try:
        return float(eccodes.codes_grib_find_nearest(handle, LAT, LON)[0].value)
    finally:
        eccodes.codes_release(handle)


async def fetch_field(client, base, index, name):
    start, stop = byte_range(index, name)
    suffix = "2m" if name == "TMP" else "surface"
    run_label = base.split("/")[-4]
    cache_file = CACHE / f"{run_label}_{base.rsplit('/', 1)[-1]}_{name}_{suffix}.grb"
    if cache_file.exists():
        data = cache_file.read_bytes()
    else:
        data = await get_retry(client, base, {"Range": f"bytes={start}-{stop}"})
        if len(data) != stop - start + 1 or not data.startswith(b"GRIB"):
            raise ValueError(f"Bad range response for {base} {name}: {len(data)} bytes")
        cache_file.write_bytes(data)
    return decode_nearest(data)


async def fetch_step(client, run_day, lead):
    base = base_path(run_day, lead)
    idx_path = CACHE / f"{run_day:%Y%m%d}_f{lead:03d}.idx"
    if idx_path.exists():
        index = idx_path.read_text()
    else:
        index = (await get_retry(client, base + ".idx")).decode()
        idx_path.write_text(index)
    field_names = ["DSWRF"] + (["TMP"] if lead != 21 else [])
    values = await asyncio.gather(*(fetch_field(client, base, index, name) for name in field_names))
    return lead, dict(zip(field_names, values))


async def fetch_day(client, target: date, scale: float):
    run_day = target - timedelta(days=1)
    values = dict(await asyncio.gather(*(fetch_step(client, run_day, h) for h in HOURS)))
    # GFS DSWRF blocks alternate 3h and 6h averages. Subtract the first
    # three hours from a six-hour average to recover the latter three hours.
    ghi = [
        2 * values[24]["DSWRF"] - values[21]["DSWRF"],
        values[27]["DSWRF"],
        2 * values[30]["DSWRF"] - values[27]["DSWRF"],
        values[33]["DSWRF"],
        2 * values[36]["DSWRF"] - values[33]["DSWRF"],
    ]
    ghi = np.clip(ghi, 0, 1250)
    temp = [values[h]["TMP"] - 273.15 for h in (24, 27, 30, 33, 36)]
    kwh = float(np.sum(physical_energy(ghi, temp, 3, scale=scale)))
    return {
        "date": target.isoformat(), "run_utc": f"{run_day:%Y-%m-%d}T00:00:00Z",
        "raw_physics_kwh": round(kwh, 3), "ghi_3h_wm2": [round(float(x), 2) for x in ghi],
        "temp_3h_c": [round(float(x), 2) for x in temp],
        "forecast_grid_lat": 22.25, "forecast_grid_lon": 114.25,
    }


async def main(
    first: date,
    last: date,
    output_name: str = "gfs_backtest.json",
    freeze_calibration_through: date | None = None,
):
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    method = json.loads((OUT / "method.json").read_text())
    scale = method["empirical_site_scale"]
    actual = pd.read_csv(OUT / "sq1_daily.csv", parse_dates=["date"]).set_index("date")
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    existing_path = OUT / output_name
    existing = {}
    if existing_path.exists():
        existing = {r["date"]: r for r in json.loads(existing_path.read_text())}
    timeout = httpx.Timeout(60.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        rows = []
        for i, day in enumerate(days, 1):
            if day.isoformat() in existing:
                row = existing[day.isoformat()]
            else:
                try:
                    row = await fetch_day(client, day, scale)
                except Exception as exc:
                    print(f"Failed {day}: {exc}", flush=True)
                    continue
            history_end = (
                pd.Timestamp(freeze_calibration_through)
                if freeze_calibration_through else pd.Timestamp(day) - pd.Timedelta(days=2)
            )
            history = actual.loc[history_end - pd.Timedelta(days=29):history_end]
            history = history[(history.physics_observed_weather_kwh > 5) & (history.weather_hours >= 20)]
            ratios = history.actual_kwh / history.physics_observed_weather_kwh
            adaptive = float(np.clip(ratios.median(), 0.4, 1.6)) if len(ratios) >= 14 else 1.0
            if "raw_physics_kwh" not in row:
                row["raw_physics_kwh"] = row["physics_kwh"]
            row["adaptive_factor"] = round(adaptive, 4)
            row["physics_kwh"] = round(row["raw_physics_kwh"] * adaptive, 3)
            if freeze_calibration_through:
                row["calibration_frozen_through"] = freeze_calibration_through.isoformat()
            rows.append(row)
            existing[row["date"]] = row
            existing_path.write_text(json.dumps([existing[k] for k in sorted(existing)], indent=2))
            print(f"{i}/{len(days)} {day} {row['physics_kwh']:.1f} kWh", flush=True)
    if len(rows) != len(days):
        raise RuntimeError(f"Only {len(rows)}/{len(days)} days fetched; rerun to resume cached fields")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=date.fromisoformat, default=date(2023, 8, 1))
    parser.add_argument("--last", type=date.fromisoformat, default=date(2023, 8, 31))
    parser.add_argument("--output-name", default="gfs_backtest.json")
    parser.add_argument("--freeze-calibration-through", type=date.fromisoformat)
    args = parser.parse_args()
    asyncio.run(main(args.first, args.last, args.output_name, args.freeze_calibration_through))
