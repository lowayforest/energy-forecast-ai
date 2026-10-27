"""Prepare the public SQ1 PV series and a calibrated PVWatts baseline.

The weather files are nearby observed conditions. Their use here is limited to
calibration and retrospective context; genuine day-ahead weather is fetched by
fetch_gfs.py from a previously issued GFS run.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pvlib.pvsystem import pvwatts_dc
from pvlib.temperature import ross


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/hkust_rooftop_pv_2021_2023.zip"
OUT = ROOT / "data/processed"
PREFIX = "Dataset/Time series dataset/"
SITE = PREFIX + "PV generation dataset/PV stations with panel level optimizer/Site level dataset/SQ1.csv"
WEATHER = PREFIX + "Meteorological dataset/"
CAPACITY_W = 27_600.0
GAMMA = -0.004


def physical_energy(irradiance_wm2, temp_c, hours, scale=1.0):
    """Simplified PVWatts DC plus fixed inverter efficiency and fitted losses."""
    ghi = np.maximum(np.asarray(irradiance_wm2, dtype=float), 0)
    ghi = np.where(ghi < 20, 0, ghi)  # observed sensor has a small night offset
    air = np.asarray(temp_c, dtype=float)
    cell = ross(ghi, air, noct=45.0)
    dc = pvwatts_dc(ghi, cell, pdc0=CAPACITY_W, gamma_pdc=GAMMA)
    return np.maximum(dc * 0.96 * scale, 0) * np.asarray(hours) / 1000


def read_zip_csv(source, name):
    with source.open(name) as file:
        return pd.read_csv(file)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as source:
        pv = read_zip_csv(source, SITE)
        pv["Time"] = pd.to_datetime(pv["Time"])
        pv["generation(kWh)"] = pd.to_numeric(pv["generation(kWh)"], errors="coerce")
        pv = pv.set_index("Time").sort_index()
        daily = pv["generation(kWh)"].resample("D").agg(["sum", "count"])
        daily = daily[daily["count"] >= 90].rename(columns={"sum": "actual_kwh", "count": "pv_intervals"})

        weather_by_year = []
        for year in (2021, 2022, 2023):
            irr = read_zip_csv(source, f"{WEATHER}Irradiance/Irradiance_{year}.csv")
            temp = read_zip_csv(source, f"{WEATHER}Temperature/Temperature_{year}.csv")
            irr["Time"] = pd.to_datetime(irr["Time"])
            temp["Time"] = pd.to_datetime(temp["Time"])
            irr = irr.set_index("Time").iloc[:, 0].rename("ghi_wm2")
            temp = temp.set_index("Time").iloc[:, 0].rename("temp_c")
            weather_by_year.append(pd.concat([irr, temp], axis=1))
    weather = pd.concat(weather_by_year).sort_index()
    hourly = weather.resample("h").agg({"ghi_wm2": "mean", "temp_c": "mean"}).dropna()
    hourly["raw_physics_kwh"] = physical_energy(hourly.ghi_wm2, hourly.temp_c, 1)
    daily_physics = hourly.raw_physics_kwh.resample("D").sum()
    daily_weather = hourly.resample("D").agg(ghi_peak_wm2=("ghi_wm2", "max"), temp_mean_c=("temp_c", "mean"), weather_hours=("ghi_wm2", "count"))
    result = daily.join(daily_physics).join(daily_weather)
    training = result.loc["2021-06-01":"2022-12-31"]
    training = training[(training.raw_physics_kwh > 5) & (training.weather_hours >= 20)]
    ratio = training.actual_kwh / training.raw_physics_kwh
    scale = float(np.clip(ratio.median(), 0.2, 1.5))
    result["physics_observed_weather_kwh"] = result.raw_physics_kwh * scale
    result.index.name = "date"
    result.to_csv(OUT / "sq1_daily.csv", float_format="%.4f")
    metadata = {
        "station": "SQ1", "latitude": 22.33, "longitude": 114.18,
        "rated_capacity_kw": 27.6, "gamma_pdc_per_c": GAMMA, "noct_c": 45,
        "inverter_efficiency": 0.96, "empirical_site_scale": round(scale, 5),
        "calibration_start": "2021-06-01", "calibration_end": "2022-12-31",
        "calibration_days": len(training),
        "weather_note": "Nearby onsite weather; used for calibration and retrospective context only. Site scale also absorbs unknown panel tilt and orientation, so it is not a loss estimate.",
    }
    (OUT / "method.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({"daily_rows": len(result), **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
