"""Verify the downloaded Romanian PV candidate without training or scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from audit_research_dataset import ROOT


DATA = ROOT / "data/raw/research/romania_pv"
OUT = ROOT / "data/processed/research"
REVISION = "e14509a8c7f8d12a63afbedb052a8f6caa8c98fa"
EXPECTED = {
    "site_a_hourly_weather.csv": "295509cb2b0e332f81cbe63c7462f15eb29ffe84930b836df2e1a19d8537786d",
    "site_b_hourly_weather.csv": "4cc144a101d6953895b66dcdf47f5058eab4952a6a6954840b10f041d3ddf4b1",
    "site_a_and_b_lead1.csv": "9c18e860f78b351990bc69546eb417948acc125f91e007f24039c793f2add4a4",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for name, expected in EXPECTED.items():
        if sha256(DATA / name) != expected:
            raise ValueError(f"Downloaded source changed: {name}")
    forecast = pd.read_csv(DATA / "site_a_and_b_lead1.csv", parse_dates=["valid_time_utc"])
    if forecast["valid_time_utc"].duplicated().any() or forecast.isna().any().any():
        raise ValueError("Forecast timestamps duplicated or fields missing")
    times = pd.DatetimeIndex(forecast["valid_time_utc"])
    if not times.is_monotonic_increasing or (times.to_series().diff().dropna() != pd.Timedelta(hours=1)).any():
        raise ValueError("Forecast is not a continuous hourly series")
    report = {
        "purpose": "candidate data quality only; no model training or scoring",
        "upstream_url": "https://github.com/upb-nlp/PV-Generation-Dataset",
        "revision": REVISION,
        "source_sha256": EXPECTED,
        "forecast_rows": len(forecast),
        "forecast_time_min_utc": str(times.min()),
        "forecast_time_max_utc": str(times.max()),
        "sites": {},
    }
    for station in ("a", "b"):
        rows = pd.read_csv(DATA / f"site_{station}_hourly_weather.csv", parse_dates=["datetime_utc"])
        timestamps = pd.DatetimeIndex(rows["datetime_utc"])
        if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
            raise ValueError(f"Site {station}: duplicated or unsorted hours")
        if (timestamps.to_series().diff().dropna() != pd.Timedelta(hours=1)).any():
            raise ValueError(f"Site {station}: UTC hours are not continuous")
        if not rows["datetime_local"].str.fullmatch(r"\d{4}-\d{2}-\d{2} .+").all():
            raise ValueError(f"Site {station}: local timestamp format changed")
        if not timestamps.isin(times).all():
            raise ValueError(f"Site {station}: forecast does not cover every row")
        dates = rows["datetime_local"].str[:10]
        grouped = rows.groupby(dates, sort=True)
        day_counts = grouped.size()
        usable = grouped["active_power_kw"].apply(lambda values: values.notna().all())
        report["sites"][station] = {
            "hourly_rows": len(rows),
            "local_first_day": dates.min(),
            "local_last_day": dates.max(),
            "local_days": len(day_counts),
            "hours_per_day_distribution": {str(int(k)): int(v) for k, v in day_counts.value_counts().items()},
            "power_missing_hours": int(rows["active_power_kw"].isna().sum()),
            "usable_full_label_days": int(usable.sum()),
            "usable_by_month": {str(k): int(v) for k, v in usable.groupby(usable.index.str[:7]).sum().items()},
            "matched_forecast_hours": int(timestamps.isin(times).sum()),
            "max_active_power_kw": float(rows["active_power_kw"].max()),
        }
    path = OUT / "romanian_candidate_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
