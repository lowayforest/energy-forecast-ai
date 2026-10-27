"""Export a clearly labelled month without PV generation ground truth."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from prepare_data import OUT, ROOT


TARGET = ROOT / "site/solar-forecast/src/future-data.json"
MONTH = "2024-01"


def main():
    actual = pd.read_csv(OUT / "sq1_daily.csv").set_index("date")
    forecast = json.loads((OUT / "gfs_jan2024.json").read_text())
    ai_path = OUT / "doubao_predictions_v2_jan2024.json"
    ai = {r["date"]: r for r in json.loads(ai_path.read_text())} if ai_path.exists() else {}
    if len(forecast) != 31 or len({r["date"] for r in forecast}) != 31:
        raise ValueError("2024-01 must contain 31 distinct forecast dates")
    rows = []
    for row in sorted(forecast, key=lambda r: r["date"]):
        day = row["date"]
        if not day.startswith(MONTH) or day in actual.index:
            raise ValueError(f"Target month must be 2024-01 with no actual PV data: {day}")
        if date.fromisoformat(row["run_utc"][:10]) != date.fromisoformat(day) - timedelta(days=1):
            raise ValueError(f"Weather forecast issuance does not precede {day}")
        if row.get("calibration_frozen_through") != "2023-12-30":
            raise ValueError("Missing frozen calibration date")
        ai_row = ai.get(day)
        rows.append({
            "date": day, "actual": None, "physics": row["physics_kwh"],
            "ai": ai_row["ai_kwh"] if ai_row else None,
            "aiFactor": ai_row["factor"] if ai_row else None,
            "aiConfidence": ai_row["confidence"] if ai_row else None,
            "aiReason": ai_row["reason"] if ai_row else None,
            "ghi": row["ghi_3h_wm2"], "temp": row["temp_3h_c"],
            "runUtc": row["run_utc"],
        })
    ai_complete = len(ai) == len(rows) and all(r["ai"] is not None for r in rows)
    data = {
        "meta": {
            "period": MONTH, "station": "HKUST SQ1",
            "kind": "数据集外样例月 · 无真实发电量",
            "isCurrentForecast": False,
            "lastActualDate": actual.index.max(),
            "calibrationFrozenThrough": "2023-12-30",
            "calibrationPeriod": "2021-06 至 2022-12；2023-12 历史用于冻结校正",
            "weather": "NOAA GFS 0.25°；逐日取目标日前一天 00Z 的归档预报",
            "aiModel": "doubao-seed-2-1-pro-260915",
            "aiStatus": "ready" if ai_complete else "partial",
        },
        "metrics": {
            "days": len(rows), "actualTotal": None,
            "physicsTotal": round(sum(r["physics"] for r in rows), 2),
            "aiTotal": round(sum(r["ai"] for r in rows), 2) if ai_complete else None,
            "mae": None, "mape": None,
        },
        "daily": rows,
    }
    TARGET.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"Exported {len(rows)} dates; {len(ai)} AI values; actual PV values absent")


if __name__ == "__main__":
    main()
