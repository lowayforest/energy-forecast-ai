"""Compare PV physics and Doubao prompt versions on complete calendar months."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import OUT


FILES = {
    "2023-08": ("gfs_backtest.json", "doubao_predictions.json", "doubao_predictions_v2_aug2023.json"),
    "2023-09": ("gfs_sep2023.json", "doubao_predictions_v1_sep2023.json", "doubao_predictions_v2_sep2023.json"),
    "2023-10": ("gfs_oct2023.json", None, "doubao_predictions_v2_oct2023.json"),
}


def main():
    actual = pd.read_csv(OUT / "sq1_daily.csv").set_index("date")
    output = {}
    for month, filenames in FILES.items():
        forecasts = json.loads((OUT / filenames[0]).read_text())
        dates = [r["date"] for r in forecasts]
        observed = np.array([float(actual.loc[d, "actual_kwh"]) for d in dates])
        variants = {"physics": np.array([r["physics_kwh"] for r in forecasts])}
        for label, filename in zip(("v1", "v2"), filenames[1:]):
            if filename and (OUT / filename).exists():
                rows = {r["date"]: r for r in json.loads((OUT / filename).read_text())}
                if all(d in rows for d in dates):
                    variants[label] = np.array([rows[d]["ai_kwh"] for d in dates])
        result = {"days": len(dates), "actual_total_kwh": round(float(observed.sum()), 2)}
        for label, pred in variants.items():
            error = pred - observed
            result[label] = {
                "mae_kwh": round(float(np.mean(np.abs(error))), 3),
                "bias_kwh": round(float(np.mean(error)), 3),
                "predicted_total_kwh": round(float(pred.sum()), 2),
                "monthly_absolute_error_kwh": round(float(abs(error.sum())), 2),
            }
        output[month] = result
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
