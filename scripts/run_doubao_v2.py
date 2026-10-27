"""Conservative, evidence-based Doubao residual correction (prompt v2).

Keeps v1 results untouched. Historical examples are cut off at D-2 to match
what could have been known when the D-1 00Z GFS run became available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import date, timedelta

import httpx
import pandas as pd

from prepare_data import OUT, ROOT
from run_doubao import API, MODEL, api_key


PROMPT_FILE = ROOT / "prompts/pv_correction_v2.md"
PROMPT = PROMPT_FILE.read_text()
PROMPT_HASH = hashlib.sha256(PROMPT.encode()).hexdigest()[:12]
FUTURE_APPENDIX = """

本次目标月没有任何真实发电量记录。站点物理校正系数冻结在输入给出的日期；
`latest_available_real_generation_date` 之后的发电量未知。随着该日期变远，
历史偏差的可信度降低。绝不能把目标月的天气预报或物理预测当成真实发电量，
也不能根据不存在的月内实绩声称已滚动校准。证据过旧时输出 factor=1.000。
"""


def system_prompt(row):
    return PROMPT + (FUTURE_APPENDIX if row.get("calibration_frozen_through") else "")


def prompt_hash(row):
    return hashlib.sha256(system_prompt(row).encode()).hexdigest()[:12]

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "pv_residual_correction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "factor": {"type": "number"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "evidence_dates": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["factor", "confidence", "evidence_dates", "reason"],
            "additionalProperties": False,
        },
    },
}


def summarize_ratios(history):
    ratios = [h["actual_kwh"] / h["physics_kwh"] for h in history if h["physics_kwh"] > 5]
    if not ratios:
        return {"count": 0}
    return {
        "count": len(ratios),
        "median_actual_over_physics": round(statistics.median(ratios), 3),
        "overprediction_days": sum(r < 1 for r in ratios),
        "underprediction_days": sum(r > 1 for r in ratios),
    }


def build_input(row, forecasts, actual):
    target = date.fromisoformat(row["date"])
    latest_known = target - timedelta(days=2)
    earlier = [
        f for f in forecasts
        if date.fromisoformat(f["date"]) <= latest_known
        and f["date"] in actual.index
        and f["physics_kwh"] > 0
    ][-14:]
    issued_history = []
    for f in earlier:
        measured = round(float(actual.loc[f["date"], "actual_kwh"]), 2)
        issued_history.append({
            "date": f["date"], "physics_kwh": round(f["physics_kwh"], 2),
            "actual_kwh": measured,
            "actual_over_physics": round(measured / f["physics_kwh"], 3),
            "forecast_irradiance_sum_3h_wm2": round(sum(f["ghi_3h_wm2"]), 1),
        })
    last_observed = actual.loc[actual.index <= latest_known.isoformat()].tail(7)
    last_actual_date = last_observed.index[-1] if len(last_observed) else None
    site_history = [
        {
            "date": d, "actual_kwh": round(float(r.actual_kwh), 2),
            "observed_weather_physics_kwh": round(float(r.physics_observed_weather_kwh), 2),
            "observed_ghi_peak_wm2": round(float(r.ghi_peak_wm2), 1),
        }
        for d, r in last_observed.iterrows()
    ]
    target_ghi = sum(row["ghi_3h_wm2"])
    similar = [
        h for h in issued_history
        if 0.7 * target_ghi <= h["forecast_irradiance_sum_3h_wm2"] <= 1.3 * target_ghi
    ]
    return {
        "target": {
            "date": row["date"], "issued_at_utc": row["run_utc"],
            "capacity_kw": 27.6, "physics_forecast_kwh": row["physics_kwh"],
            "forecast_ghi_3h_wm2": row["ghi_3h_wm2"],
            "forecast_temp_3h_c": row["temp_3h_c"],
            "forecast_irradiance_sum_3h_wm2": round(target_ghi, 1),
            "calibration_frozen_through": row.get("calibration_frozen_through"),
        },
        "latest_permitted_observation_date": latest_known.isoformat(),
        "latest_available_real_generation_date": last_actual_date,
        "days_since_last_real_generation": (
            (target - date.fromisoformat(last_actual_date)).days if last_actual_date else None
        ),
        "recent_issued_forecasts": issued_history,
        "all_recent_error_summary": summarize_ratios(issued_history),
        "similar_irradiance_error_summary": summarize_ratios(similar),
        "recent_site_observations": site_history,
    }


def correct_one(client, key, row, forecast_context, actual):
    evidence = build_input(row, forecast_context, actual)
    response = client.post(
        API,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": MODEL, "temperature": 0,
            "thinking": {"type": "disabled"},
            "max_completion_tokens": 180,
            "response_format": SCHEMA,
            "messages": [
                {"role": "system", "content": system_prompt(row)},
                {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
            ],
        },
    )
    response.raise_for_status()
    payload = response.json()
    parsed = json.loads(payload["choices"][0]["message"]["content"])
    factor = float(parsed["factor"])
    if not 0.94 <= factor <= 1.06:
        raise ValueError(f"factor outside prompt bounds: {factor}")
    cited = parsed["evidence_dates"]
    available_dates = {
        h["date"] for h in evidence["recent_issued_forecasts"] + evidence["recent_site_observations"]
    }
    if len(cited) > 3 or not all(d in available_dates for d in cited):
        raise ValueError(f"unsupported evidence dates: {cited}")
    return {
        "date": row["date"], "ai_kwh": round(row["physics_kwh"] * factor, 3),
        "factor": factor, "confidence": parsed["confidence"],
        "evidence_dates": cited, "reason": parsed["reason"][:60],
        "model": payload.get("model", MODEL), "prompt_sha256": prompt_hash(row),
        "gfs_issued_at_utc": row["run_utc"], "usage": payload.get("usage", {}),
    }


def main(forecast_file: str, cache_file: str, history_files: list[str], max_days: int | None):
    key = api_key()
    if not key:
        raise SystemExit("ARK_API_KEY is missing")
    forecasts = json.loads((OUT / forecast_file).read_text())
    contexts = {}
    for filename in [*history_files, forecast_file]:
        contexts.update({r["date"]: r for r in json.loads((OUT / filename).read_text())})
    context = [contexts[k] for k in sorted(contexts)]
    actual = pd.read_csv(OUT / "sq1_daily.csv").set_index("date")
    cache_path = OUT / cache_file
    cache = {r["date"]: r for r in json.loads(cache_path.read_text())} if cache_path.exists() else {}
    attempts = 0
    with httpx.Client(timeout=90) as client:
        for row in forecasts:
            if cache.get(row["date"], {}).get("prompt_sha256") == prompt_hash(row):
                continue
            attempts += 1
            try:
                result = correct_one(client, key, row, context, actual)
            except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as exc:
                print(f"V2 failed on {row['date']}: {exc}", flush=True)
                if max_days and attempts >= max_days:
                    break
                continue
            cache[row["date"]] = result
            cache_path.write_text(json.dumps([cache[k] for k in sorted(cache)], ensure_ascii=False, indent=2))
            print(f"{row['date']}: {result['ai_kwh']:.1f} kWh, factor={result['factor']:.3f}, confidence={result['confidence']}", flush=True)
            if max_days and attempts >= max_days:
                break
            time.sleep(0.2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-file", default="gfs_backtest.json")
    parser.add_argument("--cache-file", default="doubao_predictions_v2_aug2023.json")
    parser.add_argument("--history-file", action="append", default=[])
    parser.add_argument("--max-days", type=int)
    args = parser.parse_args()
    main(args.forecast_file, args.cache_file, args.history_file, args.max_days)
