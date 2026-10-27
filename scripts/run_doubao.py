"""Use Doubao through Ark to correct the issued-weather PVWatts forecast.

Set ARK_API_KEY in the environment or the ignored project-root .env file.
Responses are cached locally; the key never enters a browser build.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

from prepare_data import OUT, ROOT


MODEL = "doubao-seed-2-1-pro-260915"
API = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"


def api_key():
    key = os.getenv("ARK_API_KEY", "").strip()
    if key:
        return key
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "ARK_API_KEY":
                return value.strip().strip('"\'')
    return ""


def correct_one(client, key, row, history):
    prompt = {
        "forecast_date": row["date"],
        "station_capacity_kw": 27.6,
        "physics_forecast_kwh": row["physics_kwh"],
        "gfs_issued_at_utc": row["run_utc"],
        "gfs_3h_irradiance_wm2": row["ghi_3h_wm2"],
        "gfs_3h_temperature_c": row["temp_3h_c"],
        "recent_observed_daily_kwh_before_forecast_date": history,
    }
    body = {
        "model": MODEL,
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "max_completion_tokens": 128,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "你是光伏预测辅助模型。根据已发布的GFS天气预报、PVWatts物理基线和预测日前已知的发电历史，只修正物理基线的系统偏差。不能使用预测日当天或之后的真实发电量。输出一个JSON对象，字段factor为0.6到1.4之间的数字，reason为20字以内中文理由。不要编造其他观测数据。"},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }
    reply = client.post(API, headers={"Authorization": f"Bearer {key}"}, json=body)
    reply.raise_for_status()
    payload = reply.json()
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    factor = float(parsed["factor"])
    if not (0.6 <= factor <= 1.4):
        raise ValueError(f"Out-of-range Doubao factor: {factor}")
    return {
        "date": row["date"], "ai_kwh": round(row["physics_kwh"] * factor, 3),
        "factor": factor, "reason": str(parsed.get("reason", ""))[:60],
        "model": payload.get("model", MODEL), "gfs_issued_at_utc": row["run_utc"],
        "usage": payload.get("usage", {}),
    }


def main(max_days=None, forecast_file="gfs_backtest.json", cache_file="doubao_predictions.json"):
    key = api_key()
    if not key:
        raise SystemExit("ARK_API_KEY is not set in the environment or .env. No AI values have been generated.")
    physics = json.loads((OUT / forecast_file).read_text())
    actual = pd.read_csv(OUT / "sq1_daily.csv").set_index("date")
    cache_path = OUT / cache_file
    cache = {r["date"]: r for r in json.loads(cache_path.read_text())} if cache_path.exists() else {}
    attempts = 0
    with httpx.Client(timeout=90) as client:
        for row in physics:
            if row["date"] in cache:
                continue
            attempts += 1
            last_complete_day = (date.fromisoformat(row["date"]) - timedelta(days=2)).isoformat()
            history = actual.loc[actual.index <= last_complete_day, "actual_kwh"].tail(14)
            past = [{"date": d, "kwh": round(float(v), 2)} for d, v in history.items()]
            try:
                result = correct_one(client, key, row, past)
            except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as exc:
                print(f"Doubao failed on {row['date']}: {exc}")
                if max_days and attempts >= max_days:
                    break
                continue
            cache[row["date"]] = result
            cache_path.write_text(json.dumps([cache[k] for k in sorted(cache)], ensure_ascii=False, indent=2))
            print(f"{row['date']}: {result['ai_kwh']:.1f} kWh, factor={result['factor']:.3f}", flush=True)
            if max_days and attempts >= max_days:
                break
            time.sleep(0.2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-days", type=int)
    parser.add_argument("--forecast-file", default="gfs_backtest.json")
    parser.add_argument("--cache-file", default="doubao_predictions.json")
    args = parser.parse_args()
    main(args.max_days, args.forecast_file, args.cache_file)
