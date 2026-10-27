"""Export only measured or genuinely computed values to the React dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import OUT


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "site/solar-forecast/src/demo-data.json"


def metric(values):
    arr = np.asarray(values, dtype=float)
    return round(float(np.mean(np.abs(arr))), 2) if len(arr) else None


def main():
    actual = pd.read_csv(OUT / "sq1_daily.csv").set_index("date")
    gfs = json.loads((OUT / "gfs_backtest.json").read_text())
    ai_path = OUT / "doubao_predictions.json"
    ai = {r["date"]: r for r in json.loads(ai_path.read_text())} if ai_path.exists() else {}
    future_ai_path = OUT / "doubao_predictions_v2_jan2024.json"
    future_ai_days = len(json.loads(future_ai_path.read_text())) if future_ai_path.exists() else 0
    method = json.loads((OUT / "method.json").read_text())
    rows = []
    for forecast in gfs:
        day = forecast["date"]
        if day not in actual.index:
            continue
        observed = float(actual.loc[day, "actual_kwh"])
        ai_row = ai.get(day)
        rows.append({
            "date": day, "actual": round(observed, 2),
            "physics": forecast["physics_kwh"],
            "ai": ai_row["ai_kwh"] if ai_row else None,
            "aiReason": ai_row["reason"] if ai_row else None,
            "aiFactor": ai_row["factor"] if ai_row else None,
            "ghi": forecast["ghi_3h_wm2"], "temp": forecast["temp_3h_c"],
            "runUtc": forecast["run_utc"],
            "adaptiveFactor": forecast["adaptive_factor"],
            "observedWeatherPhysics": round(float(actual.loc[day, "physics_observed_weather_kwh"]), 2),
        })
    phys_err = [r["physics"] - r["actual"] for r in rows]
    ai_rows = [r for r in rows if r["ai"] is not None]
    ai_err = [r["ai"] - r["actual"] for r in ai_rows]
    output = {
        "meta": {
            "title": "光伏发电量预测实验平台",
            "site": "HKUST SQ1", "region": "中国香港 · 公开数据验证站点",
            "period": "2023-08", "ratedKw": 27.6,
            "latitude": 22.33, "longitude": 114.18,
            "calibration": "2021-06 至 2022-12",
            "calibrationDays": method["calibration_days"],
            "siteScale": method["empirical_site_scale"],
            "aiModel": "doubao-seed-2-1-pro-260915",
            "aiStatus": "ready" if len(ai_rows) == len(rows) and rows else "partial" if ai_rows else "pending_key",
            "disclaimer": "公开数据验证站点位于香港，并非上海；当前是历史回测演示，不代表上海项目现场预测。",
        },
        "metrics": {
            "days": len(rows), "completeMonth": len(rows) == 31,
            "actualTotal": round(sum(r["actual"] for r in rows), 2),
            "physicsTotal": round(sum(r["physics"] for r in rows), 2),
            "aiTotal": round(sum(r["ai"] for r in rows), 2) if len(ai_rows) == len(rows) and rows else None,
            "physicsMae": metric(phys_err),
            "aiMae": metric(ai_err), "aiEvaluatedDays": len(ai_rows),
        },
        "daily": rows,
        "sources": [
            {"name": "HKUST Rooftop PV Dataset", "use": "SQ1 站点 15 分钟发电量、站点额定容量和坐标；2021–2023 年", "url": "https://datadryad.org/dataset/doi:10.5061/dryad.m37pvmd99", "provider": "Dryad / HKUST"},
            {"name": "HKUST 气象观测", "use": "分钟级太阳辐照度和气温；用于历史校准、滚动校正与事后对照；2024-01 无新观测", "url": "https://zenodo.org/records/10909062", "provider": "Zenodo / HKUST"},
            {"name": "NOAA GFS 0.25° 预报归档", "use": "前一天 00Z 运行的短波辐射和 2 米气温预报；用于 2023-08 回测与 2024-01 无实测样例", "url": "https://registry.opendata.aws/noaa-gfs-bdp-pds/", "provider": "NOAA / AWS Open Data"},
            {"name": "pvlib PVWatts", "use": "光伏直流功率温度修正及电池温度近似模型", "url": "https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html", "provider": "pvlib"},
            {"name": "火山引擎 Ark 豆包", "use": f"读取物理基线与当时可用历史，输出受限修正系数；8 月 {len(ai_rows)} 天、1 月 {future_ai_days} 天均为真实 API 调用，未微调", "url": "https://docs.volcengine.com/docs/ark/model-list?lang=zh", "provider": "火山引擎"},
        ],
        "method": [
            {"step": "01", "title": "数据准备", "text": "SQ1 的 15 分钟发电量汇总为每日 kWh；2021–2022 年用于校准，2023 年 8 月用于回测。"},
            {"step": "02", "title": "天气预报输入", "text": "对每个目标日，读取前一天 00Z 的 GFS 预报，抽取邻近网格的辐照度与气温。"},
            {"step": "03", "title": "物理基线", "text": "简化 PVWatts 使用辐照度、组件温度、容量与效率，积分得到日发电量；再用预报发布时已完整取得的近 30 天资料滚动校正。"},
            {"step": "04", "title": "豆包修正", "text": "将预报天气、物理结果及截至目标日前两日的真实发电量交给豆包，输出受限修正系数。未调用 API 的日期不显示 AI 预测。"},
        ],
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Exported {len(rows)} days, {len(ai_rows)} real AI results to {TARGET}")


if __name__ == "__main__":
    main()
