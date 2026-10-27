# 光伏发电量预测演示

本仓库是申请书中“AI 大模型驱动的新能源出力预测”的缩小验证版：先针对单个公开屋顶光伏站点做**次日发电量**回测，再把逐日预测累加成月度回测值。申请书提出的多尺度时序建模、多任务协同预测、机理约束预训练和场景适配属于后续正式项目研究范围，本演示尚未实现。

## 当前结果

- 站点：HKUST 公开数据集 SQ1，位于中国香港，额定容量 27.6 kW；不是上海站点。
- 历史数据：2021-06 至 2023-12 的 944 个有效日，其中 2021-06 至 2022-12 的 572 天用于固定站点系数校准。
- 回测：2023 年 8 月 31 天，每天读取**前一日 00Z GFS 运行**的天气预报；用运行时已完整取得的近 30 天实际发电量与历史天气观测滚动校正，历史截止目标日前两日。
- 这里的“前一日”是天气**预报的发布时间**；读取的是这次预报中对应**目标日白天**的辐照度和气温，而不是拿前一天的实测天气代替目标日天气。目标日实际发电量只用于事后评分。
- 物理模型：pvlib 的简化 PVWatts 直流功率模型 + Ross 组件温度近似 + 逆变器效率假设；以历史记录拟合站点系数。组件倾角和方位角缺少可核实字段，GHI 暂作为有效辐照度近似，因此不是完整的组件平面辐照模型。
- 大模型：`scripts/run_doubao.py` 使用火山引擎 Ark API 调用 `doubao-seed-2-1-pro-260915`，以物理预测、GFS 天气与目标日前历史发电量输出修正系数。31 天均已完成真实 API 调用；脚本仍在没有 API 密钥时拒绝生成 AI 预测。
- 目前物理回测：8 月真实累计 3,762.8 kWh，物理逐日预测累计 3,837.9 kWh，日 MAE 29.35 kWh。这是单月、单站点结果，不说明泛化能力。
- 豆包修正回测：预测累计 3,916.0 kWh，日 MAE 29.56 kWh，略高于物理基线。当前结果不能宣称大模型提高了精度；后续应在独立月份上调整修正策略并评估泛化性。

## 运行

```bash
uv venv .venv
uv pip install --python .venv/bin/python pandas numpy pvlib httpx pydantic eccodes
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/fetch_gfs.py --first 2023-08-01 --last 2023-08-31
```

将 `.env.example` 复制为仓库根目录的 `.env`，在 `.env` 中填入 `ARK_API_KEY=你的密钥`（不要把密钥发在聊天中）。`.env` 已被 Git 忽略。然后运行真实豆包调用：

```bash
.venv/bin/python scripts/run_doubao.py
.venv/bin/python scripts/build_demo.py
cd site/solar-forecast
npm install
npm run dev
```

请勿将密钥写入代码、提交到仓库或放进 React 前端。若仅查看物理模型演示，可跳过 `run_doubao.py`，直接运行 `build_demo.py`。

## 数据与方法来源

| 输入 | 用途 | 来源 |
| --- | --- | --- |
| HKUST 屋顶光伏公开数据 | SQ1 站点发电量、容量、坐标和历史气象观测 | [Dryad 原始数据](https://datadryad.org/dataset/doi:10.5061/dryad.m37pvmd99)；[Zenodo 镜像](https://zenodo.org/records/10909062) |
| NOAA GFS 0.25° 预报归档 | 预测日前已发布的辐照度和 2 米气温预报 | [NOAA / AWS Open Data](https://registry.opendata.aws/noaa-gfs-bdp-pds/) |
| PVWatts | 辐照度和温度到直流功率的物理计算 | [pvlib 文档](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html) |
| Ross 温度模型 | 从辐照度与环境温度近似组件温度 | [pvlib 文档](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.ross.html) |
| 豆包 Ark API | 对物理预测输出受限修正系数 | [火山引擎模型文档](https://docs.volcengine.com/docs/ark/model-list?lang=zh) |

原始压缩包已下载在 `data/raw/hkust_rooftop_pv_2021_2023.zip`，大小约 296 MB；原始数据、GRIB 缓存和处理后的数据均被 `.gitignore` 排除。React 页面展示文件由 `scripts/build_demo.py` 生成。

## 解释边界

1. 月度卡片是 31 个逐日次日预测结果的**累计回测**，不是月初一次性做出的整月预测。正式项目要做真正的月末前瞻预测，还需长期天气预报或季节预测及对应的误差评估。
2. 2023 年的现场辐照与站点产出关系相对 2021–2022 年明显变化，所以演示加入只使用预报发布时已完整取得的近 30 天数据的自适应校正。这个系数不能被解释为单独的光伏损耗。
3. 在拿到上海项目现场数据后，需要替换站点与气象输入，补齐组件倾角、方位角、实际可用率及限电等参数，再做跨月份、跨站点验证。当前结果不应当用作商业发电量承诺。

## 提示词实验

另见[新版完整提示词](prompts/pv_correction_v2.md)与[三个月对照报告](reports/prompt_v2_experiment.md)。新版加入历史已发布预测的误差、相似辐照度比较和严格输出约束；10 月留出回测有小幅改善，但整体日预测精度收益尚不稳定。

## 数据集外的无实测预测月

演示平台另加了 **2024 年 1 月**：HKUST 公开数据中的 SQ1 发电实绩截止 2023-12-31，因此该月没有真实发电量可供评分。选择紧邻数据集的月份，可以沿用已知站点容量、近期状态和气象预报归档；比直接跳到 2026 年、却缺少中间数年电站运行信息更适合作为方法演示。

- 天气输入：每天目标日前一日 00Z 的 NOAA GFS 归档预报，按当时信息重放；这**不是当前实时预报**。
- 物理校正：统一冻结在 2023-12-30 可用的最近 30 天资料，1 月期间没有使用任何未提供的发电实绩。
- 豆包：读取当时可用的 2023 年 12 月历史与天气预报，真实调用 31 次；历史资料变旧后优先保持物理基线。
- 2023 年数据并未丢弃：8 月回测时按日期使用此前已完整取得的实测数据做滚动校正；预测 2024 年 1 月时使用 2023 年末的实测记录。固定站点系数只在 2021–2022 年拟合；2023 年用于按时间顺序回测和动态校正，并非完全不参与模型调整的独立测试集。豆包调用结果缓存于 `data/processed/doubao_predictions_v2_jan2024.json`，前端只展示整理后的预测值与解释。
- 预测月累计：物理模型 **2,276.65 kWh**，豆包修正 **2,272.60 kWh**。这些是逐日预报之和；因为没有实绩，MAE、MAPE 和精度提升均**不可计算**。

本地复现使用 `scripts/fetch_gfs.py --first 2024-01-01 --last 2024-01-31 --output-name gfs_jan2024.json --freeze-calibration-through 2023-12-30`，然后运行 `scripts/run_doubao_v2.py` 与 `scripts/build_future_demo.py`。相关结果缓存保存在被 Git 忽略的 `data/processed/`，前端展示数据见 `site/solar-forecast/src/future-data.json`。
