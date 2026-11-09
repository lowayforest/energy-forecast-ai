# 步骤 4：未微调本地大模型的验证集对照

## 目的与实验边界

在[步骤 2 的固定时间划分](research_daily_dataset.md)和[步骤 3 的物理基线](research_physics_baseline.md)上，测量**未针对本数据集微调**的本地大模型能否改善日发电量预测。只对 2019-03-01 至 2019-04-15 的验证集评分，共 46 天 × 4 站 = **184 条**；2019-04-16 之后的测试标签仍未使用。物理基线使用已在训练集拟合的站点系数，验证阶段不重新校准。

模型为本机 Ollama 的 `qwen3.5:9b`，来自 [Ollama 模型库](https://ollama.com/library/qwen3.5)所列 Qwen3.5 系列；上游 [Qwen3.5-9B 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)。本地模型为 Q4_K_M 量化、约 9.7B 参数，运行时模型 digest：`6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`。运行环境：MacBook Pro、Apple M5、16 GB 内存，Python 3.12.14，Ollama 0.33.3。**本步骤只有推理，没有执行微调或训练**。

## 输入、提示词与推理配置

固定提示词为 [`prompts/research_pv_zero_shot_v1.md`](../prompts/research_pv_zero_shot_v1.md)，SHA-256：`07db52bbd1317dfc9a2598a2e862b5cb5d610940da9ee6ab91da99eed31565a6`。每条输入包含站点、目标日期、额定容量、目标日 NWP 辐照度/温度/湿度/风速摘要、截至目标日前 3 天的 7 个历史等效满负荷小时数，以及目标日物理基线。**不包含目标日真实功率、真实气象、标签或未来日期数据**。提示词要求模型以物理预测为锚点，只输出 `forecast_full_load_hours` 一个数字；再按

$$\widehat E_{s,d}^{\mathrm{LLM}}=C_s\widehat H_{s,d}^{\mathrm{LLM}}$$

转换为日发电量 MWh，其中 $C_s$ 为 MW，$\widehat H$ 为小时。输出限在 0–24 小时。若 JSON 非法、字段不符、数字非有限或越界，脚本使用物理基线回退并计数；API 请求错误则停止，可继续运行而不重复已完成样本。本次**回退 0 条**。

使用 Ollama 本地 [`/api/generate`](https://docs.ollama.com/api/generate) 与 [JSON Schema 结构化输出](https://docs.ollama.com/capabilities/structured-outputs)，设置：`temperature=0`、`seed=42`、`num_ctx=4096`、`num_predict=80`、`think=false`、`stream=false`。先用前 4 条做接口和格式试运行；未改提示词及参数，随后完成全部 184 条。模型没有看到验证标签；本地评分脚本在推理之后读取标签。底层推理软件和硬件可能使逐字结果略有差异，模型 digest、提示词与运行参数已记录供复核。

## 指标与结果

与步骤 3 一致，在等效满负荷小时数上计算 MAE、RMSE 和平均偏差：

$$e_j=\widehat H_j-H_j,\quad \mathrm{MAE}=N^{-1}\sum_j|e_j|,\quad \mathrm{RMSE}=\sqrt{N^{-1}\sum_j e_j^2},\quad \mathrm{Bias}=N^{-1}\sum_j e_j.$$

| 站点 | 样本数 | 物理 MAE（h） | 大模型 MAE（h） | 物理 RMSE（h） | 大模型 RMSE（h） | 大模型 Bias（h） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S-1 | 46 | **0.7051** | 0.7388 | **0.8548** | 0.8858 | −0.5122 |
| S-2 | 46 | 1.0689 | **0.9987** | 1.3181 | **1.2270** | +0.8255 |
| S-3 | 46 | **0.6430** | 0.6488 | 0.8169 | **0.8132** | +0.0166 |
| S-4 | 46 | 0.7039 | **0.6956** | 0.8698 | **0.8578** | +0.0698 |
| **整体** | **184** | 0.7802 | **0.7705** | 0.9864 | **0.9601** | +0.0999 |

整体验证 MAE 差为 $0.7802-0.7705=0.0097$ 小时，约 **1.2%** 的相对下降；S-1 和 S-3 的 MAE 反而升高。大模型相对物理基线的平均有符号修正为 −0.0562 小时，平均绝对修正为 0.0688 小时，95% 分位绝对修正为 0.2036 小时。184 条中 102 条误差下降、82 条误差上升。

考虑同一日四站天气相关，按**日期块**先算四站平均 MAE 改善，再对 46 个日期做 10,000 次有放回重采样（随机种子 42）。改善量定义为 $|e^{\mathrm{physics}}|-|e^{\mathrm{LLM}}|$，正值表示大模型更好。均值 0.009716 小时，百分位法 95% bootstrap 区间为 **[−0.004580, +0.023276] 小时**；46 天中 28 天改善、18 天退化。区间跨零，说明这组验证数据**不足以支持稳定改善的结论**。这只是单一时间段的探索性区间，不能替代独立测试集。

## 复现、文件与限制

在原始数据已下载、步骤 2 和步骤 3 已生成的前提下，确认本地 Ollama 服务运行并已有模型，然后执行：

```bash
ollama list
.venv/bin/python scripts/evaluate_research_ollama.py --limit 4
.venv/bin/python scripts/evaluate_research_ollama.py
```

第二条命令会接续前 4 条；第三条后再次运行只重算评分，不重复请求。脚本为 [`scripts/evaluate_research_ollama.py`](../scripts/evaluate_research_ollama.py)。被 Git 忽略的 `data/processed/research/` 保存：

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `ollama_zero_shot_config.json` | 模型、提示词和数据文件校验值、生成参数、run ID | `d2703beb1b278aab3d98932c5cbdbe07314d616524180e20c6754b86815b4623` |
| `ollama_zero_shot_valid.jsonl` | 每条无标签预测、原始 JSON 响应、耗时、回退状态 | `1510a5813f4f3f51298d8a6aa51c7f41182288a314d97e9f0c8ef6436` |
| `ollama_zero_shot_summary.json` | 整体、逐站指标与日期块 bootstrap 结果 | `18935a10a03c620ad34fbe7c4505df5bda007e88d711214585ade65a8df7b76e` |

固定 run ID：`c4dddfc5b33b33537bef56266ff2c72797eddaeb500db42b6a4c55d2d3f30dc1`。平均单条接口耗时 1.692 秒，184 条累计约 311 秒（包括首次加载和本地请求开销）。无格式回退，所有请求正常结束。

本实验不能把“略优”解释成大模型已经学到可推广的光伏规律。模型是在通用预训练基础上的零样本推理，缺少专门训练；物理基线也使用 GHI 近似组件面辐照度，误差来源尚未分离。下一步可在**训练集 890 条**上设计带物理锚点的残差学习任务，先验证本机微调框架与内存是否可行，再与本步骤固定的零样本版本比较；验证集可用于选择方案，测试集只在方案冻结后评分。
