# 日前光伏发电量预测研究

`research/ollama-pv-paper` 分支用于人工智能课程论文和本课题组探索者计划项目的光伏预测拓展研究。这里研究的是河北四座公开光伏电站的**次日总发电量**：先用目标日天气预报计算并校正物理预测值，再检验 9B 大语言模型能否进一步修正误差。线上日、月预测演示及其部署代码见 [`main` 分支](https://github.com/lowayforest/energy-forecast-ai/tree/main)。

本分支保留可复现的代码、提示词和依赖说明。论文、实验记录、检测报告及生成的图表放在本地 `reports/` 或仓库外，不纳入版本控制。

## 当前进展

- 完成四站数据审计、按日期构造样本、物理预测与条件校正，并按时间划分训练、验证和测试集。
- 使用四位量化的 Qwen3.5-9B，在最后两层训练 rank 8 LoRA adapter，共更新 890 步。模型预测残差所属的 A/B/C 三档，再用类别概率和训练期残差中位数计算连续修正量。
- 完成同权重未微调基座、LoRA 模型和两种物理方法的对照。**主测试集上未微调基座最好，LoRA 微调尚未带来稳定增益。**
- 补做只监督类别 token 的 v4 消融，以及 XGBoost 和物理残差斜率探索。它们用于分析失败原因；旧测试集已被查看，后续方案需要新的独立时间段检验。

## 数据与预测边界

数据来自 [PVOD](https://github.com/yaotc/PVODataset)，使用 [AI-PVOD 整理的四站数据](https://huggingface.co/datasets/yujiaA/AI-PVOD/tree/main)。四站位于河北，额定容量分别为 6.6、20、20、20 MW；本研究没有上海场站数据。原始 15 分钟交流功率积分为日发电量，再除以额定容量得到等效满负荷小时，作为跨站评价单位。

从 1,346 个完整站点日构建 1,310 条带历史窗口的样本。训练集 890 条（截至 2019-02-28），验证集 184 条（2019-03-01 至 04-15），测试集 236 条（2019-04-16 至 06-13）。四站的同一日期始终属于同一集合。评分前曾意外看到 2019-04-16 的一条真实标签，因此主测试排除该日期的**全部四站样本**，使用之后 58 天、232 条样本。

预测目标日时，只使用该日的 `nwp_*` 天气预报和目标日前第 9 天至第 3 天的七个完整日历史发电量。目标日功率及现场实测 `lmd_*` 天气仅用于形成标签或事后分析。整理后的 CSV 没有保存每条 NWP 的发布时间，因此无法逐条核验其在固定日前时点的可用性。

## 方法与结果

物理部分用数值天气预报中的辐照度和气温驱动 Ross 组件温度模型与 PVWatts 功率近似；训练期数据用于估计场站比例及高、低辐照条件下的平均偏差。大模型接收校正后的物理预测值、天气和历史发电信息，预测剩余误差。未微调基座与 LoRA 模型使用相同的量化权重、提示和解码规则。

下表为原先固定方案在主测试集 232 条样本上的结果。MAE、RMSE 的单位都是等效满负荷小时（h）；某站的预测日发电量（MWh）可由预测小时数乘以该站额定容量（MW）得到。

| 方法 | MAE（h） | RMSE（h） |
| --- | ---: | ---: |
| 原校准物理 | 1.0253 | 1.3507 |
| 条件物理 | 0.8622 | 1.1376 |
| 未微调 9B 基座 + 条件物理 | **0.7960** | **1.0441** |
| LoRA 微调 9B + 条件物理 | 0.8437 | 1.1110 |

LoRA 比条件物理的 MAE 低 0.0185 h，但比未微调基座高 0.0477 h。验证期 LoRA 的 MAE 为 0.6757 h，基座为 0.7035 h；这个排序没有延续到主测试期。只监督类别 token 的 v4 在旧测试集上的回顾性 MAE 为 0.8422 h，仍高于基座；该数字不能作为新的独立测试结果。XGBoost 在验证期的探索性 MAE 为 0.8776 h，尚无独立测试分数。这些结果只适用于上述数据与评估设置，不能推出上海场站的部署精度。

## 复现

原实验在 16 GB Apple M5 上运行，使用 Python 3.12、MLX-LM 0.32.2。`requirements-research.txt` 固定主要依赖；macOS 上运行 XGBoost 还需要 OpenMP 运行库。以下命令从仓库根目录执行：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-research.txt
mkdir -p data/raw/research
curl -L --fail --retry 3 \
  https://huggingface.co/datasets/yujiaA/AI-PVOD/resolve/main/full_dataset.zip \
  -o data/raw/research/ai_pvod_full_dataset.zip
shasum -a 256 data/raw/research/ai_pvod_full_dataset.zip
```

原始 ZIP 的预期 SHA-256 为 `ba0639dc3c69c49b425f36877e5aa8f2923ca5d1fff6ca9a76a5aed17ba53383`。核对后构造数据、物理预测和训练样本：

```bash
.venv/bin/python scripts/audit_research_dataset.py
.venv/bin/python scripts/build_research_daily.py
.venv/bin/python scripts/build_research_physics.py
.venv/bin/python scripts/build_research_conditional.py
.venv/bin/python scripts/build_research_sft_v2.py
```

下载固定版本的 [Qwen3.5-9B 四位量化权重](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit)：

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mlx-community/Qwen3.5-9B-4bit",
    revision="8b2b98c00a6b4d291155e4890773ca8f769aee53",
    local_dir="data/cache/research/mlx_qwen35_9b_4bit",
)
PY
```

随后运行 v3 训练、验证与原先固定方案的测试评分：

```bash
bash scripts/run_research_v3.sh
.venv/bin/python scripts/prepare_research_final_test.py
.venv/bin/python scripts/evaluate_research_final_test.py --variant base
.venv/bin/python scripts/evaluate_research_final_test.py --variant adapter
.venv/bin/python scripts/score_research_final_test.py
```

`run_research_v3.sh` 只训练并评估验证集；后续命令才生成测试预测与评分。测试脚本会检查模型、输入和 adapter 的哈希。已有训练结果、旧测试集及验证集都经历过分析，改进方案须使用新的独立数据检验。

## 仓库内容

- `scripts/`：数据审计、物理预测、样本构造、训练、评分、补充实验和绘图代码。
- `prompts/`：零样本、连续残差与分档任务的提示词。
- `requirements-research.txt`：研究环境的主要 Python 依赖。
- `data/`：本地下载的数据、模型权重、训练产物和日志，已被 Git 忽略。
- `reports/`：本地论文、实验记录与生成图表，已被 Git 忽略；克隆仓库后不会包含该目录。

研究分支地址：<https://github.com/lowayforest/energy-forecast-ai/tree/research/ollama-pv-paper>。
