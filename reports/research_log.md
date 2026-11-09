# 光伏预测课程论文实验日志

研究分支：`research/ollama-pv-paper`。每一步在本日志登记目的、数据来源、处理方法与关键公式、运行命令、结果、局限；详细内容链接到对应报告或脚本。原始数据及大体积中间结果保存在 Git 忽略目录，日志记录下载地址和校验值，便于论文复现。

论文参考文献目标为 **10—15 篇**，优先引用数据集原论文、物理模型原始方法、大模型微调方法和直接相关的光伏预测实验。每篇文献须在正文中有明确用途，逐条核对作者、年份、题名与 DOI/正式链接；不为凑数量加入无关文献。关键公式要保留符号定义、单位和来源。

## 2026-11-09：步骤 0，筛选数据集

- **目的**：为“物理基线 + 本地大模型修正”的次日光伏发电量预测找到有真实功率和提前发布天气预报的数据。
- **来源**：[PVOD 原始数据与论文](https://github.com/yaotc/PVODataset)、[AI-PVOD 整理数据](https://huggingface.co/datasets/yujiaA/AI-PVOD)；候选对照见[数据集调研报告](research_dataset_selection.md)。
- **操作**：下载 AI-PVOD `full_dataset.zip` 到 `data/raw/research/ai_pvod_full_dataset.zip`；SHA-256：`ba0639dc3c69c49b425f36877e5aa8f2923ca5d1fff6ca9a76a5aed17ba53383`。
- **结论**：选用四个河北站点的数据作主实验；现有 HKUST 数据保留为独立场景。原始压缩包不提交到 Git。

## 2026-11-09：步骤 1，数据单位与时间审计

- **目的**：确认目标量、时区、记录完整性和 NWP 的预测属性，避免日发电量计算错误或使用未来信息。
- **依据**：[原 PVOD 论文](https://doi.org/10.1016/j.solener.2021.09.050)定义 `power` 的单位为 MW，并说明 NWP 每日运行、使用未来约 28–54 小时的预报；[AI-PVOD 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13424122/)列出四站额定容量。
- **运行**：`.venv/bin/python scripts/audit_research_dataset.py`。
- **关键公式**：`Time` 是北京时间，`Timestamp` 是 UTC，逐行验证

  $$t_{\mathrm{BJT}}=t_{\mathrm{UTC}}+8\,\mathrm{h}.$$

  用 15 分钟功率样本近似计算站点 $s$ 在北京时间日期 $d$ 的发电量：

  $$E_{s,d}^{\mathrm{MWh}}=\sum_{i\in d}P_{s,i}^{\mathrm{MW}}\Delta t,\qquad \Delta t=0.25\,\mathrm{h};\qquad E_{s,d}^{\mathrm{kWh}}=1000E_{s,d}^{\mathrm{MWh}}.$$

- **结果**：S-1 至 S-4 各有 302、348、348、348 个完整日，合计 **1,346 个站点日**；每个北京时间日 96 条、时间连续无重复、`power` 与 7 个 `nwp_*` 字段没有缺失。四站容量分别为 6.6、20、20、20 MW；功率范围没有超出对应容量。详细数字见[数据集调研报告](research_dataset_selection.md#第一步单位与时间审计)。
- **局限**：整理 CSV 只有预报的目标时刻，没有逐条 NWP 运行时间。预报“提前可获得”的依据来自原论文，不能把逐条发布时间写成已由 CSV 验证。目标日的 `power`、`lmd_*` 只用于标签和评分，不进入当天预测输入。

## 2026-11-09：步骤 2，按时间划分并生成每日样本

- **目的**：为次日发电量任务建立输入与标签分离、按时间顺序切分的数据。
- **方法与公式**：见[步骤 2 报告](research_daily_dataset.md)。以 $E_{s,d}=0.25\sum_{i\in d}P_{s,i}$（MWh）积分日发电量，以 $H_{s,d}=E_{s,d}/C_s$（小时）归一化不同容量站点；历史输入仅取 $d-9$ 至 $d-3$ 的 7 个完整日。
- **运行**：`.venv/bin/python scripts/build_research_daily.py`。
- **结果**：训练 890、验证 184、测试 236 条，共 1,310 条。输出为 `data/processed/research/daily_{train,valid,test}.jsonl` 和 `daily_manifest.json`；文件哈希、字段来源和划分日期见[步骤 2 报告](research_daily_dataset.md)。另逐条核验 1,310 个站点日不重复、历史日期恰为 $d-9$ 至 $d-3$，且 `inputs` 不含目标日发电标签，检查通过。
- **局限**：逐条 NWP 运行时间仍不可从 CSV 独立核查；物理预测尚未计算。测试集保留给方案冻结后的最终评估。

## 后续步骤

步骤 3：确定河北站点的物理模型参数，只在训练集校准，并比较未校准与校准后的基线。随后做未微调模型与 QLoRA 微调模型的对照。未执行的步骤不写入实验结果。

现有平台的物理基线采用 [Ross 温度模型](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.ross.html)和 [PVWatts 直流功率模型](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html)：

$$T_{\mathrm{cell}}=T_{\mathrm{air}}+\frac{\mathrm{NOCT}-20}{800}G,\qquad P_{\mathrm{dc}}=P_{\mathrm{STC}}\frac{G}{1000}\left[1+\gamma(T_{\mathrm{cell}}-25)\right].$$

这里 $G$ 的单位为 W/m²。现有香港演示用 GHI 近似组件面辐照度，并做经验系数校准；河北站点的参数与辐照度处理仍需单独确定，不能直接复用香港参数。
