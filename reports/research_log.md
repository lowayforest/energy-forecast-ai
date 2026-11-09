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

## 2026-11-09：步骤 3，建立物理预测基线

- **目的**：给后续本地大模型实验提供可复现的预报天气驱动数值锚点，同时检查训练集站点校准是否能在未来日期有效。
- **来源与方法**：详见[步骤 3 报告](research_physics_baseline.md)。使用 [Ross 温度模型](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.ross.html)和 [PVWatts 直流功率模型](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html)，以 NWP GHI 近似组件面辐照度；NOCT 45 °C、$\gamma=-0.004\,^{\circ}\mathrm C^{-1}$、逆变器效率 0.96 均为实验假设。
- **关键公式**：$T_{\mathrm{cell}}=T_{\mathrm{air}}+(\mathrm{NOCT}-20)G/800$；$P_{\mathrm{dc}}=C_s(G/1000)[1+\gamma(T_{\mathrm{cell}}-25)]$；$E_{s,d}^{\mathrm{raw}}=0.25\sum_i\min(C_s,\max(0,0.96P_{\mathrm{dc},i}))$ MWh；$a_s=\operatorname{median}_{d\in\mathrm{train}}(E_{s,d}^{\mathrm{actual}}/E_{s,d}^{\mathrm{raw}})$，仅纳入原始预测至少 $0.1C_s$ MWh 的训练日。符号与单位完整定义见步骤 3 报告。
- **运行**：`.venv/bin/python scripts/build_research_physics.py`；使用 `pvlib 0.16.1`、`numpy 2.5.4`、`pandas 3.1.0`。输出无标签物理预测 `physics_{train,valid,test}.jsonl` 和含参数、哈希、训练及验证评分的 `physics_manifest.json`，位于 Git 忽略目录 `data/processed/research/`。
- **结果**：四站训练校准系数依次为 0.75732562、1.36628215、0.88630825、0.99456074。验证集 184 个站点日，整体 MAE 从 0.8142 降至 0.7802 等效满负荷小时，RMSE 从 1.0131 降至 0.9864 小时；**S-2 单站 MAE 从 0.9816 升至 1.0689 小时**。测试集 236 个站点日仅生成预测，未评分。
- **局限**：缺少可核验的倾角、朝向和设备参数，GHI 近似及固定温度/效率参数会带来系统误差；系数不是纯损耗率。NWP 逐条发布时间仍不可核查。S-2 的验证退化表明固定站点系数有跨季节风险。

## 2026-11-09：步骤 4，未微调本地大模型对照

- **目的**：固定零样本提示词和验证协议，评估本地 `qwen3.5:9b` 在物理基线之上的实际效果；详细过程与逐站结果见[步骤 4 报告](research_zero_shot_ollama.md)。
- **来源与配置**：[Qwen3.5-9B 上游模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)、[Ollama 模型库](https://ollama.com/library/qwen3.5)、[Ollama 结构化输出说明](https://docs.ollama.com/capabilities/structured-outputs)；本机模型 digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`。提示词为 `prompts/research_pv_zero_shot_v1.md`；`temperature=0`、`seed=42`、`think=false`、JSON Schema 输出。未进行训练或微调。
- **关键公式**：$\widehat E_{s,d}^{\mathrm{LLM}}=C_s\widehat H_{s,d}^{\mathrm{LLM}}$（MW × h = MWh）；$\mathrm{MAE}=N^{-1}\sum_j|\widehat H_j-H_j|$，$\mathrm{RMSE}=\sqrt{N^{-1}\sum_j(\widehat H_j-H_j)^2}$。同日四站相关，改进区间按日期块 bootstrap 10,000 次计算；完整定义见步骤 4 报告。
- **运行**：先 `.venv/bin/python scripts/evaluate_research_ollama.py --limit 4` 做格式试运行，再执行 `.venv/bin/python scripts/evaluate_research_ollama.py` 接续完成 184 条验证样本。运行文件、哈希、硬件和软件版本见步骤 4 报告；测试标签未访问。
- **结果**：物理基线验证 MAE 0.7802 小时、RMSE 0.9864 小时；未微调大模型 MAE **0.7705** 小时、RMSE **0.9601** 小时。S-1、S-3 的 MAE 升高。日期块 bootstrap 的 MAE 改善 95% 区间 **[−0.004580, +0.023276] 小时**，跨零，不能声称稳定优于物理基线。184 条均有有效 JSON，无回退。
- **局限**：只是一段验证期的零样本结果，改善幅度很小；物理基线的 GHI 近似误差仍在。尚未亲自训练模型，也未进行测试集最终评估。

## 2026-11-09：步骤 5，物理残差样本与本机 QLoRA 试运行

- **目的**：用训练集构建监督残差任务，验证 16 GB Apple M5 能否真正反向传播 9B 量化模型；详细过程见[步骤 5 报告](research_qlora_feasibility.md)。
- **来源与方法**：[MLX-LM 微调文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)、[Qwen3.5-9B-4bit 权重](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit)、[LoRA 论文](https://arxiv.org/abs/2106.09685)、[QLoRA 论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract.html)。固定权重 revision `8b2b98c00a6b4d291155e4890773ca8f769aee53`，两块权重 SHA-256 见报告。
- **关键公式**：$r_{s,d}=H_{s,d}^{\mathrm{actual}}-E_{s,d}^{\mathrm{cal}}/C_s$（h）；$\widehat E_{s,d}=C_s\operatorname{clip}(E_{s,d}^{\mathrm{cal}}/C_s+\widehat r_{s,d},0,24)$（MWh）。LoRA 通过低秩矩阵学习冻结权重的增量，概念公式与符号见报告。
- **运行**：`.venv/bin/python scripts/build_research_sft.py` 生成训练 890 条、验证 184 条聊天格式样本；无测试文件。安装 `mlx-lm[train] 0.32.2`、`mlx 0.32.4`。用前 16/8 条样本、512 token、batch 1、最后 2 层、秩 8、梯度检查点运行 **2 次参数更新**；完整命令与哈希见报告。
- **结果**：样本最长 438 tokens，无截断。9B 模型两步训练成功，训练损失 1.938→1.741，保存 5.31 MB adapter；30 个张量均非零。另一次单步测得 macOS 进程峰值内存占用约 **11.93 GiB**、墙钟 **12.93 秒**、无交换。此结果仅证明短训练可执行，尚未评估微调后预测精度。
- **局限**：长时间训练可能遇到内存或速度问题；短训练的损失下降不能作为泛化结论。MLX 与 Ollama 量化格式不同，未来须在同一 MLX 基座上做微调前后对照；测试集继续封存。

## 2026-11-09：步骤 6，同一 MLX 基座的微调前后验证对照

- **目的**：在相同四站、相同 46 个验证日期和相同提示/解析规则下，检验亲自训练的残差 adapter 是否降低日发电量预测误差；完整配置、命令、哈希、逐站结果见[步骤 6 报告](research_mlx_finetune_v1.md)。
- **方法与公式**：MLX 量化 `Qwen3.5-9B-4bit` 固定基座，训练最后 2 层 rank 8 LoRA，学习目标为 $r_{s,d}=H_{s,d}^{\mathrm{actual}}-H_{s,d}^{\mathrm{physics}}$；推理后 $\widehat E_{s,d}=C_s\operatorname{clip}(H_{s,d}^{\mathrm{physics}}+\widehat r_{s,d},0,24)$。训练集共 890 条监督样本，本次进行 256 次单样本更新，尚未完成一轮遍历；验证集 184 条仅用于训练期间的语言损失观察与训练后的预测评分。按日期块 bootstrap 10,000 次估计配对 MAE 改善区间。
- **运行**：`bash scripts/train_research_qlora.sh`；`.venv/bin/python scripts/evaluate_research_mlx.py --variant base` 和 `--variant adapter`；`.venv/bin/python scripts/compare_research_mlx.py`。最终 adapter、预测与日志在 Git 忽略目录，SHA-256 见步骤 6 报告。
- **结果**：256 步训练用时 1,058.29 秒，验证语言损失 1.910→0.539。物理、MLX 基座、微调后验证 MAE 依次为 **0.7802、0.8398、0.7907 h**；RMSE 为 **0.9864、1.0399、0.9795 h**，两组模型各 184 条均 0 回退。微调相对基座的 MAE 改善 95% 日期块 bootstrap 区间为 **[−0.015092, +0.124738] h**，相对物理为 **[−0.054620, +0.031039] h**，均跨零。逐站差异见报告。
- **局限**：256 次更新不足一个轮次；语言损失下降不等于预测误差改善有统计把握；训练和比较都观察了验证集，测试集仍封存。河北四站实验不能直接外推至上海场站；完整训练峰值内存未可靠测得。

## 2026-11-09：步骤 7，逐站与天气条件下的验证误差诊断

- **目的**：判断 256 步 adapter 为何相对物理基线的 MAE 未改善，并明确下一次受控实验的假设；详细数字与分组边界见[步骤 7 报告](research_error_analysis_v1.md)。
- **方法与公式**：只读 890 条训练、184 条验证及其物理预测和验证期两组 MLX 预测。真实残差 $r=H^{\mathrm{actual}}-H^{\mathrm{phys}}$，实际修正 $c=H^{\mathrm{adapt}}-H^{\mathrm{phys}}$，逐条 MAE 改善 $\Delta=|r|-|H^{\mathrm{actual}}-H^{\mathrm{adapt}}|$。用训练集 GHI 上三分位 3.5408 kWh/m² 定义高辐照度条件，并描述验证期天气三分组。
- **运行**：`.venv/bin/python scripts/analyze_research_errors.py`；脚本校验六个输入文件哈希和日期边界。机器可读结果在 Git 忽略目录 `data/processed/research/error_analysis_valid.json`，SHA-256 `db395b43599304202905373710873a6dfbe5fd344a0294c74c848c8c7340134a`。
- **结果**：验证期真实残差与模型修正的标准差分别为 0.9740、0.2237 h，逐条相关系数 0.0901。S-1 真实残差均值 +0.4836 h，模型平均修正 −0.1176 h，MAE 变差；S-2 真实残差均值 −0.9342 h，模型修正 −0.2577 h，MAE 略好。验证期平均 NWP GHI 5.0097，高于训练期的 3.1226 kWh/m²；81% 的验证样本超过训练期 GHI 上三分位。训练/验证高 GHI 组的 S-1 残差均偏正，S-2 均偏负。验证期低/中 GHI 组 adapter MAE 变差，高 GHI 组改善。
- **局限与决策**：分组和个案分析是探索性的，不能断言因果；验证集不能当最终独立测试。下一步优先检验训练集拟合的站点与辐照度条件物理校正，再训练大模型修正剩余残差，并分别报告物理增强基线和大模型效果。测试集继续封存。

## 后续步骤

步骤 8：基于步骤 7 的诊断，预先固定站点与预报辐照度条件相关的物理残差校正及大模型剩余残差方案；只用训练集拟合、验证集比较，方案冻结后才做一次独立测试评分。论文撰写时补齐并核对 10—15 篇参考文献。
