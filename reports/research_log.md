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

## 2026-11-09：步骤 8，条件物理校正与 9B 剩余残差微调

- **目的与方案**：检验步骤 7 发现的站点与预报 GHI 条件误差能否由训练集偏差校正缓解，以及同一 9B 模型能否进一步修正剩余残差；[运行前方案](research_conditional_v2_protocol.md)与[结果报告](research_conditional_v2_results.md)分别保存，以便核查事前约定和事后解释。
- **方法与公式**：训练集全局 GHI 上三分位 3.54083333 kWh/m² 划分高/低，逐站求训练残差均值 $m_{s,b}=n_{s,b}^{-1}\sum_{d\in\mathcal T_{s,b}}(H_{s,d}^{\mathrm{actual}}-H_{s,d}^{\mathrm{phys}})$；训练样本用留一均值，验证样本只用完整训练均值。条件物理锚点 $H^{\mathrm{cond}}=\operatorname{clip}(H^{\mathrm{phys}}+m_{s,b},0,24)$；同一 MLX 9B 学习剩余残差。固定比较原物理、条件物理、旧 adapter、新基座、新 adapter。
- **运行与复核**：用户运行 `bash scripts/run_research_v2.sh`，256 步训练用时 927.02 秒；训练语言损失 1.904→0.549。两组新模型各完成 184 条验证推理且 0 回退。重新运行 `.venv/bin/python scripts/compare_research_mlx_v2.py` 后哈希不变；条件物理预测和新模型预测均未含实测标签。模型、输入、输出和日志 SHA-256 见结果报告；测试集没有读取或评分。
- **结果**：原物理、条件物理、旧 adapter、新基座、新 adapter 的验证 MAE 依次为 **0.7802、0.6834、0.7907、0.7759、0.7083 h**。条件物理对原物理的 MAE 改善 +0.096764 h，日期块 bootstrap 95% 区间 **[+0.052904,+0.139349] h**；新 adapter 对条件物理的改善 **−0.024877 h**，区间 **[−0.051481,+0.001484] h**。新 adapter 的 MAE 四站均未低于条件物理。大模型没有证明额外精度贡献。
- **局限与决策**：步骤 7 已依据相同验证集设计本轮方案，因此验证区间不是独立确认；256 步不足完整一轮，河北四站不能代表上海。训练损失下降不等于预测改善。测试集继续封存，下一步应先决定是否开展新的受控大模型实验或如实以条件物理预测为当前最佳验证方案。

## 2026-11-10：步骤 9，9B 残差分档实验

- **目的**：检验 9B 模型能否在相同验证集上超过条件物理 MAE 0.6834 h；完整设计与文献动机见[步骤 9 方案](research_class_v3_protocol.md)，数字、逐站表现、区间和文件哈希见[步骤 9 结果](research_class_v3_results.md)。
- **方法与公式**：训练集残差按三分位分为 A/B/C，边界约为 −0.3944、+0.3250 h；模型训练预测单 token 类别，推理时用三类下一 token 概率与各类训练中位数的加权和 $\widehat r=\sum_k p_k\widetilde r_k$ 修正条件物理锚点。固定同一 9B 基座的 890 步 QLoRA、单样本 batch、最后 2 层 rank 8 和种子 42；主结果用最终 adapter，按日期块比较条件物理、新基座及旧 adapter。
- **内存修订与执行**：首次运行在 0/890 步触发 Metal 内存不足，无有效训练结果；保留失败日志和旧输入哈希。缩短提示词后样本最长由 459 降至 386 tokens，最长样本的 1 步 QLoRA 试运行成功，完整记录见步骤 9 方案。用户授权助手运行 `bash scripts/run_research_v3.sh`，最终完成 890/890 步，耗时 2484.38 秒；基座和 adapter 分别完成 184 条验证预测，均 0 回退。复算比较 JSON 后 SHA-256 不变，权重、输入、输出哈希核对通过。测试集继续封存。
- **结果与判断**：原物理、条件物理、步骤 8 adapter、新基座、新 adapter 的验证 MAE 依次为 **0.7802、0.6834、0.7083、0.7035、0.6757 h**。新 adapter 相对条件物理的点改善 **+0.007695 h（约 28 秒）**，46 个日期块 bootstrap 95% 区间 **[−0.005798,+0.020153] h**，跨零；相对新基座区间也跨零。因此只能称为验证集探索性小幅改善，不能声称稳定的大模型增益。修正与真实剩余残差的逐条相关约 0.1777，模型对每日误差的辨别仍有限。
- **下一步**：将此最终 adapter 暂作候选，停止在同一验证集上挑选检查点或调阈值；最终方案冻结后再一次性评分封存测试集。论文要保留全部固定对照、局限和 10—15 篇核对过的参考文献。

## 2026-11-10：步骤 10，冻结模型的最终测试与文献核对

- **方案**：最终第 890 步 adapter、A/B/C 解码和训练期条件物理校正固定，方案与测试脚本先于评分提交为 `01352d9`；详细边界见[步骤 10 方案](research_final_test_protocol.md)。为查看 JSON 结构，评分前意外打印 S-1 在 2019-04-16 的一条真实标签；没有据此调参，预先将该完整日期块从主结果排除，主测试为后续 58 天、232 条，全 59 天、236 条另列补充。
- **执行和复核**：`prepare_research_final_test.py` 先生成无标签条件物理预测和 system/user 测试提示；`evaluate_research_final_test.py` 分别运行基座与最终 adapter，均 236/236 条、0 回退；所有预测完成后才用 `score_research_final_test.py` 读取测试标签。预测的唯一站点日、无目标标签和来源哈希检查通过，比较 JSON 复算 SHA-256 不变；文件哈希见[步骤 10 结果](research_final_test_results.md)。
- **主测试结果**：原物理、条件物理、未微调基座、最终 adapter 的 MAE 分别为 **1.0253、0.8622、0.7960、0.8437 h**。最终 adapter 相对条件物理改善 **+0.018547 h**，58 个日期块 bootstrap 95% 区间 **[+0.009800,+0.027509] h**；相对未微调基座 **−0.047704 h**，区间 **[−0.076986,−0.017547] h**。所以本次微调未超过基座，验证集的模型排序没有保持到测试期。
- **事后诊断与后续**：测试期条件物理正偏差 +0.4520 h，高于验证期 +0.0340 h；基座平均负修正约 −0.2340 h，最终 adapter 约 −0.0273 h，更接近验证期剩余残差均值。这解释了测试期基座的偏差优势，但不是可据测试集再调参的规则。已核对[15 条参考文献候选](research_references.md)的出处和用途。下一次方法优化需用训练期滚动划分，并获取新的独立评估期；当前测试期不可重复用于独立确认。

## 2026-11-10：步骤 11，训练期滚动 XGBoost 对照与论文初稿

- **目的与方案**：作为“物理 + 大模型”以外的探索性小模型对照，只用训练期四个滚动折选三个预定 XGBoost 配置，不读取测试期；方案及数据哈希见[步骤 11 方案](research_xgb_exploration_protocol.md)，并先于运行提交 `14a9b23`。
- **执行**：安装 `xgboost==3.4.1` 与本机 `libomp 23.1.5`。首次导入因缺动态库失败；第二次在训练和预测后因误用评分函数接口失败。修复函数导入后完整运行，保留两个失败日志，未改候选配置。完整重跑两次的模型、预测与 manifest 哈希一致。
- **结果**：训练期四折平均 MAE A/B/C 为 **0.7358/0.7057/0.7285 h**，选 B（深度 3、学习率 0.05、200 树）；春季验证 MAE **0.8776 h**、偏差 **−0.5904 h**，弱于条件物理 0.6834 h 和最终 9B adapter 0.6757 h。结果只说明这个具体回归方案跨季节低估，不能泛化为大模型优于所有传统方法；详见[步骤 11 结果](research_xgb_exploration_results.md)。不对已使用的测试期做新模型选型或独立评分。
- **写作与后续数据**：已完成[课程论文终稿](research_paper_final.md)，含关键公式、验证/测试对照、数据限制及在正文实际使用的 15 条[核对过的参考文献](research_references.md)。给导师讨论的[新数据需求与优化路线](research_next_data_requirements.md)明确了预报发布时间、场站物理参数、上海外部验证，以及日前日预测与真正月前预测的区别。

## 2026-11-10：步骤 12，类别 token 单目标微调消融

- **假设与方案**：核查步骤 9 的 0.183 语言损失包含模板标记的问题，只对真实 A/B/C token 监督，其余 9B 模型、样本、提示、参数与 890 次更新不变；[运行前方案](research_class_only_v4_protocol.md)先于完整训练提交。
- **实现核查**：全部 890/184 条训练/验证样本每条恰一个受监督类别 token。首次单步试运行发现 `mlx-lm` 默认长度掩码多计一个补齐位置，保留失败日志后改用严格右端掩码，再完成 890/890 次更新，无内存错误。权重、预测和审计哈希见[步骤 12 结果](research_class_only_v4_results.md)。
- **结果**：v4 验证 MAE **0.6774 h**，v3 为 0.6757 h；受限三类交叉熵 v4 1.0935、v3 1.0856。旧测试 232 条 v4 MAE **0.8422 h**，v3 0.8437 h，未微调基座 0.7960 h。v4 对 v3 的微小差异区间跨零，不能声称精度提升；旧测试数字均是**回顾性诊断**，不是新的独立测试。
- **判断**：修复损失目标提升了训练定义的准确性，但没有产生可靠预测增益。模型修正仍近似常数；物理锚点的跨季节站点偏差是下一步应检查的问题。

## 2026-11-10：步骤 13，训练期滚动物理斜率补丁

- **方案**：先提交[运行前方案](research_physics_slope_protocol.md)，在训练期四个滚动折内重新拟合物理比例、辐照分组和分组残差均值，比较原条件物理与三个岭斜率候选。旧测试完全不读。
- **结果**：原条件物理四折平均 MAE **0.8589 h**，斜率候选最优为 **0.8523 h**，仅改善 0.0066 h，未达到预定 0.01 h 门槛，选择原条件物理。春季验证所选模型与原基线相同，MAE 0.6834 h。完整分折与文件哈希见[步骤 13 结果](research_physics_slope_results.md)。
- **判断**：简单辐照度斜率不足以可靠修复跨月偏差。下一次性能确认须在新数据上预先划分未见日期，尤其应获得 NWP 发布时间和场站设备/运行状态。

## 2026-11-10：步骤 14，另一个独立场景的数据审计

- **来源**：[罗马尼亚双站光伏数据](https://github.com/upb-nlp/PV-Generation-Dataset)，revision `e14509a8c7f8d12a63afbedb052a8f6caa8c98fa`；下载两站逐小时真实功率和 `lead1` 提前一天归档预报到 Git 忽略目录。
- **审计**：A/B 站有记录日期分别为 543/396 天，真实功率无缺失的日期为 **532/388 天**；所有逐小时记录均匹配 `lead1` 预报时间。原始文件、缺失字段、夏令时和哈希见[步骤 14 报告](research_romanian_data_audit.md)。
- **边界**：只检查数据结构与可用性，没有训练、推理或评分。逐条发布时刻仍不在 CSV 中；日前固定时点、小时初末对齐、缺失日和未来新测试期须在新实验开始前确定。该数据不能代表上海。

## 2026-11-10：步骤 15，论文方法图和结果图

- **目的**：为课程论文提供可直接排版的学术风格插图，分别说明“物理 + 本地 9B”的数据路径与真实预测结果，而不在图中暗示不存在的微调收益。
- **制作**：使用 Matplotlib 从固定脚本生成 300 dpi PNG、可编辑文字的 SVG 和矢量 PDF。方法图明确区分预测时信息与训练/评分所用的目标日真实值；结果图直接读取原冻结 v3 验证和测试评分，并核验来源哈希。图 2 的新增“基座相对条件物理”区间使用相同的 58 个日期块与 10,000 次 bootstrap。
- **结果与边界**：两图已经嵌入[论文终稿](research_paper_final.md)，生成命令、数据来源与区间解释见[插图记录](research_figures.md)。结果图不把事后 v4 消融误标为独立测试；方法图对应 v3 adapter，测试期首日排除规则在图注中说明。

## 2026-11-10：步骤 16，课程论文终稿

- **题目与定位**：[《物理模型与大语言模型结合的日前光伏发电量预测研究》](research_paper_final.md)；课程论文终稿采用 Markdown，正文与摘要约 5,980 个汉字（不含参考文献）。按引言、数据与任务、方法、实验结果、讨论和结论组织，两张经核对的论文插图直接嵌入。
- **内容核对**：正文所用训练/验证/主测试 MAE 与冻结评分 JSON 一致；v4 数字明确标为旧测试期回顾性诊断，XGBoost 与物理斜率实验标为探索性对照。15 条参考文献均在正文出现，题目不使用“本地大语言模型”，正文仍准确说明本机 9B 量化 LoRA 实现。
- **项目关联与复现**：讨论部分按用户提供的项目背景注明本研究是课题组探索者计划项目中光伏发电预测任务的拓展，并给出项目 GitHub 的 `research/ollama-pv-paper` 分支链接；说明仓库包含代码、方案、图表脚本和哈希记录，大体积原始数据与权重未入库。

## 后续步骤

已完成冻结模型的最终测试、类别目标消融、训练期滚动物理校正探索，并完成新数据可用性审计。论文定稿如实呈现验证集小幅改善、测试期 adapter 弱于基座、v4 消融无收益、数据与物理假设限制。下一轮可在罗马尼亚数据或今后获取的上海场站数据上预先固定全新独立评估期；这不改变现稿的河北实验结论。
