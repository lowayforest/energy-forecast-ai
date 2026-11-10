# 面向日前光伏发电量预测的物理模型与本地大语言模型残差修正研究

> 课程论文初稿；数据、方法与结果已按本仓库实验记录复核。投稿前需按学校模板排版，并由作者核对课程要求和参考文献格式。

## 摘要

日前光伏发电量预测同时受到天气预报误差、场站参数缺失和季节变化影响。本文以公开的河北四座光伏电站数据为例，研究将简化物理模型与本地 9B 大语言模型结合的方法。首先，以数值天气预报的辐照度和温度驱动 Ross 组件温度模型及 PVWatts 直流功率模型，并用训练集进行站点和辐照度条件校正。随后，构建基于 Qwen3.5-9B 的剩余误差三档预测任务，通过低秩微调和类别概率加权恢复连续修正量。样本按时间划分为训练 890 条、验证 184 条和测试 236 条，最终测试方案先于评分固定。验证集上，微调模型的等效满负荷小时 MAE 为 0.6757 h，条件物理为 0.6834 h，改善幅度较小且日期块区间跨零。排除一日已意外暴露标签后，独立测试期 232 条样本的 MAE 分别为条件物理 0.8622 h、未微调基座 0.7960 h、微调模型 0.8437 h。微调模型优于条件物理，却弱于未微调基座，说明本次微调尚未产生稳定的额外泛化收益。本文展示了物理锚点与本地大模型结合的可复现实验流程，也指出数据可用时间、场站物理参数和跨季节偏差是进一步改进的关键。

**关键词**：光伏发电量预测；日前预测；物理模型；大语言模型；低秩微调；残差修正

## 1 引言

光伏出力随辐照度和天气变化，日前预测需要同时利用未来天气信息和历史发电记录。已有综述按预测时间尺度梳理了物理、统计和混合方法 [3]；针对数值天气预报的多模型比较也表明，传统机器学习是不可忽略的对照 [4]。本研究使用的 PVOD 原始数据及其 AI-PVOD 整理版本分别见 [1]、[2]。整理数据提供四座场站的功率与天气字段，适合构建一个公开、可重复的日前发电量实验，但不能直接代表上海场站。

大模型在时间序列预测中的有效性仍有争议。数字分词可能显著影响输出 [9]，Chronos 探索了数值离散化 [11]；相反，对多个方法的消融研究提示应与不用大模型的基线直接比较 [12]。据此，本文不把“使用了 9B 模型”视为性能证据，而把问题设定为：**在同一条件物理锚点上，未微调大模型和微调后的大模型能否降低独立时间段的发电量误差？**

本文的主要工作是：（1）明确 NWP、历史功率与目标日实测的时间和字段边界；（2）建立可解释的简化物理预测及训练期条件校正；（3）将连续剩余误差转为 A/B/C 三档单 token 任务，并用本地 9B 模型的类别概率恢复数值预测；（4）同时报告验证与独立测试表现、逐站误差、成对日期块区间及失败案例。本文定位为课程研究原型，不声称已经形成上海地区可部署的场站模型。

## 2 数据与任务定义

### 2.1 数据来源和时间划分

使用 [1] 的 PVOD 数据经 [2] 提供的 AI-PVOD 格式整理后的四站数据。站点额定容量分别为 6.6、20、20、20 MW。原始功率为 15 分钟间隔的 MW 数值，以北京时间日 $d$ 积分得到日发电量，随后用额定容量归一化：

$$E_{s,d}=0.25\sum_{i\in d}P_{s,i}^{\mathrm{actual}}\quad(\mathrm{MWh}),\qquad H_{s,d}=E_{s,d}/C_s\quad(\mathrm{h}).\tag{1}$$

其中 $C_s$ 是站点 $s$ 的额定容量。预测目标为次日 $E_{s,d}$，评估采用 $H_{s,d}$，使不同容量站点具有可比较的误差单位。时间划分为训练期截至 2019-02-28（890 个站点日）、验证期 2019-03-01 至 2019-04-15（184 个）、测试期 2019-04-16 至 2019-06-13（236 个）。不随机打散日期，四站同日属于同一时间块。详细字段和 SHA-256 见[数据构建记录](research_daily_dataset.md)。

### 2.2 预测时可见的信息

输入包括目标日 `nwp_*` 天气预报的辐照度、温度、湿度和风速摘要，以及 $d-9$ 至 $d-3$ 的七天历史发电量。目标日实测功率和 `lmd_*` 现场实测天气只用于形成标签与事后评分，不进入模型。原始 PVOD 论文描述了每日运行的 WRF 预报和约 28—54 小时的预报提前量 [1]；**整理 CSV 没有逐条 NWP 发布时间**，因此每个样本在实际日前调度时是否恰好可得，不能由 CSV 独立核验。本研究采用历史末日为 $d-3$ 的保守输入边界，仍把预报发布时间缺失列为实质限制。

## 3 方法

### 3.1 天气驱动的简化物理模型

借助 `pvlib` [6] 实现 Ross 组件温度与 PVWatts 直流功率近似 [5]。对每个 15 分钟时刻，以目标日 NWP 水平面总辐照度 $G_{s,i}$（W/m²）与空气温度 $T_{s,i}^{\mathrm{air}}$（°C）计算：

$$T_{s,i}^{\mathrm{cell}}=T_{s,i}^{\mathrm{air}}+\frac{45-20}{800}G_{s,i},\tag{2}$$

$$P_{s,i}^{\mathrm{dc}}=C_s\frac{G_{s,i}}{1000}\left[1-0.004\left(T_{s,i}^{\mathrm{cell}}-25\right)\right],\qquad
E_{s,d}^{\mathrm{raw}}=0.25\sum_{i\in d}\operatorname{clip}\left(0.96P_{s,i}^{\mathrm{dc}},0,C_s\right).\tag{3}$$

这里的 45 °C NOCT、−0.004 °C⁻¹ 温度系数及 0.96 逆变效率是统一实验假设。实际 Ross/PVWatts 应使用阵列平面辐照度或有效辐照度；本数据缺少可核验的组件倾角、方位和组件型号，故用 GHI 代理，固定效率也不是完整逆变器模型。站点缩放 $a_s$ 由**训练集**中 $E^{\mathrm{raw}}\geq0.1C_s$ 的日期求实测/原始模拟发电量比值的中位数：

$$a_s=\operatorname{median}_{d\in\mathcal T_s}\frac{E_{s,d}^{\mathrm{actual}}}{E_{s,d}^{\mathrm{raw}}},\qquad
H_{s,d}^{\mathrm{phys}}=\frac{\min(a_sE_{s,d}^{\mathrm{raw}},24C_s)}{C_s}.\tag{4}$$

$a_s$ 是经验校正，包含多种未建模效应，不应解释为纯系统损耗率。实现与误差见[物理基线报告](research_physics_baseline.md)。

### 3.2 条件物理校正

训练集的日 NWP GHI 上三分位为 3.54083333 kWh/m²，用它把站点日分为高/低辐照两组。在每个站点 $s$ 和组 $b$ 内，只用训练集的真实减物理预测残差求均值：

$$m_{s,b}=\frac{1}{|\mathcal T_{s,b}|}\sum_{d\in\mathcal T_{s,b}}(H_{s,d}^{\mathrm{actual}}-H_{s,d}^{\mathrm{phys}}),\qquad
H_{s,d}^{\mathrm{cond}}=\operatorname{clip}(H_{s,d}^{\mathrm{phys}}+m_{s,b},0,24).\tag{5}$$

训练样本构造大模型目标时，使用同组**留一均值**，避免把本样本标签直接放进锚点；验证和测试只使用训练集完整组均值。条件校正是经验统计层，不是新的物理定律。方案、组参数和验证结果见[步骤 8 记录](research_conditional_v2_results.md)。

### 3.3 本地 9B 残差分档和数值恢复

采用 Qwen3.5-9B 的本地 4-bit 量化权重 [13]，用 LoRA 思路 [7]在量化基座上训练最后 2 层 rank 8 adapter；此实现与量化低秩微调研究 [8]相关，但不声称复现其全部细节。可训练参数为 1.328 M，占模型总参数约 0.015%。输入由场站、目标日期、目标日 NWP 摘要、$d-9$ 至 $d-3$ 的历史发电量，以及条件物理等效小时数组成。

令剩余残差 $r_{s,d}=H_{s,d}^{\mathrm{actual}}-H_{s,d}^{\mathrm{cond}}$。仅对 890 条训练残差求三分位，得到边界 $q_1=-0.39436667$ h、$q_2=+0.32496667$ h，标记 A/B/C；各档训练中位残差分别为 $\widetilde r_A=-0.8728$ h、$\widetilde r_B=-0.0215$ h、$\widetilde r_C=+0.8141$ h。输出类别各占一个 token。推理读取首个 assistant token 的三个 logit，在允许的 A/B/C 上归一化：

$$p_k=\frac{\exp(z_k)}{\sum_{j\in\{A,B,C\}}\exp(z_j)},\qquad
\widehat r_{s,d}=\sum_{k\in\{A,B,C\}}p_k\widetilde r_k,\qquad
\widehat E_{s,d}=C_s\operatorname{clip}(H_{s,d}^{\mathrm{cond}}+\widehat r_{s,d},0,24).\tag{6}$$

这种离散表示借鉴数值 token 化的研究动机 [9]、[11]，但不同于训练专用时间序列基础模型。比较对象包括原校准物理、条件物理、未微调 9B 基座，以及最终 890 步 adapter；验证期还比较前轮连续小数残差 adapter。训练方案、内存错误处理和哈希见[步骤 9 方案](research_class_v3_protocol.md)。

### 3.4 指标和统计比较

记 $e_j=\widehat H_j-H_j$，报告 MAE、RMSE 和平均偏差：

$$\mathrm{MAE}=N^{-1}\sum_j|e_j|,\qquad
\mathrm{RMSE}=\sqrt{N^{-1}\sum_je_j^2},\qquad
\mathrm{Bias}=N^{-1}\sum_je_j.\tag{7}$$

MAE 和 RMSE 提供不同的误差侧面 [14]。同一天四站为一个日期块，对参照与候选的绝对误差差值按日期重采样 10,000 次，取 2.5% 与 97.5% 分位为描述性区间；方法基础可追溯到 bootstrap [15]。区间未处理相邻日时间相关、多轮验证选择或站点之外的分布变化，不能充当普适显著性证明。

## 4 实验结果

### 4.1 验证期

| 方法 | MAE（h） | RMSE（h） |
| --- | ---: | ---: |
| 原校准物理 | 0.7802 | 0.9864 |
| 条件物理 | 0.6834 | 0.8534 |
| 连续小数残差 9B adapter | 0.7083 | 0.8725 |
| 分档任务未微调 9B 基座 | 0.7035 | 0.8712 |
| 分档任务最终 9B adapter | **0.6757** | **0.8444** |
| 训练期滚动选择的 XGBoost（后续探索） | 0.8776 | 1.0548 |

最终 adapter 比条件物理 MAE 低 0.007695 h；46 个日期块 bootstrap 区间为 [−0.005798,+0.020153] h，跨零。分类语言损失从第 1 步 4.249 降至第 890 步 0.183，但此损失还包含容易预测的回复结构 token，不等于类别学习或发电量误差必然改善。验证期训练与结果见[步骤 9 结果](research_class_v3_results.md)。

为核查训练目标，事后固定一轮仅对 A/B/C 类别 token 计算损失的 v4 消融，其余配置、训练样本和 890 次更新不变。v4 的验证 MAE 为 **0.6774 h**，与 v3 的 0.6757 h 基本相当、略差；在全部 184 条验证样本上计算的受限三类交叉熵由 v3 的 1.0856 变为 1.0935，未提高类别预测。训练器最后一次抽样验证损失为 1.086，不能与完整样本的受限三类交叉熵直接比较。训练时发现并修正默认损失掩码会额外计入一个补齐位置的问题，失败试运行与修订后的训练均已存档。完整消融见[步骤 12 结果](research_class_only_v4_results.md)。

XGBoost [10] 对照是在最终测试评分**之后**另行设计的探索实验：仅用训练期四个滚动时间折选配置，春季验证 MAE 0.8776 h、偏差 −0.5904 h，未读取或评分测试期。它说明一个具体的天气、历史与原始物理特征回归模型也会遇到跨季节误差；不能据此认定所有传统机器学习模型都不如大模型。完整方案和失败日志见[步骤 11 报告](research_xgb_exploration_results.md)。

针对 S-2 的系统偏差，又在训练期四个滚动折上尝试站点和辐照度分组的正则化线性残差斜率。最好的四折平均 MAE 为 0.8523 h，原条件物理为 0.8589 h，未达到预先固定的 0.01 h 选用门槛，故保留原物理锚点。该探索没有读取旧测试期，结果见[步骤 13 报告](research_physics_slope_results.md)。

### 4.2 方案冻结后的测试期

在编写测试方案前，为查看文件结构意外打印了 2019-04-16 的一条真实标签；模型和参数此前已固定，但该标签不再严格未见。按**评分前写入方案**的规则，主测试剔除该日完整四站块，使用其余 58 天、232 条；全 59 天、236 条另报告，且结论方向一致。详细处理与哈希见[步骤 10 结果](research_final_test_results.md)。

| 方法 | MAE（h） | RMSE（h） | 平均偏差（h） |
| --- | ---: | ---: | ---: |
| 原校准物理 | 1.0253 | 1.3507 | +0.5715 |
| 条件物理 | 0.8622 | 1.1376 | +0.4520 |
| **未微调 9B 基座** | **0.7960** | **1.0441** | **+0.2180** |
| 最终 9B adapter | 0.8437 | 1.1110 | +0.4247 |

最终 adapter 对条件物理的 MAE 改善为 +0.018547 h，日期块区间 [+0.009800,+0.027509] h；对未微调基座的改善为 −0.047704 h，区间 [−0.076986,−0.017547] h。未微调基座是本段时间中最佳的既有方法。最终 adapter 在验证期位于基座前面、测试期落在基座后面，因此本文不能宣称此次低秩微调带来了稳定的额外收益。全 236 条补充评分的四组 MAE 为 1.0356、0.8737、0.8071、0.8558 h，顺序相同。

在上述测试评分之后才训练的 v4 adapter，在同一旧测试期 232 条上的 MAE 为 **0.8422 h**，相对 v3 改善仅 0.0015 h，仍弱于未微调基座 0.7960 h。因为此时旧测试已被用于错误分析，v4 数字只是回顾性消融，不是新一轮独立测试；不能将它与本节事前冻结的测试结果混同。

## 5 讨论与局限

测试期条件物理平均偏差 +0.4520 h，远高于验证期 +0.0340 h；对应的真实剩余残差均值从 −0.0340 h 移到 −0.4520 h。未微调基座在两期平均施加约 −0.2380/−0.2340 h 修正，而最终 adapter 为约 −0.0343/−0.0273 h。基座的更大负修正因而在测试期更能抵消整体正偏差。测试期 NWP 日 GHI 均值也高于验证期（5.8668 对 5.0097 kWh/m²），但这些同时发生的变化**不能证明**辐照度是因果原因。两种模型修正的标准差均小于 0.08 h，而测试剩余残差标准差约 1.04 h，逐日变化辨别仍不足。

本研究至少有五项限制。第一，目标日 NWP 逐条发布时间不可核验，无法完全证明每条样本符合严格日前可用性。第二，倾角、方位、遮挡、组件和逆变器参数缺失，使 GHI 代理组件面辐照度及统一参数带来系统误差。第三，只有河北四站的一段时间，缺乏上海地区外部验证。第四，同一验证期曾用于多轮分析和模型设计，验证区间易显乐观；测试期虽在模型冻结后评分，仍出现一条标签意外暴露，已通过主结果排除整日处理。第五，虽按 NWP 机器学习对照研究 [4] 的动机补充了训练期滚动选择的 XGBoost 探索实验，但它是在测试评分后设计，未在新的独立时间段与最终大模型公平比较；其他强回归基线也尚未覆盖。这些限制决定了本文更适合呈现可复现实验和负结果分析，而不是宣称普适的模型优势。

优化顺序是：优先获取覆盖四季、具备逐条预报发布时间和场站设备元数据的新数据；在新的独立评估期之前，先于训练期内部建立多个滚动时间验证窗口，并加入线性/树模型等强对照；然后检验更细的组件面辐照度与热模型，以及更能表达跨季节偏差的残差目标。本次类别目标修复及训练期辐照度斜率补丁均未带来可靠增益，因此不宜继续在旧测试期上增加迭代。当前测试标签已用于本文分析，不能在其上调参后再次把同一段时间称为独立测试。

## 6 结论

简化物理预测经训练期条件校正后，在主测试期的 MAE 从 1.0253 h 降到 0.8622 h。本地 9B 模型与物理锚点结合具有可行的运行路径；未微调基座在本测试期取得 0.7960 h MAE。残差分档微调模型为 0.8437 h，虽然比条件物理稍好，却没有超过基座。关键经验是：**验证集上的微小收益不能代替跨时期的独立比较**，应把物理假设、数据时效、强对照和模型微调收益分别报告。

## 参考文献

以下 15 条的作者、题名、年份及原始链接已在[文献核对清单](research_references.md)逐项登记；定稿时按课程要求调整格式。

[1] Yao, T., et al. (2021). A photovoltaic power output dataset: Multi-source photovoltaic power output dataset with Python toolkit. *Solar Energy*, 230, 122–130. [DOI](https://doi.org/10.1016/j.solener.2021.09.050).

[2] Zhang, Y., et al. (2026). Rethinking the use of deep learning methods for photovoltaic power forecasting. *Nature Communications*, 17, 7613. [DOI](https://doi.org/10.1038/s41467-026-73817-3).

[3] Antonanzas, J., et al. (2016). Review of photovoltaic power forecasting. *Solar Energy*, 136, 78–111. [DOI](https://doi.org/10.1016/j.solener.2016.06.069).

[4] Markovics, D., & Mayer, M. J. (2022). Comparison of machine learning methods for photovoltaic power forecasting based on numerical weather prediction. *Renewable and Sustainable Energy Reviews*, 161, 112364. [DOI](https://doi.org/10.1016/j.rser.2022.112364).

[5] Dobos, A. P. (2014). *PVWatts Version 5 Manual*. NREL/TP-6A20-62641. [DOI](https://doi.org/10.2172/1158421).

[6] Holmgren, W. F., Hansen, C. W., & Mikofski, M. A. (2018). pvlib python: a python package for modeling solar energy systems. *Journal of Open Source Software*, 3(29), 884. [DOI](https://doi.org/10.21105/joss.00884).

[7] Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. [预印本](https://arxiv.org/abs/2106.09685).

[8] Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS 2023*. [论文页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html).

[9] Gruver, N., et al. (2023). Large Language Models Are Zero-Shot Time Series Forecasters. [arXiv:2310.07820](https://arxiv.org/abs/2310.07820).

[10] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. [DOI](https://doi.org/10.1145/2939672.2939785).

[11] Ansari, A. F., et al. (2024). Chronos: Learning the Language of Time Series. [arXiv:2403.07815](https://arxiv.org/abs/2403.07815).

[12] Tan, M., et al. (2024). Are Language Models Actually Useful for Time Series Forecasting? [arXiv:2406.16964](https://arxiv.org/abs/2406.16964).

[13] Qwen Team. (2026). Qwen3.5: Towards Native Multimodal Agents. [官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B).

[14] Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688. [DOI](https://doi.org/10.1016/j.ijforecast.2006.03.001).

[15] Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. *The Annals of Statistics*, 7(1), 1–26. [DOI](https://doi.org/10.1214/aos/1176344552).
