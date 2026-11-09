# 课程论文参考文献候选清单（已核对原始出处）

以下 15 条覆盖数据、光伏预测、物理模型、大模型方法及评估。题名、作者、年份和 DOI/原始链接已按出版社、作者仓库、模型发布方或论文原页核对；正式论文中只保留正文实际引用的条目。期刊论文与预印本分开标识，不能把 arXiv 预印本写成已发表期刊论文。

| 编号 | 文献与原始链接 | 在本文中的用途 |
| --- | --- | --- |
| [1] | Yao, T., Wang, J., Wu, H., Zhang, P., Li, S., Wang, Y., Chi, X., & Shi, M. (2021). “A photovoltaic power output dataset: Multi-source photovoltaic power output dataset with Python toolkit.” *Solar Energy*, 230, 122–130. [DOI: 10.1016/j.solener.2021.09.050](https://doi.org/10.1016/j.solener.2021.09.050); [作者数据仓库](https://github.com/yaotc/PVODataset). | PVOD 原始数据、功率单位和 NWP 预报说明。 |
| [2] | Zhang, Y., Zhang, Y., Dai, Z., & Zhang, R. (2026). “Rethinking the use of deep learning methods for photovoltaic power forecasting.” *Nature Communications*, 17, 7613. [DOI: 10.1038/s41467-026-73817-3](https://doi.org/10.1038/s41467-026-73817-3). | AI-PVOD 整理数据来源；历史功率与前瞻天气联合输入的相关工作。本文不把其模型结果直接与本研究日能量误差比较。 |
| [3] | Antonanzas, J., Osorio, N., Escobar, R., Urraca, R., Martínez-de-Pisón, F. J., & Antonanzas-Torres, F. (2016). “Review of photovoltaic power forecasting.” *Solar Energy*, 136, 78–111. [DOI: 10.1016/j.solener.2016.06.069](https://doi.org/10.1016/j.solener.2016.06.069). | 交代光伏预测的时间尺度、天气输入和物理/统计方法背景。 |
| [4] | Markovics, D., & Mayer, M. J. (2022). “Comparison of machine learning methods for photovoltaic power forecasting based on numerical weather prediction.” *Renewable and Sustainable Energy Reviews*, 161, 112364. [DOI: 10.1016/j.rser.2022.112364](https://doi.org/10.1016/j.rser.2022.112364). | 说明日前 NWP 驱动预测中常规机器学习对照的重要性；其数据和任务不同，不引用其数值作直接优劣结论。 |
| [5] | Dobos, A. P. (2014). *PVWatts Version 5 Manual*. NREL/TP-6A20-62641. [DOI: 10.2172/1158421](https://doi.org/10.2172/1158421). | PVWatts 直流功率和系统效率近似的技术来源；本文的 GHI 代替阵列平面辐照度及固定逆变效率是简化假设。 |
| [6] | Holmgren, W. F., Hansen, C. W., & Mikofski, M. A. (2018). “pvlib python: a python package for modeling solar energy systems.” *Journal of Open Source Software*, 3(29), 884. [DOI: 10.21105/joss.00884](https://doi.org/10.21105/joss.00884). | 本文物理基线的 `pvlib` 实现来源，包括所调用的 Ross 温度和 PVWatts 函数；[Ross 函数文档](https://pvlib-python.readthedocs.io/en/v0.11.1/reference/generated/pvlib.temperature.ross.html)另列明原始 Ross (1981) 会议出处。 |
| [7] | Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). “LoRA: Low-Rank Adaptation of Large Language Models.” *ICLR 2022*；[2021 年 arXiv 预印本](https://arxiv.org/abs/2106.09685). | 低秩 adapter 的方法依据，解释仅训练约 1.328 M 参数。 |
| [8] | Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). “QLoRA: Efficient Finetuning of Quantized LLMs.” *NeurIPS 2023*. [会议论文页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html). | 量化基座上训练低秩 adapter 的方法动机。本文使用 MLX-LM 的量化 LoRA 实现，不声称复现 QLoRA 论文的全部量化/优化器细节。 |
| [9] | Gruver, N., Finzi, M., Qiu, S., & Wilson, A. G. (2023). “Large Language Models Are Zero-Shot Time Series Forecasters.” [arXiv:2310.07820](https://arxiv.org/abs/2310.07820). | 数值文本分词和数值输出校准问题，为残差分档设计提供动机；不意味着该论文验证了本设计。 |
| [10] | Jin, M., Wang, S., Ma, L., et al. (2023). “Time-LLM: Time Series Forecasting by Reprogramming Large Language Models.” [arXiv:2310.01728](https://arxiv.org/abs/2310.01728). | 时间序列与语言模型对齐的相关方向；本文并未使用 Time-LLM 架构。 |
| [11] | Ansari, A. F., Stella, L., Turkmen, C., et al. (2024). “Chronos: Learning the Language of Time Series.” [arXiv:2403.07815](https://arxiv.org/abs/2403.07815). | 时间序列离散化/量化为 token 的相关工作；本文 A/B/C 仅是残差分档，不等同于 Chronos。 |
| [12] | Tan, M., Merrill, M. A., Gupta, V., Althoff, T., & Hartvigsen, T. (2024). “Are Language Models Actually Useful for Time Series Forecasting?” [arXiv:2406.16964](https://arxiv.org/abs/2406.16964). | 支持同时报告未微调基座与物理对照，避免把语言模型参与误写为已证明的预测贡献。 |
| [13] | Qwen Team. (2026). “Qwen3.5: Towards Native Multimodal Agents.” [官方 Qwen3.5-9B 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B). | 标注本文所用 9B 模型系列；本地量化版的具体权重 revision 与哈希另见实验日志。模型卡不是同行评审论文。 |
| [14] | Hyndman, R. J., & Koehler, A. B. (2006). “Another look at measures of forecast accuracy.” *International Journal of Forecasting*, 22(4), 679–688. [DOI: 10.1016/j.ijforecast.2006.03.001](https://doi.org/10.1016/j.ijforecast.2006.03.001). | 解释 MAE/RMSE 等误差指标的选择与报告。 |
| [15] | Efron, B. (1979). “Bootstrap Methods: Another Look at the Jackknife.” *The Annals of Statistics*, 7(1), 1–26. [DOI: 10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552). | bootstrap 方法基础出处。本文按同日四站成块重采样是具体实现，不能声称 Efron 原文直接给出了本数据的日期块区间。 |

## 正文放置建议

- 数据与场景：[1]、[2]；日前预测背景：[3]、[4]。
- 简化物理基线和实现：[5]、[6]；明确 GHI 代理阵列平面辐照度的误差来源。
- 模型与数值表示：[7]—[13]；把支持方法选择的引用与本实验结果分开写。
- 误差统计：[14]、[15]；区间跨零和重复使用验证集的局限须在结果讨论中说明。
