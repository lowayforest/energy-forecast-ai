# 步骤 8：条件物理校正与 9B 剩余残差微调方案（运行前固定）

## 研究问题与边界

[步骤 7](research_error_analysis_v1.md)发现高预报辐照度下各站物理误差方向不同，旧 adapter 的逐日修正与真实残差相关性低。本轮预先固定一个**训练集拟合的经验偏差校正**，检验它能否改进物理锚点，以及同一 9B 模型能否进一步预测剩余残差。该校正利用物理模型与预报辐照度，但不是新物理定律。所有比较仍限于 2019-03-01 至 2019-04-15 的 184 条验证样本；**不读取或评分测试集**。

输入沿用[步骤 2—3](research_physics_baseline.md)的四站每日样本和 Ross + PVWatts 校准物理预测。源文件 SHA-256 固定在 [`scripts/build_research_conditional.py`](../scripts/build_research_conditional.py)；原始数据来源与天气预报发布时间的证据及限制见[步骤 1—2](research_daily_dataset.md)。目标日 `power` 仅作为训练标签、验证评分；目标日现场 `lmd_*` 不参与模型输入。

## 预先固定的校正公式

以训练集 890 条的目标日 **NWP 日 GHI 积分** $G_{s,d}$ 的上三分位设阈值 $q_{2/3}^{\mathrm{train}}$。分组 $b(s,d)=\mathrm{high}$ 当 $G_{s,d}>q_{2/3}^{\mathrm{train}}$，否则为 `low`；阈值对所有站点相同，等于号归 `low`。单位为 kWh/m²。令已校准物理预测的等效满负荷小时为 $H^{\mathrm{phys}}_{s,d}$，训练残差 $r_{s,d}=H^{\mathrm{actual}}_{s,d}-H^{\mathrm{phys}}_{s,d}$。每个站点和 GHI 组仅用训练日计算均值

$$m_{s,b}=\frac{1}{n_{s,b}}\sum_{j\in\mathcal T_{s,b}}r_{s,j}.$$

验证日的条件物理预测为

$$H^{\mathrm{cond}}_{s,d}=\operatorname{clip}(H^{\mathrm{phys}}_{s,d}+m_{s,b(s,d)},0,24).$$

为避免某个训练样本的自身标签直接进入**新增的条件校正值**，训练日使用该组的**留一均值** $m^{(-d)}_{s,b}=(\sum_{j\in\mathcal T_{s,b}}r_{s,j}-r_{s,d})/(n_{s,b}-1)$ 生成训练锚点；验证日使用完整训练组均值 $m_{s,b}$。每组至少需两个训练日。继承的原物理模型已用全部训练样本拟合站点系数，因此训练期物理锚点并非完全折外预测；这是本实验的限制，不影响验证期只用训练数据拟合的边界。输出文件 `conditional_{train,valid}.jsonl` 只含预测与分组，不含目标日真实值；对应 manifest 记录组样本数、均值、哈希和训练/验证评分。训练分数是内部诊断，**不作为独立泛化结论**。

## 大模型训练与对照

[`scripts/build_research_sft_v2.py`](../scripts/build_research_sft_v2.py)继续使用既有提示词 `prompts/research_pv_residual_sft_v1.md`（SHA-256 `8f5004dc8f08832c6eff3b956884f91a7b9953a0597d6841eb4b4e6c58e313aa`），只把其中的 `physics_full_load_hours` 换为条件物理锚点。目标残差变为 $r^{(2)}_{s,d}=H^{\mathrm{actual}}_{s,d}-H^{\mathrm{cond}}_{s,d}$；推理时只给 `system` 和 `user`，不把 SFT 的 assistant 标签传入模型。预测能量仍为 $\widehat E_{s,d}=C_s\operatorname{clip}(H^{\mathrm{cond}}_{s,d}+\widehat r^{(2)}_{s,d},0,24)$ MWh。

基座固定为 `mlx-community/Qwen3.5-9B-4bit` revision `8b2b98c00a6b4d291155e4890773ca8f769aee53`；权重 SHA-256 与[步骤 5](research_qlora_feasibility.md)一致。微调参数与[步骤 6](research_mlx_finetune_v1.md)保持一致：最后 2 层、rank 8、batch 1、学习率 $10^{-5}$、**256 次更新**、512 tokens、assistant-only loss、梯度检查点、种子 42、每 64 步保存。固定使用第 256 步 adapter，不按验证结果选择检查点。256 次更新仍不足完整一轮训练集；本轮只改变物理锚点与相应目标残差。

验证推理使用相同模型聊天模板、`enable_thinking=False`、贪心生成、最多 64 tokens；只解析单字段数值 JSON。无效输出回退到条件物理预测。比较以下五组，单位均为等效满负荷小时：

1. 原校准物理预测；
2. 新条件物理预测；
3. 步骤 6 的旧 adapter 预测；
4. 新条件锚点上的未微调 MLX 基座预测；
5. 新条件锚点上的 256 步 adapter 预测。

报告整体和逐站 MAE、RMSE、平均偏差及回退数。预先计算五组成对日期块 MAE 改善：条件物理对原物理、新 adapter 对条件物理、新 adapter 对新基座、新 adapter 对旧 adapter、新 adapter 对原物理；同日四站为一块，46 个日期块 bootstrap 10,000 次，种子 42。区间跨零时不声称稳定改善。验证集供模型选择，测试集留待方案冻结后的最终评估。

## 由用户运行

先切到研究分支并确认本机 `.venv`、下载的 9B MLX 权重、步骤 2—6 的 Git 忽略数据仍在。然后在项目根目录运行：

```bash
bash scripts/run_research_v2.sh
```

[`scripts/run_research_v2.sh`](../scripts/run_research_v2.sh)依次构建条件物理预测与 SFT、训练 adapter、推理新基座和 adapter、生成比较 JSON；标准输出同时保存到 `data/processed/research/logs/`。训练文件位于 Git 忽略目录 `data/processed/research/adapters/qwen35_9b_residual_v2/`，脚本不会覆盖已存在的 adapter；训练正常退出后写入包含数据和 adapter 哈希的 `training_provenance.json`，重跑时凭此接续验证推理，只有部分训练目录时会停止以待检查。步骤 6 的 256 步训练约用 18 分钟，本轮总耗时还包括两次完整验证推理，实际时间以本机日志为准。完整训练的峰值内存不从包裹 shell 的 `/usr/bin/time -l` 推断。

运行成功的标志是末行 `V2 validation pipeline finished. The sealed test split was not read.`，并生成 `data/processed/research/mlx_v2_comparison_valid.json`。运行后由我核对日志、哈希、各站指标与日期块区间，再补齐本步骤的结果和论文记录；本文件在运行前固定方案，暂不填预测结果。
