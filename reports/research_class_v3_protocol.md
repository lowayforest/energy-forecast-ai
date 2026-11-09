# 步骤 9：9B 残差分档预测方案（运行前固定）

## 目标与动机

目标是让本地 9B 模型在相同 184 条验证样本上，**优于步骤 8 的条件物理 MAE 0.6834 等效满负荷小时**。前两轮自由生成小数残差时，语言损失下降，但模型输出几乎总是小幅负修正，未能超过物理锚点。[数字文本分词与时间序列预测研究](https://arxiv.org/abs/2310.07820)指出数值编码方式会影响模型表现；[LLM 时间序列消融研究](https://arxiv.org/abs/2406.16964)也提醒不能把预训练大模型的存在直接等同于预测增益。本轮尝试把连续残差改为单 token 分档，再由档位概率得到数值修正。引用这些文献是方法动机，不预设本数据集一定改善。

延用[步骤 8](research_conditional_v2_protocol.md)已经固定的条件物理锚点、四站训练/验证划分和本地 `mlx-community/Qwen3.5-9B-4bit` 量化权重。原始数据、天气预报来源及逐条发布时间不可核验的限制见[步骤 1—2](research_daily_dataset.md)。训练和验证使用各自已有的 v2 残差样本；文件 SHA-256 固定在 [`scripts/build_research_sft_v3.py`](../scripts/build_research_sft_v3.py)。**不读取测试集**。

## 训练目标与数值还原

仍令 $r_{s,d}=H_{s,d}^{\mathrm{actual}}-H_{s,d}^{\mathrm{cond}}$。只用 **890 条训练残差**的三分位设置 A/B/C 边界：$q_{1/3}^{\mathrm{train}}=-0.39436667$ h、$q_{2/3}^{\mathrm{train}}=+0.32496667$ h，等于号归较低档。训练集三档样本数为 297/296/297；每档的训练残差**中位数**作为还原值，分别约为 **−0.8728、−0.0215、+0.8141 h**，精确值由脚本写入 manifest。验证标签只用于训练期间的语言损失计算和事后评分，不能参与边界或代表值的拟合。

新提示词固定为 [`prompts/research_pv_residual_class_v3.md`](../prompts/research_pv_residual_class_v3.md)，SHA-256 `c0b9cc042eb7ef691b4146100462a0b871a0bcf8049a924ddae5fb0c76827b16`。模型只需生成 `A`、`B` 或 `C`；本机分词器检查这三个字母各是单个 token。推理时不贪心选一档，而从 assistant 首 token 的三档 logits 计算

$$p_k=\frac{e^{z_k}}{\sum_{j\in\{A,B,C\}}e^{z_j}},\qquad
\widehat r_{s,d}=\sum_{k\in\{A,B,C\}}p_k\,\widetilde r_k,$$

$$\widehat H_{s,d}=\operatorname{clip}(H^{\mathrm{cond}}_{s,d}+\widehat r_{s,d},0,24),\qquad
\widehat E_{s,d}=C_s\widehat H_{s,d}.$$

$z_k$ 是类别 token 的下一 token logit，$\widetilde r_k$ 是训练集该档中位数；$C_s$ 单位 MW，$H$ 单位小时，$E$ 单位 MWh。$p_k$ 只在三个允许类别中归一化，**不是整个词表上的已校准概率**；另记录三个类别 token 的总词表概率质量作为诊断。若 logit 非有限，回退到零修正的条件物理预测并计数。推理提示只有 system/user 消息，**不传入 assistant 类别标签**。

## 预先固定的训练与比较

采用与前轮相同的模型权重 revision `8b2b98c00a6b4d291155e4890773ca8f769aee53`、最后 2 层 rank 8 QLoRA、batch 1、学习率 $10^{-5}$、最多 512 tokens、assistant-only loss、梯度检查点和种子 42。本轮固定 **890 次更新，即 batch 1 对 890 条训练样本的一轮遍历**；依据本机 `mlx-lm 0.32.2` 的批次迭代逻辑，每轮随机排列各训练样本一次。每 128 次更新计算 16 个验证批次的分类语言损失，每 256 次保存中间 adapter；主结果只用最终第 890 次更新的 adapter，不事后挑选最好检查点。此轮同时改变输出表示与训练步数，若有提升也不能单独归因于分档。

固定比较：条件物理、步骤 8 的数值残差 256 步 adapter、新分档任务的未微调 9B 基座、新分档任务的 890 步 adapter，以及原物理基线。整体和逐站报告 MAE、RMSE、平均偏差、回退数量；用 46 个日期块、10,000 次 bootstrap（种子 42）计算新 adapter 相对条件物理、新基座、旧 adapter 和原物理的 MAE 改善区间。**主要判据**是新 adapter 相对条件物理的 MAE 是否降低，并同时看日期块区间是否跨零。若只得到较低点估计但区间跨零，仅称探索性改善；若未降低，按实报告，不调参追逐当前验证集。

## 用户运行与复核

在研究分支的项目根目录执行：

```bash
bash scripts/run_research_v3.sh
```

[`scripts/run_research_v3.sh`](../scripts/run_research_v3.sh)先生成 A/B/C 样本，再运行 890 步训练、基座与 adapter 的验证预测、最后生成对照 JSON。日志和权重保存在 Git 忽略目录 `data/processed/research/`；训练成功会写入输入与 adapter 哈希的 provenance，重跑可接续已经完成的验证预测，部分训练目录会触发停止检查。依据前轮 256 步约 15 分钟的记录，本轮训练可能接近一小时，实际以本机日志为准。不要用包裹 shell 的 `/usr/bin/time -l` 输出当作 MLX 进程的可靠峰值内存。

本方案提交前，只对脚本做语法检查、分词器聊天模板检查，以及**合成输入**上的基座前向传播：A/B/C token ID 分别为 32/33/34，assistant 生成前缀与训练模板一致，logits 形状和三类概率计算正常。训练样本按新模板编码后最长 459 tokens，低于 512 上限。**尚未运行本轮训练或验证评分**。用户运行结束后再由我核对哈希、指标与限制，补写结果报告；测试集在最终方案冻结前继续封存。
