# 步骤 6：同一 MLX 基座的微调前后验证对照

## 目标与对照边界

基于[步骤 5 的物理残差样本](research_qlora_feasibility.md)，比较相同 `mlx-community/Qwen3.5-9B-4bit` 权重在未微调与加载本项目 adapter 后的光伏日发电量预测。两组使用同一数据、系统提示、聊天模板、数值解析、回退策略和验证日期；唯一预设差异是是否加载训练所得 adapter。目标日的实测值只用于推理后评分。测试集仍未打开评分。

模型固定为 Hugging Face revision `8b2b98c00a6b4d291155e4890773ca8f769aee53`，两块 safetensors 的 SHA-256 见[步骤 5](research_qlora_feasibility.md#模型下载与样本长度)。`scripts/evaluate_research_mlx.py` 每次运行检查两块权重、验证 SFT 文件、每日标签文件和物理预测文件的哈希。推理只取 SFT `system`、`user` 消息，**不会向模型传入 `assistant` 残差标签**。使用该模型聊天模板的 `enable_thinking=False`、贪心生成 `temperature=0`、最多 64 tokens。

## 固定解析与指标

只接受 JSON 对象 `{"correction_full_load_hours": 数字}`，数值必须有限且绝对值不超过 24 小时；否则残差回退为 0，即沿用已校准物理预测。模型输出 $\widehat r_{s,d}$ 后，统一计算

$$\widehat H_{s,d}=\operatorname{clip}(H_{s,d}^{\mathrm{physics}}+\widehat r_{s,d},0,24),\qquad \widehat E_{s,d}=C_s\widehat H_{s,d}.$$

其中 $H$、$r$ 单位为小时，$C_s$ 为 MW，$E$ 为 MWh。按等效满负荷小时评估 $\mathrm{MAE}=N^{-1}\sum_j|\widehat H_j-H_j|$、$\mathrm{RMSE}=\sqrt{N^{-1}\sum_j(\widehat H_j-H_j)^2}$ 和平均偏差。微调与基座的逐日 MAE 差再按 46 个日期块进行 10,000 次 bootstrap 重采样（种子 42）；同日四站作为一块，避免把相关站点误当独立样本。比较脚本为 [`scripts/compare_research_mlx.py`](../scripts/compare_research_mlx.py)。

## 未微调基座：已完成

先运行前 4 条作为接口检查，再在相同设置下接续完成全部 **184 条**验证样本：

```bash
.venv/bin/python scripts/evaluate_research_mlx.py --variant base --limit 4
.venv/bin/python scripts/evaluate_research_mlx.py --variant base
```

所有 184 条返回可解析的残差 JSON，回退 **0 条**；平均生成耗时 1.366 秒/条。基座验证 MAE **0.8398 h**、RMSE **1.0399 h**、平均偏差 **−0.1400 h**。同批已校准物理基线 MAE 为 0.7802 h，因此本次基座残差提示**尚未优于物理模型**。逐站 MAE：S-1 0.9371、S-2 1.0545、S-3 0.6759、S-4 0.6916 h。无标签预测保存在 Git 忽略目录 `data/processed/research/mlx_base_valid.jsonl`，SHA-256 `2c9bfdca815fa6a5003e973287d76fa13efd780ced2545fa903ea1d582cad10f`，配置文件 SHA-256 `e77a6c3ce589b62495d41109cb9954524b9b27c6a1b7bb8d3a607bb7f0dcf2ba`。

步骤 4 的 Ollama 结果使用另一种量化格式和“直接输出总小时数”的提示形式，可作背景参考，**不能作为微调效果的因果对照**。

## 微调配置与运行

固定训练入口为 [`scripts/train_research_qlora.sh`](../scripts/train_research_qlora.sh)。它先校验模型权重与 890/184 条训练、验证 JSONL 的 SHA-256，再使用 [MLX-LM 的 LoRA/QLoRA 流程](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)。预设：量化 9B 基座、最后 2 层、秩 8、batch size 1、学习率 $10^{-5}$、最多 256 次更新、512 token 上限、assistant-only 损失、梯度检查点、种子 42；每 64 次更新保存 adapter，并用 8 个验证批次计算语言模型损失。**256 次单样本更新少于 890 个训练样本，不应写成完成一个训练轮次。**主结果使用第 256 次更新的 adapter，不事后挑选更有利的检查点。

```bash
/usr/bin/time -l bash scripts/train_research_qlora.sh > /tmp/qlora-train-256.log 2>&1
```

实际完成 **256/256 次更新**，可训练参数 **1.328 百万 / 8,953.802 百万（0.015%）**，墙钟 **1,058.29 秒（约 17 分 38 秒）**，未出现内存不足。每次验证只抽取 8 个批次，语言模型损失从初始 1.910 依次变为第 64/128/192/256 步的 0.554/0.549/0.564/0.539；该损失衡量残差文本的拟合，**不能替代发电量预测误差**。第 256 步训练日志显示的损失为 0.553。`/usr/bin/time -l` 包裹的是 shell，所报进程内存峰值不是 MLX 训练进程峰值；本次未可靠测得完整训练的峰值内存。

最终 adapter 位于 Git 忽略目录 `data/processed/research/adapters/qwen35_9b_residual_v1/`：`adapters.safetensors` SHA-256 `ddea9f7387a41816613c76774da980e906332de787ec373d19ef3b2d78991a95`，`adapter_config.json` SHA-256 `39ad21694337d07e5c063351887e938d66d0e9ed450e20e85a36fe2cf402bd29`。训练日志另存为 `data/processed/research/logs/qlora_train_256.log`，SHA-256 `cbfa3956349c351db41be084dded62b776a7957c72820d0d0c5428f237a259e6`。这些哈希记录本次结果，但大文件未入库，复现需先按[步骤 5](research_qlora_feasibility.md)准备数据和模型再重新训练。短试运行 adapter 不参与正式比较。

## 结果与解释

正式评估使用第 256 步 adapter，在与基座完全相同的 **184 个站点日、46 个日期**上运行；两组均 **0 条回退**。运行命令：

```bash
.venv/bin/python scripts/evaluate_research_mlx.py --variant adapter
.venv/bin/python scripts/compare_research_mlx.py
```

| 方法 | MAE（h） | RMSE（h） | 平均偏差（h） |
| --- | ---: | ---: | ---: |
| 训练集校准的物理预测 | 0.7802 | 0.9864 | +0.1561 |
| 同一 MLX 未微调基座 | 0.8398 | 1.0399 | −0.1400 |
| MLX 基座 + 本次 adapter | **0.7907** | **0.9795** | −0.0006 |

| 站点 | 物理 MAE（h） | 未微调 MAE（h） | 微调后 MAE（h） |
| --- | ---: | ---: | ---: |
| S-1 | 0.7051 | 0.9371 | 0.7962 |
| S-2 | 1.0689 | 1.0545 | 1.0186 |
| S-3 | 0.6430 | 0.6759 | 0.6756 |
| S-4 | 0.7039 | 0.6916 | 0.6722 |

定义“MAE 改善”为**参照方法的绝对误差减去 adapter 的绝对误差**。相对同一 MLX 基座，平均改善 **+0.049114 h**，按日期块 bootstrap 的 95% 区间为 **[−0.015092, +0.124738] h**；相对物理模型，平均改善 **−0.010451 h**，区间为 **[−0.054620, +0.031039] h**。两者均跨零。逐日四站平均绝对误差相对基座改善的日期有 25/46 天，相对物理模型为 24/46 天。

因此，本次 **256 步微调在验证集上降低了相对同一未微调模型的点估计 MAE，但尚无稳健的改善证据**；相对物理模型的 MAE 略差，RMSE 略好。S-1、S-3 的预测效果仍不如物理基线；S-2、S-4 略有改善，说明不能以整体均值代替逐站结论。也不能仅凭语言模型损失下降宣称预测准确率提高。

adapter 预测文件 `data/processed/research/mlx_adapter_valid.jsonl` 不含实测标签，SHA-256 `08d1446a148c2a6b1deecce17dacb61808cba07ea992fcb9a7b58eb67c44599e`；配置文件 SHA-256 `e516d52c1511ae36ab2ecf42b77c053488d6113064bb0fcc0a23d56d072891d6`。对照结果 `data/processed/research/mlx_comparison_valid.json` 的 SHA-256 为 `488ad610ec454b2eeb5a3135074097e3b9ed383ed26aedbc67abee8de34ea2ae`。这些结果均保存在 Git 忽略目录，提交的是计算脚本与本报告。验证集既用于训练期间观察语言损失，又用于本次比较，因此不能当成最终独立测试；测试集继续封存。数据来自河北四站，也不能直接推出上海光伏场站的效果。训练未满一个轮次，后续是否增加步数、调整目标或强化物理约束，须在测试前记录方案并只用训练/验证集选择。
