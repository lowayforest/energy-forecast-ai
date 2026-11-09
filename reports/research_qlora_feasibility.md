# 步骤 5：物理残差微调样本与本机 QLoRA 可行性

## 任务与依据

在[步骤 4 的零样本对照](research_zero_shot_ollama.md)之后，本步骤先准备针对光伏任务的监督微调样本，并检查 16 GB Apple M5 电脑能否对 9B 模型完成至少一次真实的反向传播。正式训练轮数与模型精度比较须在可行性验证后确定，不能把短试运行的损失变化当成预测效果。

[MLX-LM 官方微调文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)说明：对量化模型用 LoRA 训练即其实现中的 QLoRA；本地 JSONL 可采用 `messages` 聊天格式，`--mask-prompt` 使损失只计算最后的 assistant 回复。[LoRA 原论文](https://arxiv.org/abs/2106.09685)提出冻结预训练权重并学习低秩增量，概念上可写作 $W'=W+(\alpha/r)BA$，其中 $A,B$ 为可训练低秩矩阵，$r$ 为秩，$\alpha$ 为缩放系数。[QLoRA 论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract.html)进一步研究量化基座上的参数高效训练。本实验选用 [mlx-community/Qwen3.5-9B-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit)，固定仓库 revision `8b2b98c00a6b4d291155e4890773ca8f769aee53`；其权重是 MLX affine 4-bit 量化，不能写成采用了 QLoRA 原论文所有 NF4 等具体实现。它与步骤 4 的 Ollama `qwen3.5:9b` 具有相同上游模型系列，但量化格式和推理实现不同；**以后若要归因于微调，必须在同一个 MLX 权重上分别评估未训练与已训练 adapter**，不能只比较 Ollama 零样本和 MLX 微调结果。

上游项目的 [M5 Max 训练内存问题](https://github.com/ml-explore/mlx-lm/issues/1206)和 [Qwen3.5 层数与速度观察](https://github.com/ml-explore/mlx-lm/issues/1956)提示需要实机试验；这些是他人机器的报告，不能代替本机结论。

## 目标、输入与时间边界

对站点 $s$、目标日 $d$，以训练集校准的物理预测为锚点，定义小时数残差：

$$r_{s,d}=H_{s,d}^{\mathrm{actual}}-H_{s,d}^{\mathrm{physics}},\qquad H_{s,d}^{\mathrm{physics}}=\frac{E_{s,d}^{\mathrm{cal}}}{C_s}.$$

训练后的预测组合规则预先定义为：

$$\widehat H_{s,d}=\operatorname{clip}\!\left(H_{s,d}^{\mathrm{physics}}+\widehat r_{s,d},0,24\right),\qquad \widehat E_{s,d}=C_s\widehat H_{s,d}.$$

$H$ 和 $r$ 的单位为小时，$E$ 为 MWh，$C_s$ 为 MW。目标标签保留 4 位小数。输入与步骤 4 相同：目标日 NWP 摘要、额定容量、目标日期、历史 7 日发电水平、物理预测；历史最晚只到目标日前 3 天。assistant 消息包含残差标签，system/user 消息绝不包含目标日实际发电或现场实测天气。验证标签仅用于验证损失或事后评分，不参与梯度更新；**没有生成 `test.jsonl`**。

固定训练提示词在 [`prompts/research_pv_residual_sft_v1.md`](../prompts/research_pv_residual_sft_v1.md)，SHA-256：`8f5004dc8f08832c6eff3b956884f91a7b9953a0597d6841eb4b4e6c58e313aa`。构建脚本为 [`scripts/build_research_sft.py`](../scripts/build_research_sft.py)：

```bash
.venv/bin/python scripts/build_research_sft.py
```

输出位于 Git 忽略目录 `data/processed/research/sft_residual_v1/`：

| 文件 | 日期范围 | 条数 | SHA-256 |
| --- | --- | ---: | --- |
| `train.jsonl` | 2018-07-10 至 2019-02-28 | 890 | `262dd45abab6cac7f424bbe6b2785491c3b3a03911fbdf2a5039f002e6e5f1c9` |
| `valid.jsonl` | 2019-03-01 至 2019-04-15 | 184 | `aa326ce05e5d8631e80a5ef39d3bfca7b9dea471f53a025c1a67d8c0d0ac3da5` |

训练残差均值 0.0509 小时、标准差 1.0942 小时、范围 [−5.0962, 9.6248] 小时。极端值保留，以反映预报误差和可能的数据异常；正式训练需观察其影响。验证残差均值 −0.1561 小时、标准差 0.9740 小时。脚本把源日样本与物理预测按站点和日期逐一对齐，并校验源文件哈希。逐条检查了 JSONL 角色顺序、用户消息无标签、站点日期边界和无测试文件，全部通过。

## 模型下载与样本长度

本机 Python 3.12.14、macOS 26.5.1、Apple M5/16 GB；使用 `mlx-lm[train] 0.32.2`、`mlx 0.32.4`、`huggingface-hub 2.4.0`。标准下载命令为：

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

本次源站 Xet 大分片中途停滞；保留已下载且通过字节范围对照的 2.3 GB 前缀，随后用 `aria2c --continue=true --max-connection-per-server=4 --split=4` 对同一 revision 的大分片直连续传，并以源站 `x-linked-etag` 的 SHA-256 校验整文件。最终两块权重均齐全：

| 文件 | SHA-256 |
| --- | --- |
| `model-00001-of-00002.safetensors` | `a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd` |
| `model-00002-of-00002.safetensors` | `b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d` |

权重在 Git 忽略目录 `data/cache/research/mlx_qwen35_9b_4bit/`，约 5.6 GiB，不提交到仓库。使用该模型 tokenizer 的聊天模板统计，训练/验证样本 token 长度均为 **429–438 / 430–438**，中位数都是 435；没有样本超过本次设定的 512 token 上限。下载中途的重试是网络传输问题，与模型训练可行性无关。

## 本机两步训练试运行

为只测内存与反向传播，用训练 JSONL 前 16 条和验证 JSONL 前 8 条组成 `data/cache/research/sft_smoke_v1/`。其 SHA-256 分别为 `1d29ccaac0b23e3933af1c6100cdf8a596f4f307e41014c2bb732c7b26725508` 和 `de62a482e2024018b546053c9e719e02b5ac9b7e560bdf732f2ed8fc47a7cc21`。这些样本只用于**可行性试验**，不用于报告预测精度。复现命令：

```bash
mkdir -p data/cache/research/sft_smoke_v1
head -n 16 data/processed/research/sft_residual_v1/train.jsonl > data/cache/research/sft_smoke_v1/train.jsonl
head -n 8 data/processed/research/sft_residual_v1/valid.jsonl > data/cache/research/sft_smoke_v1/valid.jsonl
.venv/bin/mlx_lm.lora \
  --model data/cache/research/mlx_qwen35_9b_4bit \
  --train --data data/cache/research/sft_smoke_v1 \
  --fine-tune-type lora --mask-prompt --num-layers 2 \
  --batch-size 1 --iters 2 --val-batches 1 \
  --learning-rate 1e-5 --steps-per-report 1 --steps-per-eval 2 \
  --adapter-path data/processed/research/adapters/smoke_qwen35_9b \
  --save-every 2 --max-seq-length 512 --grad-checkpoint --seed 42
```

**运行成功，退出码 0**。模型显示约 8.954B 参数，其中 1.328M 可训练（0.015%）；LoRA 秩 8，训练最后 2 层。两步训练损失依次 **1.938、1.741**，单批次验证损失 **1.894、1.844**。这些数值只说明小样本上优化过程能执行，不能证明泛化或发电量预测提高。输出 `adapters.safetensors` 为 5,314,175 字节，SHA-256：`e44558302bb03a0391d640264b879018f96f8d78ec67936a09d113f1a3da62b3`；检查 30 个 adapter 张量均为非零，包含更新后的 LoRA B 矩阵。

另以相同模型、数据和参数运行单步训练，命令前加 `/usr/bin/time -l`，并将 `--iters`/`--save-every` 改为 1、adapter 输出改为 `smoke_qwen35_9b_mem`。测得墙钟时间 **12.93 秒**，macOS 报告的 `peak memory footprint` **12,810,277,952 字节（约 11.93 GiB）**，最大 resident set **3,971,514,368 字节**，交换次数 0。该内存数字是 macOS 对此进程的统计，不是完整系统或长期训练的保证；16 GB 机器上正式训练仍需监控。

原始训练日志保存在 Git 忽略目录 `data/processed/research/logs/`：`qlora_smoke_2iter.log` SHA-256 `46b4707aabb8294b6e20af952deb3bf441ac157e4e4e908969f6595658e314db`；`qlora_smoke_1iter_memory.log` SHA-256 `b13901374dc0cbe85d552b292cb126ad5c9c22602fe92e3cd9a7149f0b57d3d3`。训练文件 manifest 的 SHA-256 为 `52cef120b2b5fb85dafa3bc77d826896d8c988473c69f6591ca4ba4d90b700cb`。

## 下一步

9B 短训练已证实可执行。下一步在**同一 MLX 权重**上运行未训练推理与完整训练集上的残差微调，并按同一验证集计算 MAE/RMSE；先以少量更新量级建立学习曲线，再决定是否延长训练。短试运行的 adapter 不作为正式模型。最终测试集保持未评分直到方案冻结。
