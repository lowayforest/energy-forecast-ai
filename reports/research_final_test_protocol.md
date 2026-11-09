# 步骤 10：最终测试评分方案（读取测试评分前固定）

## 冻结对象和预定比较

候选模型固定为步骤 9 的 `mlx-community/Qwen3.5-9B-4bit` 第 **890** 步最终 QLoRA adapter，SHA-256 `ccfaef49cd75e4398770ab31f33a89a551b0f7610f988d27a6911985b8c8603e`。系统提示词 SHA-256 `4098a64de03d2bae1cd2b35307faee82022feca2715e5ff15dc806fd1436ee0c`，训练 SFT manifest SHA-256 `4da10cf8c060aa5e2465db1eac8a62e69055f83f761ea71ebdab45a19ac96f51`。三档边界、中位残差、A/B/C 限制 softmax、加权解码与 `clip(0,24)` 均保持[步骤 9 方案](research_class_v3_protocol.md)和[结果](research_class_v3_results.md)的定义，不用测试标签重新训练、校正或选择检查点。

主比较是最终 adapter 对训练集拟合的条件物理基线的 MAE 差异；另报告原校准物理、未微调 9B 基座和最终 adapter 的 MAE、RMSE、平均偏差、逐站 MAE，以及按日期块重采样 10,000 次（种子 42）的成对 MAE 改善 95% 区间。能量预测仍为 $\widehat E_{s,d}=C_s\widehat H_{s,d}$，其中 $C_s$ 为额定容量 MW、$\widehat H$ 为等效满负荷小时、$\widehat E$ 为 MWh。仅在所有 236 条**无标签预测**完成后运行评分脚本。

测试期为 2019-04-16 至 2019-06-13，四站 59 天共 236 条。原始 `daily_test.jsonl` SHA-256 `7fd4970581a695f2ef497017aab5065e7e0d5db3e9e737b5f971c060b97a14fe`，无标签 `physics_test.jsonl` SHA-256 `097c7bb40d52609e3223380ebc84c3320f5edd542f36ffd50a6864dddc121d57`。条件物理使用训练期已拟合的各站 GHI 高低组偏移；条件 manifest SHA-256 `c08e802f57d7074f20e675f4a51f8a1a1be9be0d227cc5b4002974b04370d6a3`。测试提示仅含 system/user 消息，不含 assistant 标签或目标日实测发电量。

## 一条测试标签意外暴露的处理

在编写本方案前，为查看文件结构执行 `head -1 daily_test.jsonl`，意外打印了 **S-1、2019-04-16** 的真实等效满负荷小时 **0.02703 h**。这发生在步骤 9 模型、提示词和验证比较已于提交 `1030a59` 固定之后，没有据此修改模型或参数；仍须如实视为该条标签已暴露。为使主结果不包含该日，**预先将 2019-04-16 整个四站日期块排除**，以 2019-04-17 至 2019-06-13 的 **58 天、232 条**作为主测试集。另完整报告 59 天、236 条结果作为补充，不隐去被排除的日期。两组结果使用完全相同的冻结预测，不能根据测试结果改选主次。

上述排除会损失一天，且其他 58 天虽然没有查看标签，仍来自同一时间段。日期块 bootstrap 只描述本测试期抽样波动，不能消除训练集、验证集反复试验的选择风险、时间相关性或天气预报发布时间缺失的问题。河北四站的测试表现不能推断上海场站效果。

## 执行顺序

先提交本方案和脚本，再依次执行：

```bash
.venv/bin/python scripts/prepare_research_final_test.py
.venv/bin/python scripts/evaluate_research_final_test.py --variant base
.venv/bin/python scripts/evaluate_research_final_test.py --variant adapter
.venv/bin/python scripts/score_research_final_test.py
```

最后记录模型、输入、输出、比较 JSON 与日志哈希。若测试集表现不如验证集，按实报告；不得在该测试集上继续调整模型后又称其为独立测试。
