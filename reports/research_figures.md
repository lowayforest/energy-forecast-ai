# 课程论文插图记录

## 图 1：物理模型与本地 9B 大模型的方法流程

文件：[PNG](figures/research_method.png) · [SVG 矢量版](figures/research_method.svg) · [PDF 矢量版](figures/research_method.pdf)。生成命令：`.venv/bin/python scripts/plot_research_method.py`；源码为 [`scripts/plot_research_method.py`](../scripts/plot_research_method.py)。内容依据[数据构建](research_daily_dataset.md)、[物理基线](research_physics_baseline.md)、[条件物理校正](research_conditional_v2_results.md)与[v3 残差分档方案](research_class_v3_protocol.md)。

图中上方实线是预测时可用信息到次日发电量的流程：目标日 NWP、目标日前历史发电、Ross/PVWatts 及训练期站点校正、本地 9B 模型、A/B/C 受限 softmax、训练期各类残差中位数还原。下方虚线是训练标签与事后评分，强调目标日真实功率不进入推理提示。绘制的是原先冻结测试的 v3 LoRA 方法；未微调基座在相同路径上不加载 adapter。此图**不是** v4 的“单类别 token 损失”消融流程。

## 图 2：验证期与原冻结测试期结果

文件：[PNG](figures/research_main_results.png) · [SVG 矢量版](figures/research_main_results.svg) · [PDF 矢量版](figures/research_main_results.pdf)。生成命令：`.venv/bin/python scripts/plot_research_main_results.py`；源码为 [`scripts/plot_research_main_results.py`](../scripts/plot_research_main_results.py)。程序在绘图前核验所有冻结评分 JSON、测试标签与预测文件的 SHA-256，具体哈希和新增的“未微调基座相对条件物理”日期块区间见[机器可读绘图数据](figures/research_main_results_data.json)。

- A：验证期 46 天、184 个站点日的 MAE；B：原最终测试中排除一整日后的 58 天、232 个站点日的 MAE。数值来自原保存的评分结果，图中保留三位小数，完整值见[原最终测试报告](research_final_test_results.md)。
- C：主测试期 S-1 至 S-4 的逐站 MAE，显示 S-2 的误差明显较大。
- D：成对的 MAE 降低量，正值表示箭头右侧方法更好。使用与原评分一致的**日期块 bootstrap 10,000 次、种子 42**；新增“基座相对条件物理”的点估计 +0.066251 h，95% 描述性区间 [+0.031723,+0.099809] h。其余两组取自原评分，不重新定义检验口径。区间未处理相邻日时间相关及跨站分布变化。

结果图只呈现原方案冻结后的 v3 测试比较；v4 旧测试数字是事后诊断，不能在同一图中标作新的独立结果。论文正文保留了该消融的文字和表格说明。两图使用 Matplotlib 3.11.2 绘制，PNG 为 300 dpi，SVG/PDF 供论文排版时无损缩放；图内文字用英文，以便在中文或英文模板中保持一致的矢量字形。

## 2026-11-10：版式修订

按 `/Users/a-void/Documents/SystemsModeling/report/figures/paper_fig2_3.png` 中 Fig. 3 的简洁框图形式，方法图改为白底、细线、单一浅色强调，删去图内主标题、副标题、编号条和底部说明；实线表示预测路径，虚线表示训练或事后评分路径。结果图删去整图主标题与底部说明，仅保留 A–D 面板标识、必要的面板名称和坐标轴。图题、排除测试首日及区间解释留在论文正文与图注中，不写入图片。数据与评价数字未改。
