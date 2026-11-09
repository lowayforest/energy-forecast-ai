# 步骤 3：预报天气驱动的光伏物理基线

## 任务与数据边界

目标是在 [AI-PVOD 四站数据](https://huggingface.co/datasets/yujiaA/AI-PVOD)上预测次日发电量（MWh），为后续本地大模型及微调实验提供对照和输入特征。本步骤沿用[步骤 2 的时间划分](research_daily_dataset.md)：训练集截至 2019-02-28，验证集为 2019-03-01 至 2019-04-15，测试集为 2019-04-16 至 2019-06-13。**物理计算只读取目标日的 15 分钟 `nwp_globalirrad`、`nwp_temperature`，校准只读取训练集目标日发电量**。目标日实测 `power` 不作为预测时的天气输入，`lmd_*` 现场气象观测完全不用。测试集已经生成无标签预测文件，但直到方案冻结前不计算测试误差。

原 [PVOD 论文](https://doi.org/10.1016/j.solener.2021.09.050)说明其 NWP 来自每日运行的 WRF 预报；整理后的 CSV 只有预报有效时刻，没有逐条发布时间。因此“次日可得”依赖原论文描述，不能作为逐条可独立审计的事实。

## 模型、公式和单位

采用 [pvlib 的 Ross 组件温度模型](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.ross.html)与 [PVWatts 直流功率模型](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html)。PVWatts 原始技术报告为 [Dobos, 2014, NREL/TP-6A20-62641](https://doi.org/10.2172/1158421)。对站点 $s$ 的 15 分钟预报时刻 $i$：

$$T_{s,i}^{\mathrm{cell}}=T_{s,i}^{\mathrm{air}}+\frac{\mathrm{NOCT}-20}{800}G_{s,i},\qquad \mathrm{NOCT}=45\,^{\circ}\mathrm C.$$

$$P_{s,i}^{\mathrm{dc}}=C_s\frac{G_{s,i}}{1000}\left[1+\gamma\left(T_{s,i}^{\mathrm{cell}}-25\right)\right],\qquad \gamma=-0.004\,^{\circ}\mathrm C^{-1}.$$

$$P_{s,i}^{\mathrm{raw}}=\min\!\left(C_s,\max(0,\eta P_{s,i}^{\mathrm{dc}})\right),\quad \eta=0.96;\qquad E_{s,d}^{\mathrm{raw}}=0.25\sum_{i\in d}P_{s,i}^{\mathrm{raw}}.$$

这里 $G$ 为 `nwp_globalirrad`（W/m²），$T^{\mathrm{air}}$ 为 `nwp_temperature`（°C），$C_s$ 为站点额定容量（MW），$P$ 为功率（MW），$E$ 为日发电量（MWh）。使用 `pvlib 0.16.1` 实现；`numpy 2.5.4`、`pandas 3.1.0` 用于数据处理。

**模型简化**：Ross 和 PVWatts 所需的实际输入分别是组件面辐照度、有效辐照度。本数据包没有四站可核验的组件倾角、朝向与组件参数，因此本实验以水平面总辐照度 GHI 近似两者；这一步会产生系统误差。用公开的站点额定容量 $C_s$ 作为 PVWatts 直流参考容量，也是容量口径上的假设。NOCT、温度系数、逆变器效率是统一的**实验假设**，不是四站设备的已知实测参数。固定逆变器效率乘法也是简化近似，并非完整逆变器模型。Ross 默认的风速假设约为 1 m/s；本版虽有 NWP 风速，尚未将其用于热模型。

站点经验系数仅用训练日估计。令 $\mathcal T_s$ 为站点 $s$ 训练集中 $E_{s,d}^{\mathrm{raw}}\geq0.1C_s$ MWh 的日期，则

$$a_s=\operatorname{median}_{d\in\mathcal T_s}\frac{E_{s,d}^{\mathrm{actual}}}{E_{s,d}^{\mathrm{raw}}},\qquad \widehat E_{s,d}^{\mathrm{cal}}=\min\left(a_sE_{s,d}^{\mathrm{raw}},24C_s\right).$$

系数是经验缩放，会同时吸收组件朝向、装机量口径、遮挡、天气预报偏差等效应，**不能解释成纯粹的系统损耗率**。它在训练集上固定后用于验证和测试，不随验证集重新拟合。`0.1C_s` 是剔除极低辐照日、避免比值不稳定的实现阈值，单位 MWh；最后的 $24C_s$ 是日发电量上界。

## 指标与实测结果

跨站汇总以容量归一化的等效满负荷小时数计分，避免 20 MW 站点的 MWh 误差天然压过 6.6 MW 站点。记 $e_j=(\widehat E_j-E_j)/C_j$（小时），则

$$\mathrm{MAE}_H=\frac1N\sum_{j=1}^N|e_j|,\qquad \mathrm{RMSE}_H=\sqrt{\frac1N\sum_{j=1}^Ne_j^2}.$$

| 站点 / 集合 | 样本数 | 原始 MAE（h） | 校准 MAE（h） | 原始 RMSE（h） | 校准 RMSE（h） | $a_s$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S-1，训练 | 188 | 0.8641 | 0.5802 | 1.0539 | 0.7542 | 0.75732562 |
| S-2，训练 | 234 | 1.3077 | 0.9943 | 1.7421 | 1.4497 | 1.36628215 |
| S-3，训练 | 234 | 0.8047 | 0.7100 | 1.0764 | 0.9866 | 0.88630825 |
| S-4，训练 | 234 | 0.7607 | 0.7598 | 1.0163 | 1.0155 | 0.99456074 |
| **整体，训练** | **890** | **0.9379** | **0.7704** | **1.2684** | **1.0954** | — |
| S-1，验证 | 46 | 0.7645 | 0.7051 | 1.0038 | 0.8548 | 0.75732562 |
| S-2，验证 | 46 | 0.9816 | **1.0689** | 1.1254 | **1.3181** | 1.36628215 |
| S-3，验证 | 46 | 0.8061 | 0.6430 | 1.0317 | 0.8169 | 0.88630825 |
| S-4，验证 | 46 | 0.7046 | 0.7039 | 0.8756 | 0.8698 | 0.99456074 |
| **整体，验证** | **184** | **0.8142** | **0.7802** | **1.0131** | **0.9864** | — |

整体验证 MAE 下降约 4.2%，幅度较小。**S-2 验证误差升高**，说明训练期固定缩放未必能跨季节适用；校准改善不能作为普遍结论。S-2 的系数超过 1，也与 GHI 代替组件面辐照度、容量口径或预报偏差等因素相容，不能单独推断原因。若未来取得真实倾角、朝向和设备规格，应重新建立更完整的组件面辐照度与逆变器模型。

## 复现与输出

从已核验的压缩包和步骤 2 日样本运行：

```bash
.venv/bin/python scripts/build_research_physics.py
```

脚本在被 Git 忽略的 `data/processed/research/` 写入 `physics_train.jsonl`、`physics_valid.jsonl`、`physics_test.jsonl` 和 `physics_manifest.json`。预测 JSONL 只有站点、日期、原始与校准预测，**没有标签**。manifest 记录源数据 SHA-256、步骤 2 manifest SHA-256、模型参数、校准样本数、训练与验证指标。三个预测文件 SHA-256 依次为：

- 训练：`63c3bf4972a505b6a9c97590224f52e5812b64024b30db6b469aa07e59502db4`
- 验证：`62aedeb0d51d9d92604f907fdbb37cf4a66c396fc97a00cf5f6785678b4a1881`
- 测试：`097c7bb40d52609e3223380ebc84c3320f5edd542f36ffd50a6864dddc121d57`

测试集为 236 个站点日，仅生成预测，**未评分**。下一步将把该物理预测作为大模型可见的数值锚点，先评估未微调的本地模型，再进行小规模 QLoRA 微调；两者均使用同一时间划分与验证指标。
