# 阶段研究结果包

本目录是两处研究的脱敏统计汇总，不包含私有原始结算或协议前缀。

| 文件 | 内容 |
|---|---|
| `historical-replay.json` | 第 27 轮三组后验评分、两次种子评分、数值稳定数量、配对比较及限制 |
| `mechanism-recovery.json` | 正式 40 次模拟恢复的分组统计和试验明细 |
| `mechanism-comparison.json` | 第 26 轮两个家族、U/G 与 COUNT/PAIR 对照的聚合分数、尾部及筛选结果 |
| `inference-development.json` | 较早推理改进实验的七组聚合报告；事件公式只保留计数与一致率 |
| `export-parity.json` | 新便携生成器相对原冻结适配器的数值核对 |
| `integration-validation.json` | 本次整合的测试、公开检查、来源复核和两个生成器的模拟运行回执 |
| `source-provenance.json` | 实际引用报告、结果和研究脚本的名称、大小、摘要哈希；没有本机绝对路径 |
| `manifest.json` | 本结果包、公开模型、报告和生成器的完整性清单 |

第 27 轮的 `absolute_percentage_error` 是比例，乘 100 得到 MAPE 百分数；旧推理报告的 `mape` 已经是百分数。第 27 轮主表使用 `game_weighted`，旧推理主表使用 `all_quotes`。价格和区间评分均是游戏原单位。

`two_bank_integration_gap_diagnostic` 由原始恢复记录中名称过强的 `integration_gap_bound` 字段转换而来。它只是两组积分估计的差异，不是严格界。源哈希能核对材料版本，不能让未公开原始数据变成可复现数据。

```bash
python scripts/verify_staged_research.py
python scripts/verify_public_release.py
```

前者复核模型概率、表格运算、正式试验数量和清单，后者同时执行仓库公开边界检查。两者都不宣称重新完成了原来的全量后验回放或 40 次拟合。
