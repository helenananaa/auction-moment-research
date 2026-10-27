# beta.7：精确后验与阶段研究

这版把已经合入 main 的推理改进打包为可安装的实验性助手，并附上两处研究的阶段成果。

- 默认使用被动抓包输入和精确条件后验；仍以 `world-model-v2-S.json` 为先验，v6 保留为对照与回退。
- 包含整数线索约束、并行采样、后台细化，以及尺寸平均值分组缓存修复；OCR/YOLO 仍可选。
- 附带 R2／KG_mix2 候选参数、离线生成器、阶段报告和脱敏统计。候选未替换默认先验：点误差没有明确胜出，已消费历史样本上的区间评分改善约 10%。
- 研究包版本 `0.1.1`，助手版本 `0.2.0b7`。五个运行时模型与 beta.6 字节一致；旧归档和 Release 不覆盖。

## 下载与启动

下载完整 ZIP `auction-moment-beta.7-windows.zip`，解压到一个新目录。需要 Windows、Python 3.12，以及下载 Python 依赖的网络连接。

```powershell
pwsh -File .\install.ps1
.\start-assistant.cmd
```

如果只有 Windows 自带 PowerShell，可用 `powershell -File .\install.ps1`。脚本先核对文件清单，再在该目录创建 `.venv` 并安装两个 wheel；不会修改其他项目的 Python 环境。首次安装需要下载 OCR、OpenCV 和 PyTorch 等依赖，ZIP 不是离线依赖仓库。

只校验模型、不连接设备：

```powershell
.\.venv\Scripts\auction-vision-assistant.exe --models .\models\latest-model-v6-v2.0.0-beta.7 --verify-models
```

默认抓包模式需要 ADB、已开启 root 的模拟器，以及模拟器内的 `tcpdump`；可用 `start-assistant.cmd --input ocr` 切换到截图识别。完整说明见 ZIP 内的 README。

## 验证与使用边界

发布资产附 `release-validation.json`、`beta-release-manifest.json` 和 `SHA256SUMS`，分别记录本次安装/启动验证、文件来源与哈希。两个 wheel、完整 ZIP、五个模型也提供为单独资产。

活动已经下线，尚未在正式活动环境完成整局端到端验收，也没有新时间块前瞻校准。所有输出保持 `actionable=false`，不会自动出价。历史回放、合成预测与无设备界面启动均不能替代现场验证；此次发布为预发布版本，项目仍处于暂时休眠状态。
