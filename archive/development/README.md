# 开发归档

这里保存代码准备阶段的历史材料，训练时无需读取。

| 内容 | 原位置 | 用途 |
| --- | --- | --- |
| [draft.md](draft.md) | 仓库根目录 | 原始研究构想 |
| [goal.md](goal.md) | 仓库根目录 | 项目背景与实验设计 |
| [plan.md](plan.md) | 仓库根目录 | 实施计划 |
| [implementation-audit.md](implementation-audit.md) | docs/ | 验收、审查和测试记录 |
| humanize/ | .humanize/ | 已完成的开发循环、审查记录及经验库，仅本地保留 |
| reports/ | reports/ | 开发阶段的数据审计、随机权重测速和前向检查输出，仅本地保留 |

归档时保留原文件内容；历史记录中的路径和状态以当时环境为准。
其中 `humanize/` 和 `reports/` 继续由 Git 忽略，不上传本地工作流日志。
后续正式实验仍输出到仓库根目录的 `runs/`、`reports/`，由命令自动创建。

当前用法见 [仓库 README](../../README.md)，详细说明见
[实验指南](../../docs/experiment-guide.md)。训练代码、测试和配置保持现有位置。
