# 训练与验证

从仓库根目录运行 `python train.py -gpu 0`，默认训练完整模型。
本目录是内部实现，无需逐个运行，也没有需要依次训练的子模块。

- `engine.py`：U-Net 参数优化，每轮结束自动完整验证并保存权重。
- `config.py`：合并默认值、模型选择、JSON 与命令行参数，后者优先。
- `losses.py` / `metrics.py`：有效 mask 内训练损失与双频 RMSE/MAE。
- `checkpoint.py`：保存/读取权重、优化器状态和随机状态，支持 epoch 边界续训。
- `evaluation.py` / `reporting.py`：共用验证与可选分组指标计算。
- `provenance.py`：随训练结果记录代码身份，便于复现。

默认 AdamW、学习率 3e-4、权重衰减 1e-4、余弦调度、MAE、20 轮、batch 16、seed 2027。
20 轮为初始预算，不保证收敛。验证在原始 dB 空间进行，各频段先累计有效像素误差，
再计算 RMSE/MAE；两频 RMSE 等权平均用于选取 `best.pt`。

比较模型时保持训练预算和 seed 一致，每次使用独立输出目录。固定物理特征随数据自动处理，
没有单独的预训练阶段。完整命令和参数见 [根目录 README](../README.md)。
