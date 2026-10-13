# LunarRM U-Net baseline

## 新实验配置（开发中）

配置注册已提供 A0/N0/N1/R0/G0/R1/R2，以及 R2 的特征、损失、
NLoS 权重和直接输出消融。新模型和训练引擎尚待实现；目前没有正式训练结果。
旧入口仍使用官方截断后的归一化标签，其验证结果不等同于完整的原始 dB GT 评价。

```bash
python scripts/check_config.py --config configs/R0.json --check
python scripts/check_config.py --config configs/R0.json --device cpu --dry-run
python -m pytest tests/test_config.py -q
```

`--check` 仅检查配置模式；`--dry-run` 同时检查设备和计划输出目录，两者均打印完整
解析配置，不读取训练数据、不创建输出目录、不启动训练或保存权重。CUDA 请求不可用时明确失败；
CPU 需显式指定。覆盖顺序为公共默认值、实验注册值、JSON、命令行参数；所有相对路径均从仓库根解析。
已有输出目录需显式指定 `--resume`，检查器只检查 checkpoint 文件存在，内容兼容性留给后续加载器。
测试依赖见 `requirements-dev.txt`。数据和评价约定见 [数据协议](docs/data-contract.md)。

## 数据审计与原始标签读取

```bash
# 部分扫描：只扫描每个数组的前 32 行，报告明确标为 partial_pass
python scripts/audit_data.py --data-root /home/DataDisk/zsfang/lunarRM --max-rows 32
# 全量只读扫描；报告不得放在数据目录，已有报告不会被覆盖
python scripts/audit_data.py --data-root /home/DataDisk/zsfang/lunarRM --output reports/audit.json
python -m pytest -q
# 显式读取真实数据，在 train/val 的双频首尾及切换点对照官方加载器
python -m pytest tests/test_data.py -q -m real_data
```

审计核对固定 train/val 计数、索引与 DEM 映射、TX、数组形状/dtype、地形隔离，
分块统计有限值、有效像素覆盖率及有效标签超界数量。JSON/CSV/官方源码记录 SHA256；
大型数组记录形状、dtype、大小和修改时间，这不是数组全内容校验和。

`lunar_rm.data.LunarDataset` 返回独立的米制 DEM、原始 dB GT、官方 mask、TX 和双频身份；
只读 mmap 在 worker 内重开。当前输入特征支持高度/TX/band，物理特征与增强在后续模块接入。
默认测试使用合成数据，不需要真实数据或 GPU。spawn 测试显式使用 `MKL_THREADING_LAYER=GNU`，
避免本机 Conda NumPy/PyTorch 的 Intel MKL 与 GNU OpenMP 冲突。

This baseline trains one frequency-conditioned U-Net for both challenge bands.
The three input channels are normalized terrain height, the transmitter one-hot
map, and a constant band channel (0 for 415 MHz and 1 for 5.775 GHz). The model
predicts normalized pathloss and is trained only on valid ray-traced pixels.

The official `lunar_dataset.py` in the dataset directory performs memory-mapped
loading, heightmap lookup, TX reconstruction, mask unpacking, normalization, and
aligned augmentation.

## Train

Check the dataset/model interface without training or changing weights:

```bash
cd /home/Users_Work_Space/zsfang/lunarRM
python check_setup.py
```

The committed full-training settings are in `run_train.sh`:

```bash
cd /home/Users_Work_Space/zsfang/lunarRM
bash run_train.sh
```

For a short CPU trial:

```bash
python train.py \
  --epochs 2 \
  --batch-size 4 \
  --base-channels 8 \
  --max-train-steps 100 \
  --max-val-batches 50 \
  --output-dir runs/cpu_trial
```

Metrics are appended to `metrics.jsonl`. `best.pt` is selected by the unweighted
mean of the 415 MHz and 5.775 GHz validation RMSE values in dB.

Checkpoints contain the model, optimizer, learning-rate scheduler, AMP scaler,
epoch, global step, and best metric. Resume with `--resume path/to/last.pt`.
