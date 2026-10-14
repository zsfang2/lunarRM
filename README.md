# LunarRM

基于 LunarRM 双频数据的月面路径损耗预测项目。输入为地形高程、发射机位置与频率，
支持 Tiny U-Net、普通 U-Net 和物理特征引导的残差预测。使用官方固定 train/val 划分，
在有效 mask 内按原始 dB 标签分别计算两频段 RMSE/MAE。

训练、续训、评价与诊断代码已准备完成；目前没有正式训练结果。模型方案为本项目设计，
不是官方发布的 baseline；GPU/AMP 尚未实测。

## 目录结构

```text
lunar_rm/               数据、物理特征、模型、训练与评价模块
configs/                模型及消融实验配置
scripts/                训练、评估、检查与报告命令
model.py                N1 普通 U-Net 实现（运行依赖）
train.py / run_train.sh  训练兼容入口，转发到 scripts/train.py
check_setup.py          无梯度前向检查兼容入口
docs/                   数据协议、详细指南、固定样例与结果模板
tests/                  合成数据及接口回归测试
archive/development/    开发规划、验收记录和本地历史日志
runs/                   训练输出（运行时创建，不入 Git）
reports/                评估与测速报告（运行时创建，不入 Git）
cache/                  可选物理特征缓存（不入 Git）
```

## 环境与数据

### 创建 Conda 环境

先进入仓库，再创建并激活名为 `lunar` 的环境：

```bash
cd /home/Users_Work_Space/zsfang/lunarRM
conda create -n lunar python=3.11 pip -y
conda activate lunar
```

如果当前终端未初始化 Conda，先执行 `source "$(conda info --base)/etc/profile.d/conda.sh"`，
再运行 `conda activate lunar`。

### 安装依赖

下面是当前 H100 / 535 系列驱动机器的安装方案，使用
[PyTorch 官方提供的 CUDA 12.1 构建](https://pytorch.org/get-started/previous-versions/)。
先安装指定 PyTorch，再安装项目依赖；在其他机器上应按其驱动选择相应的 PyTorch 构建。

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-dev.txt
```

已按项目源码和官方加载器核对直接依赖：`requirements.txt` 包含 NumPy、PyTorch、Matplotlib；
`requirements-dev.txt` 包含上述运行依赖以及 pytest。间接依赖由 pip 自动安装。
如果只运行训练、不运行测试，第二条命令可改为 `python -m pip install -r requirements.txt`。

安装后检查依赖冲突和 GPU 可见性：

```bash
python -m pip check
python -c "import numpy, matplotlib, torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

GPU 训练前应看到 `GPU available: True`；上述检查不启动训练。后续打开新终端时先执行
`conda activate lunar`，再运行训练命令。以上安装方案尚未在新建的 `lunar` 环境中完成验证。

### 数据

默认数据目录为 `/home/DataDisk/zsfang/lunarRM`，可用 `--data-root` 覆盖。
读取现有官方合并 NPY 数组与索引，不需要重新解压或复制数据；原始数据只读。
数据包含 DEM、Tx 索引、415/5775 MHz 路径损耗图和有效 mask，图像大小为 256×256。
验证集共 4,660 个双频样本项，当前无独立 test。详细语义见 [数据协议](docs/data-contract.md)。

## 模型选择

| 配置 | 方法 |
| --- | --- |
| A0 | 三维自由空间路径损耗（FSPL），无需训练 |
| N0 | Tiny U-Net，直接预测 |
| N1 | 普通 U-Net，容量对照 |
| R0 | Tiny U-Net，输入加入 FSPL 并预测其残差 |
| G0 | 与 R0 同输入、同主干，改为直接预测 |
| R1 | R0 加入地形几何特征 |
| R2 | R1 加入 LoS/Fresnel 特征 |

`configs/R2_*.json` 提供特征、损失和 NLoS 权重消融。可先按 N0 → R0/G0 → R1/R2 组织对照。
R1/R2 的物理特征在 CPU 计算，可在 JSON 配置中设置 `cache_dir` 为独立缓存目录。

## 训练

先检查配置；默认不带 `--execute` 时也只做 dry-run，不读取训练数据或更新权重。

```bash
python scripts/train.py --config configs/N0.json --device cpu --dry-run
```

正式开始 N0 训练：

```bash
MKL_THREADING_LAYER=GNU python scripts/train.py \
  --config configs/N0.json \
  --data-root /home/DataDisk/zsfang/lunarRM \
  --device cuda --output-dir runs/N0 --execute
```

默认设置为 20 epochs、batch size 16、AdamW（学习率 3e-4）、余弦调度、MAE、seed 2027。
可用 `--epochs`、`--batch-size`、`--num-workers`、`--seed` 覆盖；学习率等其他设置写入 JSON。
双频 batch size 必须为偶数，CUDA 不可用时会报错。CPU 训练需显式指定 `--device cpu`；
GPU 混合精度可加 `--amp`。`MKL_THREADING_LAYER=GNU` 用于避免本机多进程加载的线程库冲突。

训练入口也支持 `-gpu x`（或 `--gpu x`），等价于 `--device cuda:x`；二者不可同时传入。
例如指定第 2 号 GPU：

```bash
MKL_THREADING_LAYER=GNU python scripts/train.py \
  --config configs/N0.json -gpu 2 --output-dir runs/N0_gpu2 --execute
```

编号从 0 开始，指当前进程可见 GPU 的编号。如果设置了 `CUDA_VISIBLE_DEVICES=2`，
该进程仅看到一张卡，此时应传 `-gpu 0`。根目录 `train.py`、`run_train.sh` 同样支持此参数。

训练其他方法时，同时更换配置与输出目录，例如 R0：

```bash
MKL_THREADING_LAYER=GNU python scripts/train.py \
  --config configs/R0.json --device cuda --output-dir runs/R0 --execute
```

每次新训练使用不存在的输出目录，避免混写。输出包括：

- `config.json`：解析后的配置、数据和代码身份。
- `metrics.jsonl`：逐 epoch 训练日志与完整验证指标。
- `last.pt`：最近完成 epoch 的恢复状态。
- `best.pt`：按完整双频等权 RMSE 选出的最佳模型。

从最近完成的 epoch 恢复，沿用原配置和总 epoch 预算：

```bash
MKL_THREADING_LAYER=GNU python scripts/train.py --resume runs/N0/last.pt --execute
```

仅支持 epoch 边界恢复。`python train.py` 和 `bash run_train.sh` 接受相同参数。

## 评估与其他工具

训练后独立评估完整验证集；即使 checkpoint 使用 AMP 训练，也可在 CPU 上以 FP32 评价：

```bash
python scripts/evaluate.py --checkpoint runs/N0/best.pt \
  --device cpu --num-workers 0 --output reports/n0-val.json
```

A0 无需 checkpoint，改用 `--experiment A0`。加 `--max-per-band 2` 只作小子集检查，
不能作为完整验证结果。验证集用于选模，指标不应称为独立测试性能。

| 工具 | 用途 |
| --- | --- |
| `scripts/check_config.py` | 配置校验与 dry-run |
| `scripts/audit_data.py` | 只读数据审计 |
| `scripts/check_forward.py` | 双频无梯度前向检查 |
| `scripts/diagnose.py` | 分组指标及固定样例可视化 |
| `scripts/benchmark.py` | 特征生成、网络与端到端推理测速 |
| `scripts/summarize.py` | 汇总已有报告或生成空结果表 |

具体命令与测量边界见 [详细指南](docs/experiment-guide.md)，实验顺序见
[执行清单](docs/run-checklist.md)。开发材料已移至 [开发归档](archive/development/README.md)。

开发测试（默认使用合成数据，不启动正式训练）：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
