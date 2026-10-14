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

安装依赖并激活环境后，在仓库目录直接运行：

```bash
conda activate lunar
bash run_train.sh -gpu 0
```

这会直接训练完整模型 **R2**，默认读取 `/home/DataDisk/zsfang/lunarRM`，
训练 20 轮，每轮自动验证，最佳权重保存到 `runs/R2/best.pt`。
脚本自动设置线程库环境变量。换 GPU 只需修改 `-gpu` 后的编号。

```bash
bash run_train.sh --experiment N0 -gpu 0  # 改跑基础模型，保存到 runs/N0
bash run_train.sh -gpu 0 --dry-run       # 只检查，不训练
bash run_train.sh --resume runs/R2/last.pt -gpu 0  # 中断后恢复
```

如果 `runs/R2` 已存在，新实验可加 `--output-dir runs/R2_run2`。

日常使用建议明确写出 **GPU 编号、实验名称和输出目录**，无需修改脚本。
以下默认值适用于 `run_train.sh` 未提供自定义配置或恢复 checkpoint 的情况：

| 参数 | 默认值 | 使用建议 |
| --- | --- | --- |
| `-gpu` / `--gpu` | 未指定时使用默认 CUDA 设备，通常为可见的第 0 张卡 | 每次明确指定空闲 GPU；不能与 `--device` 同时使用 |
| `--experiment` | `R2` | 每次明确指定，便于记录模型和消融实验 |
| `--output-dir` | `runs/实验名称` | 每次新训练指定尚不存在的目录，区分模型、seed 和重复运行 |
| `--data-root` | `/home/DataDisk/zsfang/lunarRM` | 数据移动或换机器时修改；路径不变可省略 |
| `--epochs` | `20` | 初始训练预算；模型对照时保持一致，恢复时保持原总轮数 |
| `--batch-size` | `16` | 根据显存调整；双频训练必须为偶数 |
| `--num-workers` | `4` | 数据加载进程数，根据 CPU 和内存调整 |
| `--seed` | `2027` | 模型对照使用相同 seed，多次重复实验再更换 |
| `--amp` | 关闭 | 可选开启 GPU 混合精度；速度、显存和稳定性需实测 |

推荐的新训练命令模板：

```bash
bash run_train.sh \
  --experiment R2 \
  -gpu 0 \
  --output-dir runs/R2_seed2027 \
  --epochs 20 \
  --batch-size 16 \
  --seed 2027
```

数据路径变更时，在命令末尾追加 `--data-root /你的数据目录`。
训练会自动进行每轮完整验证，不需要额外添加验证参数。
学习率、权重衰减等设置写在 JSON 中，通过 `--config configs/R2.json` 加载；
命令行显式传入的训练参数覆盖 JSON 中的对应值。

重新训练时换一个输出目录，例如 `--output-dir runs/R2_seed2027_run2`。
中断后继续同一次训练则加载 `last.pt`，沿用其中的配置和总 epoch 预算：

```bash
bash run_train.sh --resume runs/R2_seed2027/last.pt -gpu 0
```

恢复时无需重复指定实验、配置和输出目录，也不要重新设置一套训练超参数。

下面是详细入口：`scripts/train.py` 默认不带 `--execute` 时只做 dry-run，不读取训练数据或更新权重。

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

仅支持 epoch 边界恢复。`python train.py` 与 `scripts/train.py` 一样默认只检查；
`bash run_train.sh` 接受相同参数，但默认选择 R2 并直接训练。

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
