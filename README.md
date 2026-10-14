# LunarRM

月面双频路径损耗预测：输入地形、发射机位置和频段，输出 256×256 路径损耗图（dB）。
使用官方 train/val 划分。模型为本项目实现，并非官方发布的 baseline；目前没有正式训练结果。

## 文件结构

仓库按“数据加载 → 模型 → 训练与验证”组织。下面列出运行相关文件；
各 Python 包中的 `__init__.py` 仅用于包初始化，在目录树中省略。

```text
lunarRM/
├── README.md                   项目介绍与运行方法
├── requirements.txt            运行依赖
├── .gitignore                  排除数据、权重、缓存与本地归档
├── train.py                    训练入口，默认训练完整模型
├── run_train.sh                转入仓库目录并调用 train.py
├── evaluate.py                 加载权重，独立评价验证集
├── data/
│   ├── README.md               数据格式与加载说明
│   ├── paths.py                默认数据集路径
│   ├── loader.py               读取样本并组装模型输入与标签
│   ├── storage.py              只读访问 NPY 数组和 CSV 索引
│   ├── features.py             自动计算固定物理与地形特征
│   ├── transforms.py           归一化及训练数据增强
│   ├── sampling.py             双频均衡采样
│   └── cache.py                可选特征缓存
├── models/
│   ├── README.md               模型选择与可选对照说明
│   ├── network.py              轻量 U-Net、模型构建与输出组合
│   ├── unet.py                 普通 U-Net 对照网络
│   └── presets.py              各模型的输入特征和输出方式
├── training/
│   ├── README.md               训练、验证与续训说明
│   ├── config.py               配置解析与参数检查
│   ├── engine.py               训练循环和每轮自动验证
│   ├── losses.py               有效区域内的训练损失
│   ├── metrics.py              双频 RMSE / MAE
│   ├── evaluation.py           加载模型并执行验证
│   ├── checkpoint.py           权重保存、读取和续训状态检查
│   ├── reporting.py            可选分组指标统计
│   └── provenance.py           记录本次运行的代码版本信息
└── configs/
    ├── README.md               配置使用方法
    ├── physics_unet.json       完整模型常用超参数
    └── tiny_unet.json          基础模型常用超参数
```

这些目录是代码职责划分，**不是需要逐个训练的模块**。训练入口自动加载数据、构造模型、
更新 U-Net 参数并进行验证；固定特征与输出组合自动执行。独立验证入口加载训练好的权重，
使用相同的数据处理与模型构建流程。

| 你想做什么 | 入口或文件 |
| --- | --- |
| 开始训练 | [train.py](train.py)，命令见下文 |
| 更换数据位置 | 命令行 `--data-root`；长期修改默认值可编辑 [data/paths.py](data/paths.py) |
| 修改学习率、轮数等超参数 | [configs/physics_unet.json](configs/physics_unet.json)，通过 `--config` 加载 |
| 了解或修改网络结构 | [models/network.py](models/network.py)；模型选择见 [模型说明](models/README.md) |
| 查看训练循环、损失与验证方式 | [训练说明](training/README.md) |
| 验证已有权重 | [evaluate.py](evaluate.py) |

运行产生的目录均不纳入 Git：`runs/` 保存权重和训练日志，`reports/` 保存独立验证报告，
`cache/` 保存可选特征缓存，使用时自动创建。原始数据默认位于仓库外的
`/home/DataDisk/zsfang/lunarRM`，`data/` 目录只放加载代码。

开发测试、检查工具和历史文档仅保留在本地 `archive/`，不纳入 Git；克隆仓库不需要这些文件。
各目录的细节可继续阅读 [数据说明](data/README.md)、[模型说明](models/README.md)、
[训练说明](training/README.md) 和 [配置说明](configs/README.md)。

## 安装环境

```bash
conda create -n lunar python=3.11 pip -y
conda activate lunar
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__); print('GPU available:', torch.cuda.is_available())"
```

上面采用 CUDA 12.1 的 PyTorch 构建，适用于本机 H100 / 535 系列驱动；其他机器按驱动选择
[PyTorch 构建](https://pytorch.org/get-started/previous-versions/)。新建 lunar 环境的安装验证尚未完成。
如果 `conda activate` 不可用，先运行 `source "$(conda info --base)/etc/profile.d/conda.sh"`。

## 直接训练

进入仓库并激活环境后：

```bash
python train.py -gpu 0
# 或者：bash run_train.sh -gpu 0
```

默认训练完整模型 `physics_unet`，读取 `/home/DataDisk/zsfang/lunarRM`，训练 20 轮，
每轮自动验证，最佳权重保存到 `runs/physics_unet/best.pt`。
**只有 U-Net 网络需要训练**。其他固定处理在训练和验证/推理的数据流程中自动完成，
不需要单独训练、预加载或分阶段启动。各模型独立训练，无需先训练基础模型再训练完整模型。

日常建议明确指定 GPU、模型和输出目录：

```bash
python train.py --experiment physics_unet -gpu 0 --output-dir runs/physics_unet_seed2027
```

先跑基础 U-Net 对照：

```bash
python train.py --experiment tiny_unet -gpu 0 --output-dir runs/tiny_unet_seed2027
```

| 参数 | 默认值 | 什么时候修改 |
| --- | --- | --- |
| `-gpu` / `--gpu` | 默认 CUDA 设备，通常为可见的第 0 张卡 | 建议每次指定空闲 GPU |
| `--experiment` | `physics_unet` | `tiny_unet` 为基础模型；其他可选对照见 models/README.md |
| `--data-root` | `/home/DataDisk/zsfang/lunarRM` | 换数据位置或机器时修改 |
| `--output-dir` | `runs/模型名称` | 新训练使用尚不存在的目录 |
| `--epochs` | 20 | 初始预算；模型对比时保持一致 |
| `--batch-size` | 16 | 根据显存调整，双频时必须为偶数 |
| `--num-workers` | 4 | 根据 CPU/内存调整加载进程数 |
| `--seed` | 2027 | 对照模型保持一致，重复实验时更换 |
| `--lr` | 0.0003 | 调整学习率 |
| `--weight-decay` | 0.0001 | 调整权重衰减 |
| `--amp` | 关闭 | 可选混合精度，GPU/AMP 尚未实测 |
| `--cache-dir` | 不使用缓存 | 可指定 `cache/physics`，自动缓存固定物理特征 |
| `--dry-run` | 不启用 | 仅检查配置，不训练 |

GPU 编号从 0 开始，指进程可见的设备。如果设置 `CUDA_VISIBLE_DEVICES=2`，只剩一张可见卡，
应传 `-gpu 0`。`--device cpu` 可指定 CPU，不能与 `-gpu` 同时使用。
线程库环境变量由入口自动设置，不需要手工添加。

要集中编辑超参数，可修改 [完整模型配置](configs/physics_unet.json)，然后运行：

```bash
python train.py --config configs/physics_unet.json -gpu 0
```

仅指定 `--experiment` 使用内置默认值；使用 `--config` 才会读取该 JSON。
命令行显式参数优先于 JSON。检查配置可加 `--dry-run`。

## 恢复与验证

中断后继续同一次训练：

```bash
python train.py --resume runs/physics_unet_seed2027/last.pt -gpu 0
```

沿用 checkpoint 中的设置和原总 epoch 预算，无需再次指定模型、配置或输出目录。
仅支持 epoch 边界恢复；恢复时保持原设备配置。重新训练则使用新输出目录。

输出目录保存 `config.json`（实际配置）、`metrics.jsonl`（每轮指标）、
`last.pt`（续训状态）和 `best.pt`（双频等权 RMSE 最佳权重）。

训练结束后独立验证最佳权重：

```bash
python evaluate.py --checkpoint runs/physics_unet_seed2027/best.pt \
  --device cuda:0 --output reports/physics_unet-val.json
```

完整验证包含 4,660 个双频样本项；在官方有效 mask 内，使用原始 dB 标签计算各频段 RMSE/MAE。
验证集用于选模，不能作为独立测试集性能。更多说明见
[数据加载](data/README.md)、[模型](models/README.md)、[训练](training/README.md)。
