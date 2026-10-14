# 实验与实现详细说明

## 当前状态与环境

配置注册已提供 A0/N0/N1/R0/G0/R1/R2，以及 R2 的特征、损失、
NLoS 权重和直接输出消融。数据适配、物理特征、模型、训练恢复、独立评价、诊断、测速和汇总入口均已实现；目前没有正式训练结果。
当前验证包括合成夹具优化正确性测试和真实数据无梯度检查；GPU/AMP 尚未实测。
Tiny 与残差方案是本项目实验设计，不宣称为官方推荐模型；官方参考材料和数据语义见
[数据协议](data-contract.md)，研究背景见 [goal.md](../archive/development/goal.md)。

在仓库根目录使用已安装 NumPy/PyTorch 的 Python 环境；依赖列于 `requirements.txt`
和 `requirements-dev.txt`，需配置新环境时执行 `python -m pip install -r requirements-dev.txt`。
无需下载测试数据或大型模型。运行清单见 [实验执行清单](run-checklist.md)。

```bash
python scripts/check_config.py --config configs/R0.json --check
python scripts/check_config.py --config configs/R0.json --device cpu --dry-run
python -m pytest tests/test_config.py -q
```

`--check` 仅检查配置模式；`--dry-run` 同时检查设备和计划输出目录，两者均打印完整
解析配置，不读取训练数据、不创建输出目录、不启动训练或保存权重。CUDA 请求不可用时明确失败；
CPU 需显式指定。覆盖顺序为公共默认值、实验注册值、JSON、命令行参数；所有相对路径均从仓库根解析。
已有输出目录需显式指定 `--resume`，检查器只检查 checkpoint 文件存在，内容兼容性留给后续加载器。
测试依赖见 `requirements-dev.txt`。数据和评价约定见 [数据协议](data-contract.md)。

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
只读 mmap 在 worker 内重开。输入特征支持高度/TX/band/FSPL、相对坐标/高度、坡度、曲率代理、LoS/Fresnel。
`augment=True` 仅用于 train，同步变换 DEM/TX/原始 GT/mask，随后重算 FSPL。
每个 epoch 的 DataLoader 迭代开始前调用 `dataset.set_epoch(epoch)`，共享 epoch 对 persistent
worker 可见。增强由 seed、epoch、样本身份和频段确定，独立于 worker RNG。
默认目标为原始 dB；显式 `target_mode="official_clipped"` 只截断 `training_target_db`，
评价仍使用未截断的 `target_db`。FSPL 使用原始米制高度，Tx/Rx 地上高度分别为 3/1 m。
默认测试使用合成数据，不需要真实数据或 GPU。spawn 测试显式使用 `MKL_THREADING_LAYER=GNU`，
避免本机 Conda NumPy/PyTorch 的 Intel MKL 与 GNU OpenMP 冲突。

## 物理特征与模型前向

```bash
python scripts/check_forward.py --data-root /home/DataDisk/zsfang/lunarRM --device cpu
```

此命令对七个必需方法各取 val 两频段一个样本，只检查 `(1,1,256,256)` 输出和有限值，
打印真实参数量，不训练。Tiny 使用深度可分离卷积、GroupNorm、多尺度 skip；默认约 0.236M 参数。
R0/G0 同输入同主干；直接输出 `20+208*z`，残差输出 `FSPL+208*r`，残差不加非负限制。
N1 是原普通 U-Net（约 1.964M），不标为 Tiny。参数目标命中不代表精度或时延目标已经达成。

几何原始量中，相对行列/高度单位为米，坡度无量纲，Laplacian 曲率代理为 1/m；
输入通道分别将相对坐标除以 256、相对高度除以 496，坡度/曲率代理原样输入。
剖面使用双线性地形插值，只采内部点，默认水平间隔不超过 1 m，可设置 0.5/1/2 m。
净空为直射线高于地形的高度，最小净空非负定义为 LoS；Fresnel 为内部最小净空/半径。
半径使用三维总距离与水平路径分数近似；零水平或无内部采样点路径约定 LoS=true、两个净空指标为 0。
接收像素分块，临时数组先做内存预算检查，极小步长超预算会拒绝；几何 LoS 始终独立于监督 mask。

可在配置设置 `cache_dir` 启用独立物理缓存。键包括输入数据身份和 DEM 摘要、算法版本、
参数、dtype、TX/terrain/sample、实际依赖的频段及增强；不使用 GT/mask 生成物理特征。
缓存原子保存，命中时验证字段、版本、shape/dtype/有限值，损坏或不完整缓存会报错。

## 原始 dB 独立评价与训练目标

```bash
# A0 不需要 checkpoint；此命令会完整读取 val，不做参数更新。
python scripts/evaluate.py --experiment A0 --device cpu --num-workers 0 \
  --data-root /home/DataDisk/zsfang/lunarRM --output reports/a0-val.json
# 学习模型必须指定 checkpoint；默认从中重建配置。
python scripts/evaluate.py --checkpoint runs/N0/best.pt --device cpu \
  --num-workers 0 --output reports/n0-val.json
```

训练框架使用同一评价函数，生成的 checkpoint 可直接用于此独立评价入口。
独立评价与诊断始终采用 FP32，解析报告配置中 `amp=false`；训练时启用 AMP 的 checkpoint
也可用 `--device cpu` 评价。checkpoint 本身及其训练配置不变，训练/测速的 AMP 仍要求 CUDA。
`--max-per-band 2` 直接选取各频段固定的前两项，输出标记为 `subset`，禁止用于完整验证选优。
完整评价核对 `(band,sample_id)` 唯一身份和全部索引覆盖；对官方完整 val 应读取 4,660 项。
逐频段以 float64 累计原始 dB GT 的 SSE/SAE/有效像素数，最后计算 RMSE/MAE，
双频分数为两个 RMSE 的算术平均。空频段返回 null；没有两个有效频段时不能选 best。
固定预测下累计顺序不随 batch 分界变化；神经网络在不同 batch/kernel 下的浮点舍入可能有微小差异。
报告包含解析后配置、代码摘要、数据/官方加载器身份和 checkpoint SHA256；输出文件禁止覆盖或写进数据根。

`BalancedBatchSampler` 每个 batch 的频段项数相同，尾 batch 保留且仍均衡，不丢样本；
batch size 必须能被频段数整除。状态包含 seed/epoch 和尾 batch 策略，仅承诺 epoch 边界恢复。
masked MAE/MSE 在每频段有效像素内求均值，再对非空频段等权平均；
可选 NLoS 项同样只在有有效 NLoS 像素的频段内归约后等权，加权系数为 `nlos_alpha`。
损失用 FP32，优化量除以 208 或 208²；未缩放误差分别是 dB 或 dB²。
空 mask 项计数并跳过；全空 batch 返回无损失，不允许伪造一次优化更新。

## 显式训练与恢复

```bash
# 默认和 --dry-run 都只检查配置，不创建运行目录或更新权重。
python scripts/train.py --config configs/N0.json --device cpu --dry-run
# 后续明确决定开始正式训练时才执行：
MKL_THREADING_LAYER=GNU python scripts/train.py --config configs/N0.json \
  --device cuda --output-dir runs/N0 --execute
# 默认从 checkpoint 重建原配置，继续相同总 epoch 预算：
MKL_THREADING_LAYER=GNU python scripts/train.py --resume runs/N0/last.pt --execute
```

AdamW 与余弦调度按配置运行，AMP 仅在显式开启且使用 CUDA 时启用。
损失或梯度非有限立即中止；全空 mask batch 不调用 optimizer，epoch 调度仍推进。
日志 `metrics.jsonl` 记录有效像素、跳过项/空 batch、截断数量、学习率、更新数、
训练 batch 平均误差（不作为验证指标）及完整 raw dB 验证结果。
仅完整双频且两个频段有有效像素时更新 `best.pt`；否则保留 null，不伪造 best。

`last.pt`/`best.pt` 以同目录临时文件、fsync、原子替换保存。内容包括模型、AdamW、
调度器、AMP、epoch/更新数、Python/NumPy/Torch/CUDA 随机数、sampler、最佳指标、
解析后配置、数据身份、源码摘要和运行环境。**仅承诺 epoch 边界恢复**；不支持 batch 中途恢复。
恢复保留原总 epochs（余弦预算），拒绝配置/数据/特征顺序/模型不兼容、调度器或优化器重置。
同目录只允许续接自己的 `last.pt`，并核对日志末行与 checkpoint 边界；若保存与日志追加之间
异常退出造成不一致，可指定新的 `--output-dir` 从 checkpoint 分支恢复，原日志不被修改。
CPU 单线程合成测试已验证连续运行与恢复逐值一致；GPU/AMP 正确性尚未在 GPU 上实测。
spawn worker 在 Conda MKL INTEL 环境下须通过启动环境指定 GNU 后端，程序会明确拒绝已知冲突组合。

根目录 `train.py` 与 `run_train.sh` 已转发到 `scripts/train.py`，默认 dry-run；
`check_setup.py` 转发到 `scripts/check_forward.py`，需显式提供 `--data-root`。
旧原型参数 `--base-channels` 改为 `--width`，其余训练设置写入 JSON 配置；旧 checkpoint
缺完整身份/恢复状态，不支持新框架续训。`model.py` 普通 U-Net 保留为 N1；原型历史在 Git 中保留。

## 分组诊断与固定样例

```bash
# 仓库提供固定 val 首项的双频清单，所有方法复用它；不按预测质量挑图。
python scripts/diagnose.py --experiment A0 --device cpu --max-per-band 2 \
  --examples docs/examples-val.json --output-dir reports/a0-diagnostics-smoke
# 后续训练完成后，用同一清单完整诊断；去掉 max-per-band 即完整 val。
python scripts/diagnose.py --checkpoint runs/N0/best.pt --device cpu \
  --examples docs/examples-val.json --coverage-edges 0.5 0.9 --output-dir reports/n0-diagnostics
```

`diagnostics.json` 包含 LoS/NLoS 与 valid mask 交集、single/multi、固定覆盖率区间及地形等权指标；
空组返回 null 和计数。LoS/Fresnel 是本项目几何计算，覆盖率桶是研究分组，不是官方传播分类。
默认边界 `[0.5,0.9]` 及精确分箱规则写入报告，不从 val 拟合。
每个固定身份输出六面板 PNG 与原始数组 NPZ：DEM/Tx、GT、预测、有效 mask、有效像素误差、FSPL。
色标跨方法固定；无效 GT 填充值不参加误差计算。修改清单时对所有方法一致使用并保留其 SHA256。

## 推理成本与结果汇总

```bash
# 当前允许的随机权重测速检查；不会更新权重，不是精度结果。
python scripts/benchmark.py --experiment R0 --allow-untrained --device cpu \
  --threads 1 --warmup 3 --repeats 10 --output reports/r0-untrained-timing.json
# 后续训练后以同样条件测量 checkpoint；不传 allow-untrained。
python scripts/benchmark.py --checkpoint runs/N0/best.pt --device cpu \
  --threads 1 --warmup 3 --repeats 10 --output reports/n0-timing.json
# 当前即可生成全部方法及消融的空表；所有性能栏留空。
python scripts/summarize.py --template --output-dir reports/pending-results
# 已有报告可独立汇总；可同时追加 evaluate/diagnose 的 JSON 路径。
python scripts/summarize.py reports/r0-untrained-timing.json --output-dir reports/timing-catalog
```

测速 batch=1、固定 `--index`（默认 val 首项），报告预热/重复数、硬件/线程/精度、
各通道中位数/P95。特征生成不使用 GT/mask；端到端从内存中的原始 DEM 到 CPU dB 图，
含 CPU 特征、H2D、网络和 D2H，不含磁盘读取。网络通道仅测设备内前向；缓存输入装配单列，
不能替代无缓存端到端。CUDA 测量会同步。单样本重复时间不代表全数据延迟分布。
复杂度只计 Conv2d MAC/FLOP，并列出未覆盖算子；CPU 内存是进程生命周期峰值 RSS，
GPU 峰值覆盖无缓存通道含预热。没有同设备 RT 测量，不产生加速倍数。

汇总产生 `results.csv` 快览和 `results.json` 完整目录；JSON 原样保留每份报告，包含
解析配置、数据/官方加载器身份、代码摘要、checkpoint SHA256、分组与测速边界。
每个来源独占一行，保留 full/subset/single_sample 和 analytic/checkpoint/untrained 标记；
列出相对首份报告的配置字段、数据、代码和硬件差异，不自动拼接权重不同的精度与时延，
不排名或自动平均 seed。只有同协议完整 val 可进行主对照；CSV 的子集分数仅用于检查。
静态空模板见 [results-template.json](results-template.json)，字段说明见执行清单。
500 ms、0.1M–0.5M 参数及精度改善均为实验目标，实测报告，不是代码准备的硬门槛。
当前无独立 test，Kaggle 格式检查不可填入测试性能栏。
