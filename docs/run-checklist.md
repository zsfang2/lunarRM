# 实验执行清单

当前交付为可训练但未正式训练的项目。下面前四步可以在不执行正式优化的情况下完成；
正式训练须在用户后续要求启动时进行。保留官方 train/val 划分，原始数据目录只读。

1. 核对环境与配置：安装 README 所列依赖，用 `scripts/check_config.py --dry-run` 检查
   七个方法及九个消融；固定相同 epoch、seed、batch、损失、增强、目标模式及选模规则。
   默认 20 epoch 只是初始预算。R0/G0 仅输出头不同；R2 特征消融仅移除指定特征，
   MAE/MSE 与 alpha 消融只改指定项。多 seed 不能挑最优 seed 替代平均与离散度。
2. 运行 `scripts/audit_data.py` 全量只读审计，保存 JSON。核对 train 每频 25,720 项、
   val 每频 2,330 项，地形分别 6,120/680；terrain 隔离不等于父地形隔离。
   数据同时有 DEM、Tx/索引、双频路径损耗与 packed 有效 mask。分辨率 256×256、1 m/pixel，
   频率 415/5775 MHz，原始 PL 单位 dB。mask 非 PL==145，也非几何 LoS。
3. 运行默认合成测试、可选 `real_data` 接口测试和无梯度前向。记录设备与限制；
   CPU 合成优化仅验证梯度/恢复。没有 GPU 时，GPU/AMP 保持“未验证”。
4. 用 `scripts/summarize.py --template` 生成待运行表；固定 `docs/examples-val.json`
   以及诊断覆盖率边界，测量固定样本索引、设备、线程、精度、warmup/repeats。
   随机权重测速只检查流程，不能形成方法精度或加速结论。
5. 收到正式训练要求后：先小规模训练与 epoch 边界恢复检查，再在同预算下运行
   A0（仅解析评价）、N0、R0、G0，然后 R1/R2、消融与 N1。每次使用独立运行目录。
   如未收敛，统一调整相关对照的预算并记录理由。关键方法进一步运行固定的多个 seed。
6. 从 best checkpoint 独立重算完整 raw dB val；确认 4,660 项、无重复/遗漏、两频有效像素数。
   RMSE 是总 SSE/总有效像素开根号，MAE 是总 SAE/总有效像素，双频等权 RMSE 为本地选模量。
   公开 val 用于选模，结果不能称为独立测试性能。
7. 同一 checkpoint、固定样例与分组协议输出 diagnostics；同设备测 feature/network/uncached
   end-to-end/cache lanes，保留 P95、内存、未覆盖算子和磁盘/传输边界。
8. 汇总各 JSON，检查 differences_from_first，再人工确认差异是否为指定实验因子。
   汇总不自动连接精度与时延，不自动汇总 seed；需要时根据 checkpoint SHA256 对应来源。
   原始报告与汇总 JSON 是完整证据，CSV 仅为快览。报告失败/无增益/目标未达情况。

## 待运行表字段

`docs/results-template.json` 含全部实验/消融，每行 `status=not_run`。
`config_path` 指向配置；运行后从报告保留完整 `config`，不凭表名推断真实超参数。
未执行的字段全部为 null，不用 0 表示未测量。

| 字段 | 填写依据 |
| --- | --- |
| checkpoint / data_identity / code_identity | 权重 SHA256、索引/官方加载器 SHA256、大数组版本信息及代码摘要；A0 权重可为空 |
| metrics | raw dB 双频 RMSE/MAE、SSE/SAE、有效像素、项数和 full/subset 范围 |
| diagnostics | LoS/NLoS、single/multi、固定覆盖率桶、地形等权及空组计数 |
| examples_manifest | 同一固定样例清单及 SHA256；图像和 NPZ 路径保留在诊断目录 |
| benchmark | timings 的中位数/P95、parameters、部分 complexity、hardware、CPU/GPU memory、boundaries、预热与重复数 |

D0/D1、跨 Tx/跨频率、大型外部模型、新 test 仅为后续可选研究，不是当前运行依赖。
本数据 GT 未包含绕射；未来引入绕射近似不能预先声称比 RT 更准确。
