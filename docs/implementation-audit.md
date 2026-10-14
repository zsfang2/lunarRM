# 实验准备验收记录

范围是 `plan.md` 的代码准备交付。没有执行正式 LunarRM 优化训练，也没有获得
学习模型的验证精度或独立测试性能。真实数据检查仅包含只读审计、官方接口对照和无梯度前向/测速。
所有优化器与恢复测试使用临时合成夹具；待运行表不填虚构数值。

| 验收项 | 实现与检查证据 | 边界 |
| --- | --- | --- |
| AC-1 配置 | `config.py`、`registry.py`、16 个 JSON；`test_config.py`、`test_delivery.py` 全部 dry-run | 显式 CUDA 不可用时失败；dry-run 不创建运行目录 |
| AC-2 审计 | `audit.py`、`storage.py`、`test_audit.py`；此前全量真实审计通过 | 大数组身份是 header/stat，不是内容全哈希；部分扫描不称全量 |
| AC-3 读取 | `data.py`、`test_data.py`；真实官方边界对照此前 2 项通过 | mmap、hm_row、双频身份；mask 独立于 PL 填充值 |
| AC-4 变换 | `transforms.py`、`test_transforms.py` | 公共尺度；raw GT 独立；增强只用于 train，按 seed/epoch/身份确定 |
| AC-5 FSPL | `features.py`、`test_features.py` | 米制三维距离、3/1 m AGL；float32/64 解析容差 |
| AC-6 几何与缓存 | `features.py`、`cache.py`、`test_profiles.py`、`test_cache.py` | 双线性内部剖面、分块预算、缓存身份；曲率是 Laplacian 代理 |
| AC-7 模型 | `models.py`、`model.py`、`test_models.py` | R0/G0 同输入同主干；残差可正可负，零残差还原 FSPL；N1 不称 Tiny |
| AC-8 损失 | `losses.py`、`sampling.py`、`test_losses.py` | 每频有效像素归约后等权；无效像素无梯度，空 batch 不更新 |
| AC-9 训练恢复 | `engine.py`、`checkpoint.py`、`test_engine.py` | AdamW/余弦/随机状态；CPU 合成恢复等价，仅承诺 epoch 边界；GPU/AMP 未实测 |
| AC-10 评价 | `metrics.py`、`evaluation.py`、`test_metrics.py` | 原始 dB、float64 SSE/SAE、唯一完整覆盖；子集不选 best，空频为 null |
| AC-11 诊断 | `reporting.py`、`scripts/diagnose.py`、`test_reporting.py` | 几何 LoS 与 mask 交集、固定分组与样例，R2 消融仅指定差异 |
| AC-12 测速 | `benchmark.py`、`scripts/benchmark.py`、`test_benchmark.py` | label-free、batch=1、同步、预热、各通道；Conv2d 部分计数，单样本和 RSS 范围明示 |
| AC-13 交付 | README、执行清单、空模板、兼容入口、`results.py`、`test_delivery.py` | 无正式训练结果；汇总保留来源/范围/权重身份，不排名或静默合并 |

上述模块均在 `lunar_rm/`，测试均在 `tests/`。默认 pytest 排除 `real_data`；
真实数据接口测试单独可选。旧文档 `draft.md`、`goal.md` 和受追踪计划保持原文。

## 公平性与泄漏检查

模型只接收由 DEM、Tx、频率构造的输入，以及残差所需 FSPL。标签与 valid mask 是监督/评价，
不在特征注册表。`build_inputs` 不接收 GT/mask，测试改变磁盘标签仍得到同样特征。
LoS 是几何算法输出，不从有效 mask 推断。统计常数固定，不从 val 拟合。

R0/G0 的输入顺序与宽度相同，输出头为唯一实验因子；R2 消融的特征/损失/alpha 差异有测试。
正式运行仍需对照完整 resolved config、训练预算、seed、数据/代码版本；注册表不能保证用户
覆盖参数后的任意两次运行公平。汇总列出差异并保留完整报告，不以实验名自动认定可比。

验证严格使用 raw `target_db`。`official_clipped` 只影响训练目标。公开 val 用于选模，
没有本地 test，不将 Kaggle 格式校验分数列为测试性能。固定 PNG/NPZ 身份由预先清单决定，
不按误差挑样例。LoS/NLoS、覆盖率和地形等权均是研究诊断，不能取代全体 raw dB 指标。

## 当前验证记录

- 本轮 16 个配置逐个 CPU dry-run、8 个脚本 help 检查通过；输出目录不创建。
- 本轮七个方法各两频共 14 次真实 val 无梯度前向通过，形状 `(1,1,256,256)`，
  输出记录配置/代码/数据/官方加载器身份。该检查不报告随机模型精度。
- 汇总已有随机 R0 单样本测速成功，保留 `untrained`、`single_sample`，没有填入空模板。
- 完整默认测试：156 passed、2 real_data deselected（2026-10-14）；包含合成 spawn worker 测试。
- 按计划通过 humanize ask-codex 独立只读审计：逐项确认 AC-1–AC-13 覆盖，无阻塞发现。
  审计另外执行 54 项专项测试、153 项非 spawn/非 real_data 回归（5 deselected），
  8 个 CLI help 和 R0/G0/R2 配置差异检查，全部通过。
  本地原始审计记录位于 `.humanize/skill/2026-10-14_01-28-26-2468232-4d98c170/output.md`
  （忽略的工作流产物，不是仓库运行依赖）；此文档保存可随代码分发的整合结论。
  该咨询是计划的最终分析证据，RLCR 的阶段转换仍由正式 Stop hook 单独执行。

## 限制

GPU/AMP、全体样本时延分布、正式精度和目标改善尚未实测。每图 500 ms 与参数预算是
研究目标，代码交付不预先承诺达到；没有同设备 RT 时间，不生成加速倍数。
外部模型、绕射扩展和新 test 不属于当前准备依赖。
