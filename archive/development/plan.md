# LunarRM 物理引导轻量化预测项目实施计划

> 输入：[goal.md](goal.md)；研究背景：[draft.md](draft.md)。生成日期：2026-10-13。
> 本计划采用 humanize-gen-plan 模板，正文统一为中文，保留模板标题、标识符和路由标签。
> 状态：实施计划，未实施新增模块，未启动训练。当前交付边界是代码、配置与正确性验证准备完成；正式数据训练和性能实验在用户后续要求启动时执行。

## Goal Description

将现有普通 U-Net 原型扩展为可复现的 LunarRM 实验项目：复用官方数据接口与固定划分，实现三维 FSPL、地形几何、LoS/Fresnel 特征及轻量 Tiny U-Net，支持直接预测和物理残差预测，在原始 dB 标签上进行 mask 评价，并提供公平对照、消融、效率测量和结果导出入口。

主要闭环为 A0（FSPL）、N0（纯 Tiny U-Net）、R0（FSPL 残差）和 G0（与 R0 同输入的直接预测）。第二阶段在相同框架中实现 R1（几何特征）、R2（LoS/Fresnel）及特征、损失消融。N1 保留现有普通 U-Net。D0/D1 绕射近似、跨 Tx/跨频率和大型论文模型为可选扩展，不阻塞主体代码交付。

### 事实、研究目标与硬性约束

- 硬性正确性约束：原始数据只读；官方固定 train/val；正确 hm_row/TX/mask 对齐；无标签输入泄漏；以原始 dB GT 评价；双频分别统计；完整验证不漏样本；可追溯配置和 checkpoint。
- 研究目标：Tiny 模型初始参数预算 0.1M–0.5M、每图端到端耗时低于 500 ms、相对基线改善精度。goal.md 未给精度提升的数值门槛，也没有指定测速设备，不能凭空补充门槛。
- 用户已确认 DEC-1：三项均按实验目标处理，实测报告，不作为代码交付的硬性门槛。即使目标未达到，也须如实记录和解释。
- 本地官方材料提供参考加载器，不提供本项目 Tiny 网络方案。当前 GT 未包含绕射，D0/D1 必须标注为探索性对照。
- val 标签公开且用于选模；当前 Kaggle 子集只做格式检查。无 test 文件时不得宣称测试结果，也不得把未见 tile 验证称为父地形隔离验证。

### 现有代码差距

| 现有内容 | 保留价值 | 待补充或修正 |
| --- | --- | --- |
| `model.py` 普通卷积 U-Net | 作为 N1 容量对照 | 新增深度可分离 Tiny 主干和残差组合头 |
| `train.py` 动态加载官方模块 | 官方读取语义 | 封装可在多 worker 下使用的稳定适配层，独立原始 GT |
| 当前归一化目标评价 | 接口检查可用 | 改为原始 dB GT，显式记录截断差异 |
| 当前 batch masked MSE | 基础训练流程 | MAE/MSE、逐频段等权、NLoS 加权、空 mask 策略 |
| 当前验证与部分 batch 限制 | 已尝试覆盖两频段 | 固定索引子集、完整统计、float64 累计和分组输出 |
| 当前 checkpoint | 模型/优化器/调度器/AMP | 随机数、采样状态、配置兼容验证、原子保存 |
| `check_setup.py` | 无梯度接口检查 | 覆盖 train/val 与两个频段边界，检查增强/物理组合 |
| `run_train.sh` | 单一训练入口 | 显式配置选择、设备检查与独立输出目录 |

## Acceptance Criteria

每项按 TDD 先建立具有实际故障检测能力的最小测试，再实现。下文 Negative Tests 指错误输入或错误实现应被拒绝或被测试识别，不是让正常测试套件保留失败。默认测试使用微型合成夹具，不启动 LunarRM 训练。

- AC-1：建立配置和实验注册机制，明确模型、输入特征、损失、频段、预算、设备和输出位置。
  - Positive Tests (expected to PASS)：A0/N0/N1/R0/G0/R1/R2 可解析并打印完整解析后配置；同输入对照 R0/G0 具有完全相同的通道顺序和主干宽度；参数覆盖顺序确定；`--help`、配置检查与 dry-run 不更新权重。
  - Negative Tests (expected to FAIL)：未知实验名、负数 epoch、非法宽度/GroupNorm 组合、重复特征或冲突频段设置被拒绝；未指定 resume 时禁止向已有训练输出目录混写；请求 CUDA 而不可用时明确失败，不能静默开始 CPU 长训练。

- AC-2：提供只读数据审计，检查现有文件、官方索引与固定划分。
  - Positive Tests：核验数组头形状/dtype、25,720/2,330 条索引、6,120/680 地形、双频 item 数、terrain 无交集、hm_row 与 Tx 坐标；分块统计有限值、mask 覆盖率和超出归一化边界的有效标签；报告输入文件版本或校验摘要和扫描范围。
  - Negative Tests：夹具中的越界 hm_row、重复 sample_id、跨 split 地形重叠、错误 packed mask 长度和标签行数不匹配均被检测；部分扫描不能输出“全数据核验通过”；审计不得写回数据目录。

- AC-3：数据适配与官方接口一致，同时暴露原始物理量和可追溯样本身份。
  - Positive Tests：逐项对比官方 `normalize=False, augment=False` 的 DEM、TX、PL、mask；检查 train/val 两频段首尾与频段切换点；返回明确的 sample_id、terrain_id、band、tx_row/col、raw DEM、raw GT；用 mmap 访问合并数组；微型夹具验证多 worker 读取。
  - Negative Tests：不能按标签行号直接索引去重 DEM；不能对频段同名 sample 混用标签；不能按 PL==145 构造 mask；mask 或 GT 不得被注册为输入特征；缺少 test 文件时给出明确错误。

- AC-4：统一归一化与增强，保证物理单位、原始 GT 和空间对齐。
  - Positive Tests：高度 `[0,496]` 与公共 PL `[20,228]` 的参考归一化匹配官方；原始评价 GT 独立保留；同一随机变换应用到 DEM、Tx、PL、mask 和特征；方向坐标按旋转/翻转变换或重新计算；固定样本同 seed 输出一致。
  - Negative Tests：使用逐频段边界反归一化、用归一化 DEM 算距离、把超界 GT 截断后评价、仅旋转输入未旋转标签、用 val 拟合统计量等错误均有针对性断言或回归测试。

- AC-5：实现单位一致、数值稳定的三维距离与 FSPL。
  - Positive Tests：Tx 高度 H(Tx)+3 m，Rx 高度 H+1 m；Tx 同像素 d=2 m；水平相距 3 m 且相同地面高程时 d=sqrt(13) m；FSPL 与独立公式计算相符；固定几何的频段差为 `20*log10(5775/415)`；全图高程平移不改变结果。CPU float64 解析测试以 1e-6 dB 绝对误差为初始容差，float32 以 1e-4 dB，需记录数值类型。
  - Negative Tests：非正频率、非法网格间距、非有限 DEM、越界 TX 被拒绝；二维距离、重复加入高度偏置或额外 flipud 的实现应被测试检出。

- AC-6：实现几何、LoS 与 Fresnel 特征，明确边界约定并支持受控缓存。
  - Positive Tests：相对地面高度、相对坐标、坡度和 Laplacian 曲率代理有单位说明；平地 LoS、已知山脊遮挡、Tx 同水平位置与边界 TX 用例正确；剖面默认双线性插值、水平间隔≤1 m、排除端点；净空正负定义固定，频率变化影响 Fresnel 而不影响纯几何 LoS；0.5/1/2 m 可配置。
  - Positive Tests：分块处理接收像素，缓存键含数据标识、算法版本、参数、dtype、terrain/sample/band 中实际依赖的字段；无缓存与缓存结果在指定容差内相同；更改频率或算法版本不会命中过期缓存。
  - Negative Tests：零水平路径除零、采样步长≤0、缓存不完整、缓存版本不符被处理或拒绝；有效 mask 不得代替几何 LoS；曲率代理不能以严格曲面曲率命名；所有像素与剖面一次广播导致的无界内存分配应通过小块测试和内存预算检查防止。

- AC-7：实现纯 Tiny U-Net、直接预测与残差预测，以及现有 N1 兼容入口。
  - Positive Tests：各必需配置输出 `(B,1,256,256)`；Tiny 使用深度可分离卷积、多尺度 skip 与合法 GroupNorm；统计每配置真实参数量；R0/G0 共享同一特征清单；残差输出为零时恰好返回 FSPL，正负残差均可表达；N1 仍可独立前向。
  - Negative Tests：残差经过 sigmoid/ReLU 被限制为正、误加 20 dB 截距、丢失 FSPL 或重复反归一化，应被组合头测试检测；输入通道不匹配拒绝加载；参数目标偏离须在报告中显式标记，不能把 N1 命名为满足 Tiny 预算。

- AC-8：实现 masked MAE/MSE、双频等权和可选 NLoS 加权。
  - Positive Tests：使用手算小张量验证两种损失；各频段先按有效像素归一再等权；默认双频 sampler 提供平衡项数、记录最后 batch 策略；NLoS 项仅作用于有效 NLoS；无有效像素的项被跳过并计数；损失归约 FP32；合成梯度测试检查无效像素梯度为零。
  - Negative Tests：构造两频段有效像素数量明显不同的样例，识别全 batch 像素平均代替等频段权重的错误；缺席频段不能以零损失拉低均值；空 mask 不得产生 NaN 或伪造有效更新；无效位置的大误差不能影响损失。

- AC-9：训练框架可追溯、可恢复，配置检查与训练执行显式分离。
  - Positive Tests：支持 AdamW、余弦调度、seed、设备与 AMP；日志含有效像素、跳过项、学习率、更新数；checkpoint 原子保存模型、优化器、调度器、AMP、随机数、sampler/epoch、最佳指标与完整配置；CPU 合成夹具在 epoch 边界恢复后与连续运行匹配；GPU 路径无设备时明确标记未验证。
  - Positive Tests：使用按 seed/epoch/sample 确定的增强或等价方案解决 worker RNG 重建；声明恢复粒度，默认仅承诺 epoch 边界恢复。每次真实训练只由显式训练入口触发。
  - Negative Tests：损坏 checkpoint、数据版本/特征顺序/模型结构不兼容、恢复后意外重置调度器必须报错；非有限损失被拦截；未实现的 batch 中途恢复不能宣称精确续训；不得在导入、审计或前向检查时调用参数更新。

- AC-10：独立评价入口按原始 dB GT 正确累计完整验证指标。
  - Positive Tests：逐频段累计 SSE、SAE、有效像素数（float64），最终计算 RMSE/MAE 和双频等权 RMSE；改变 batch size 不改变统计口径；完整验证覆盖全部 4,660 项且无重复；限制模式直接建立每频段固定索引子集；checkpoint 独立重算结果与训练内验证一致。
  - Negative Tests：用含不同误差和不同像素数的 batch 检测“平均 batch RMSE”错误；截断 GT、漏掉 58、空频段仍给出正式双频分数、部分验证选为完整验证最优模型均被拒绝；无有效像素分组返回空值与计数而非零误差。

- AC-11：提供分组评价、可视化和消融配置，支撑公平归因。
  - Positive Tests：输出 LoS/NLoS、single/multi、mask 覆盖率组、地形等权诊断及计数；分组边界固定在配置中，统计拟合仅用 train；提供 R2 去 LoS/Fresnel/曲率、MAE/MSE、alpha=0/0.1/0.5 配置；固定相同示例展示 DEM/Tx、GT、预测、mask、误差和物理基线。
  - Negative Tests：NLoS 与有效 mask 混用、空组记零、不同模型挑选不同优质样例、把研究分组说成官方类别，应由测试或报告审查检出；对照配置除指定消融因子外的无意差异必须标记。

- AC-12：提供复杂度与端到端性能测量入口，并准确报告测量条件。
  - Positive Tests：batch=1，记录硬件、线程、精度、预热与重复数；输出特征生成、网络前向、无特征缓存端到端的中位数/P95、参数量及峰值内存；GPU 同步；明确 I/O 与传输边界，缓存吞吐单列；MAC/FLOP 工具覆盖不足时报告未覆盖算子，不伪造完整数值。
  - Negative Tests：未预热或未同步 GPU 的结果不可标为正式测速；不能把缓存命中耗时标为完整物理推理；不能在没有同设备 RT 时间时生成加速倍数；超过性能目标须如实显示。

- AC-13：完成可运行配置、命令说明与实验执行清单，保留原始文档和数据。
  - Positive Tests：README 给出审计、无梯度检查、显式训练、恢复、独立评估、测速和汇总用法；必需实验配置全部可 dry-run；每个输出记录代码/数据/官方加载器身份与 resolved config；生成不含虚构数值的结果表模板；声明当前尚无训练结果。
  - Negative Tests：文档命令不能指向缺失模块；未来 test、真实 DEM 或大型模型下载不得成为本地必需步骤；Kaggle 格式检查分数不能填入独立测试性能栏；未经后续训练指令不得启动真实数据优化。

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

完成全部 AC-1 至 AC-13，形成模块化工程、数据审计、A0/N0/N1/R0/G0/R1/R2 配置、消融配置、训练恢复、独立评估、特征缓存、效率统计与汇总工具。可在不改变核心语义的前提下追加 D0/D1、预测地图导出与当前 Kaggle 子集 CSV 格式校验，但这些不成为主工程依赖。真实训练结果和跨场景研究执行单列为后续工作。

### Lower Bound (Minimum Acceptable Scope)

满足全部 AC-1 至 AC-13 的功能语义，可以采用单机单进程训练、简单 JSON 配置、仅 epoch 边界续训和按接收像素分块的 CPU 物理特征计算。必须提供 R1/R2 与消融能力，不能只交付普通 U-Net 原型。无需分布式训练、交互界面、远程平台、自动超参搜索或完成全部实验训练。

### Allowed Choices

- 可用：现有 Python/PyTorch/NumPy；JSON 配置；pytest；Matplotlib 可视化；必要时使用注明版本和计数约定的轻量复杂度工具。
- 可用：官方加载器的只读包装或经官方接口逐项验证的等价读取；单独保存项目特征；纯 CPU 正确性测试与可选 GPU 验证。
- 固定：官方 train/val 索引、原始 dB GT 评价、米制物理计算、有效 mask 与 LoS 分离、双频各自统计、完整 256×256 输出。
- 不采用：修改官方数据或加载器、随机按样本重新切分主验证、将 GT/mask 用作模型输入、为主项目重新部署 Sionna RT、无依据地声称官方推荐模型或已获性能提升。
- 写入边界：工程与配置仅在 `/home/Users_Work_Space/zsfang/lunarRM`；测试临时文件放项目测试临时目录或 `/tmp`。数据根目录只读；后续大规模缓存迁移到其他磁盘需显式选择路径与权限。
- 保留 `draft.md`、`goal.md` 原文；避免破坏既有代码，重构时保留 N1 及旧入口兼容说明，不执行仓库初始化、提交或外部发布。

## Feasibility Hints and Suggestions

> 本节为建议实现路径；验收以 AC 和边界为准，不强制文件拆分完全相同。

### Conceptual Approach

建议结构：

```text
lunarRM/
  goal.md, draft.md, plan.md
  model.py, train.py, check_setup.py, run_train.sh  # 保留或改为兼容入口
  lunar_rm/
    config.py
    data.py                 # 官方适配、原始 GT、身份信息
    sampling.py             # 双频均衡和可复现采样
    features.py             # FSPL、几何、剖面特征
    cache.py
    models.py               # Tiny、普通模型、dB 输出组合
    losses.py, metrics.py
    engine.py, checkpoint.py
  configs/                  # 方法与消融的显式配置
  scripts/                  # 审计、评估、测速、汇总入口
  tests/                    # 合成与接口回归夹具
  runs/, reports/, cache/    # 运行时生成，不写入原始数据目录
```

数据契约建议包含 `features`、`target_db`、`valid_mask`、`baseline_db`、`geometric_los`、`sample_id`、`terrain_id`、`band`。对无须 LoS 的训练配置可以不生成它；分区域评估时统一按固定剖面协议补算。

纯输入通道从 raw DEM 构造为官方归一化高度与 Tx/band；所有对照训练默认保留 raw GT，用公共仿射尺度训练时不截断目标，从而与 raw dB 损失等价。若需复现实例加载器的截断目标模式，作为显式兼容配置，所有比较组统一使用并报告截断数量。

直接模型可以通过 `20+208*z` 输出 dB；残差模型通过 `baseline_db+208*r` 输出 dB。统一在 dB 空间建立损失，再按固定 208 或 208² 缩放 MAE/MSE，以保持优化数值稳定；日志同时记录未缩放物理误差，不能把归一化 loss 标成 dB。

LoS/Fresnel 初版先求正确性再优化：按接收像素分块生成剖面，端点使用天线高度，内部用双线性 DEM 插值。可定义净空为直射线高于地形的高度，Fresnel 指标为内部采样点最小净空/rF，正值表示净空；无水平剖面时使用显式约定和有限值。投影距离与三维距离的几何近似需在代码文档中说明。

训练增强推荐由 `(seed, epoch, sample_id, band)` 确定，减少 persistent worker 的隐式 RNG 对续训的影响。模型和数据身份写入 checkpoint；新运行不覆盖已有产物。CPU 续训夹具使用确定性算子和固定线程；GPU 仅承诺注明环境下的复现程度。

### Relevant References

- `goal.md` 第 2–3 节：数据、mask、归一化、官方接口。
- `goal.md` 第 4–6 节：实验矩阵、损失与训练设置、评价和阶段。
- `model.py`：现有普通 U-Net，作为 N1；不能当作 Tiny 实现。
- `train.py`：现有训练、验证、checkpoint 逻辑；需修正目标评价与恢复完整性。
- `check_setup.py`：当前只检查 val 前两个 415 样本，需扩大接口覆盖。
- `/home/DataDisk/zsfang/lunarRM/lunar_dataset.py`：官方读取、频段顺序与归一化行为。
- 数据目录中的 `README.md`、`metadata.json`、`manifest.json`：传播机制、单位、评价和 Kaggle 子集用途。

## Dependencies and Sequence

### Milestones

1. M1 配置与数据契约（AC-1 至 AC-4）
   - 先建立微型官方格式夹具、配置校验和数据审计，再实现数据适配与增强。
   - 通过接口比较确认 raw GT、双频身份、索引与单位；不执行训练。
2. M2 物理模块与模型（AC-5 至 AC-7）
   - 先实现解析 FSPL 与 Tiny 输出契约，再补充几何和剖面特征。
   - 以合成平地、山脊、增强等变性与缓存一致性验证；真实数据仅前向取样。
3. M3 训练与评价工程（AC-8 至 AC-10）
   - 损失和指标使用独立手算夹具测试，然后接入训练循环与恢复。
   - 恢复测试采用极小合成数据，不构成 LunarRM 性能训练；当前计划生成阶段不执行此测试。
4. M4 实验准备交付（AC-11 至 AC-13）
   - 完成消融、分组统计、测速、命令说明及结果模板。
   - 核验必需配置 dry-run 与真实双频无梯度前向；交付“可训练但未正式训练”的项目。
5. M5 后续正式实验（不属于当前代码准备的完成条件）
   - 用户要求启动后先做小规模训练与恢复检查，再按相同预算运行 A0/N0/R0/G0。
   - 然后 R1/R2 与消融、关键方法三 seed；资源允许再运行 N1/D0/D1 和专项泛化。
   - 未收敛时统一增加对照组预算；报告失败或无增益结果，不以筛选随机种子替代结论。

依赖顺序：数据契约先于特征与损失；FSPL 先于残差组合；正确的原始 GT 指标先于 best checkpoint 选择；版本化特征先于缓存；配置和评价稳定后才进入正式对照实验。

## Task Breakdown

每项恰有一个模板路由标签。`coding` 表示实现类工作，`analyze` 表示分析/审查类工作；保留模板的 Claude/Codex 路由含义供未来 humanize 工作流使用，本次由 Codex 编制计划，不代表已经调用 Claude、ask-codex 或子代理。

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
| --- | --- | --- | --- | --- |
| task01 | 固化现有行为、差异清单与 raw GT 处理协议 | AC-1, AC-3, AC-4 | analyze | - |
| task02 | 配置模式、方法注册、路径与设备校验 | AC-1 | coding | task01 |
| task03 | 官方格式微型夹具与只读数据审计 | AC-2 | coding | task02 |
| task04 | 官方适配、身份字段和 mmap 多 worker 读取 | AC-3 | coding | task03 |
| task05 | 公共归一化、原始 GT 和确定性同步增强 | AC-4 | coding | task04 |
| task06 | 三维距离、FSPL 与解析测试 | AC-5 | coding | task04 |
| task07 | 几何、分块 LoS/Fresnel 与受控缓存 | AC-6 | coding | task05, task06 |
| task08 | Tiny 主干、直接/残差头和 N1 兼容 | AC-7 | coding | task02, task06 |
| task09 | 均衡 sampler、MAE/MSE 与 NLoS 损失 | AC-8 | coding | task05, task07 |
| task10 | 独立 dB 指标与完整/固定子集评价 | AC-10 | coding | task04, task08 |
| task11 | 训练循环、AMP、日志和 epoch 边界恢复 | AC-9 | coding | task08, task09, task10 |
| task12 | 分组指标、固定样例可视化与消融配置 | AC-11 | coding | task07, task10, task11 |
| task13 | 端到端测速、复杂度与内存报告 | AC-12 | coding | task07, task08 |
| task14 | 命令入口、README、结果模板和运行清单 | AC-13 | coding | task11, task12, task13 |
| task15 | 审查 AC 覆盖、公平性、无泄漏与未训练状态 | AC-1, AC-2, AC-3, AC-4, AC-5, AC-6, AC-7, AC-8, AC-9, AC-10, AC-11, AC-12, AC-13 | analyze | task14 |

任务表只覆盖代码准备。M5 的真实训练不是隐含 coding 任务，需在执行阶段按用户启动指令安排。

## Claude-Codex Deliberation

### Agreements

- 本轮未进行 Claude–Codex 双模型讨论，不能记录虚构共识。当前方案由 Codex 根据 goal.md、官方接口和现有代码制定。
- 已有用户约束：保留固定数据语义，准备工程与配置，暂不启动训练。

### Resolved Disagreements

- 无实际双模型争议记录。
- 文档与原型差异的处理：已有 MSE 保留为可选损失；新主配置按 goal.md 使用 MAE；Tiny 与现有普通 U-Net 分别命名；原始 GT 与官方参考截断模式明确分开。这些属于本轮技术整理，不冒充讨论结果。

### Convergence Status

- Final Status: `partially_converged`
- 原因：单代理已完成结构与依赖审查，用户已确认 DEC-1，无未解决的用户决策；保留此状态仅表示未进行模板所列的双模型讨论，不表示存在实施阻塞。

## Pending User Decisions

- DEC-1：参数量、时延与精度目标的验收性质。
  - Claude Position：未参与，不适用。
  - Codex Position：建议作为实验目标，代码交付硬性要求为正确测量、完整报告和可复现实验；Tiny 尽量达到 0.1M–0.5M，超出必须说明。
  - Tradeoff Summary：硬性参数上限可直接约束架构；时延需要指定硬件；精度提升只能实测。将三项全部作为交付硬门槛会让正确但结果不支持假设的实验无法完成验收。
  - Decision Status: `RESOLVED`。用户于本轮明确选择“按实验目标处理（推荐）”：参数预算、时延和精度提升实测报告，不作为代码交付硬性门槛。

当前无待用户决策事项。本节保留 DEC-1 作为决策记录。

其他选择已由 goal.md 或现有指令确定：中文正文、现有 train/val、先准备代码、暂不训练。未来启用真实 test、跨频率、含绕射新数据等扩展时另建实验协议。

## Implementation Notes

### Code Style Requirements

- 实现代码、注释和公开 API 不使用 `AC-*`、Milestone、Phase、task ID 等计划流程标识，采用数据和传播领域命名；实验标识 A0/N0/R0 等可以作为配置标签。
- 路径使用配置和 Path 对象，避免将单机绝对路径散布各模块；默认数据根可保留当前路径。
- 函数接口明确单位（m、Hz、dB）、tensor 维度、dtype、归一化和 mask 语义。输入特征和监督字段分离。
- 测试不要求下载新数据或 GPU；真实数据检查与 GPU 测试单独标记，缺少条件明确报告跳过。
- 未获得训练结果前，结果表使用空值/待运行状态。任何准确率或速度的改善均不得在代码准备阶段预先宣称。
- 按模板检查计划完整性、AC–任务映射与相互依赖；正文使用中文。当前 humanize 配置未启用替代语言文件，不生成额外翻译版本。
