# 数据读取与原始 dB 协议

本文件记录现有原型与后续实现之间的接口约定，尚不表示新适配层已实现或通过验收。原始数据和官方加载器只读；正式训练等待后续启动指令。

## 当前行为与差异

| 来源 | 已有行为 | 后续处理 |
| --- | --- | --- |
| `train.py:39` | 动态导入官方类，未注册为稳定可导入模块 | 新适配层需验证 spawn 多 worker 读取，不能仅以 Linux fork 可用证明兼容 |
| 官方 `lunar_dataset.py:197` | DEM 通过 `hm_row` 读取，标签通过索引行和频段读取 | 保留此语义，不能直接以标签行号索引去重 DEM |
| 官方 `lunar_dataset.py:232`、`train.py:136` | 高度和标签归一化均截断到 `[0,1]` | 独立读取原始标签；公共尺度可用于训练，评价禁止使用截断标签 |
| `train.py:50` | 全 batch 有效像素平均 MSE；空 mask 分母钳为 1 | 增加按频段有效像素归一再等权的 MAE/MSE；空项跳过并计数 |
| `train.py:65` | 按频段累计 SSE 和像素数，再平均两频段 RMSE | 保留聚合公式；目标改为原始 dB，张量归约改为 float64，并补 MAE |
| `train.py:72` | 限制模式按频段 batch 计数 | 改为固定每频段样本索引；完整模式核验唯一身份与覆盖率 |
| `train.py:109` | 保存模型、优化器、调度器、AMP 和 epoch | 补随机数、采样状态、数据/配置兼容验证及原子保存 |
| `model.py:8` | 普通卷积 U-Net、线性输出头 | 保留为 N1；另建深度可分离 Tiny，不能将现有模型称为 Tiny |
| `check_setup.py:16` | 只取 val 双频数据集的前两个样本 | 实际只覆盖 415；后续检查 train/val 两频段首尾和切换边界 |

## 样本身份与数组

固定使用官方 train/val。索引行 `i` 的 `sample_id` 和 `terrain_id` 保留字符串前导零；唯一项身份是 `(split, sample_id, band)`。另保留 `index_row`、`hm_row`、`tx_index` 和官方 `group` 字段。`group` 的 single/multi 是发射机数量分组，不是传播 LoS 分类。

合并数组以 `mmap_mode="r"` 访问；取 `hm[hm_row]`、`pl{band}[i]` 和 `mask{band}[i]`。mask 按官方 `np.unpackbits` 默认大端位序展开为 `(256,256)`，转 bool；每行应有 8192 个 uint8 字节。禁止通过标签等于 145 dB 判断有效性。TX 在 `[tx_row, tx_col]` 放置 1，其余为 0。存储数组已经对齐，不额外 `flipud`。

频段顺序为全部 `415` 后全部 `58`，对应 415e6 和 5775e6 Hz；双频条件通道分别为 0 和 1。train 每频段 25720 项，val 每频段 2330 项。缺少 test 文件须明确报错；公开 val 用于选模，不称为独立 test 或父地形隔离验证。

计划适配层每项返回以下字段。默认 collate 后增加 batch 维。

| 字段 | 形状/类型 | 含义 |
| --- | --- | --- |
| `raw_dem_m` | float32 `(1,H,W)` | 米制高度，物理计算使用此值 |
| `tx_map` | float32 `(1,H,W)` | 与 DEM 对齐的一热 TX |
| `target_db` | float32 `(1,H,W)` | 存储标签转 float32，保留原始 dB，不截断 |
| `valid_mask` | bool `(1,H,W)` | 官方监督有效区，仅供损失和评价 |
| `features` | float32 `(C,H,W)` | 严格按配置有序白名单生成，不含 GT 或 mask |
| `baseline_db` | float32 `(1,H,W)` | 按需提供的三维 FSPL |
| `geometric_los` | bool `(1,H,W)` | 按需从 DEM/TX 计算，独立于有效 mask |
| 身份字段 | 字符串或整数 | split、sample_id、terrain_id、band、原始/增强后 TX 坐标 |

读取 mmap 后复制所需样本再变换，禁止原位改写底层存储。`target_db` 独立于供兼容模式使用的截断目标。数据身份包括 metadata、索引和官方源码校验摘要，以及数组头与文件版本信息；不能把仅索引摘要称作全部数组内容校验。

## 归一化、物理量与增强

参考高度通道为 `clip(H/496, 0, 1)`。公共 PL 仿射尺度为 `(PL-20)/208`，两频段共用；默认训练目标不截断。仅显式官方兼容模式允许截断训练目标，且需记录有效标签超界数量；评价始终使用 `target_db`。不得使用逐频段范围反归一化，也不得从 val 拟合统计量。

直接模型输出 `20 + 208*z` dB，残差模型输出 `baseline_db + 208*r` dB。残差头不加 sigmoid/ReLU，不再加 20 dB 截距。损失可除以 208（MAE）或 208²（MSE）用于稳定优化；日志区分缩放损失和物理单位误差。

物理计算使用原始 DEM，网格间距读取米制元数据；Tx 为 `H(Tx)+3 m`，Rx 为 `H+1 m`。三维距离包含高度差，TX 同像素距离为 2 m。路径损耗为正 dB，不附加发射功率。

训练增强由 `(seed, epoch, sample_id, band)` 的稳定摘要确定，不能使用进程随机化的 Python `hash()`。同一旋转/翻转作用于 DEM、TX、原始 GT、mask 和已有标量特征。方向坐标应随变换正确旋转/翻转，或在变换后的 DEM/TX 上重算全部特征；推荐后者。保留原始 TX 与当前 TX 两套坐标以便追溯。val 不增强。worker 必须收到当前 epoch，不能只修改 persistent worker 不可见的主进程对象。

## 原始 dB 评价

每频段以 float64 累计有效像素 SSE、SAE 和计数，最后计算 `sqrt(SSE/count)` 与 `SAE/count`，双频分数为两个频段 RMSE 的算术平均。不得平均 batch RMSE；不得按两频段总像素数加权替代等频段权重。无有效像素返回空值和计数；缺席频段不能产生正式双频分数。

完整 val 应恰好覆盖 4660 个唯一项。调试子集在迭代前按频段固定索引，改变 batch size 不改变样本集合；部分验证必须单列，不能作为完整验证 best checkpoint 的依据。

## 后续回归验证清单

1. 微型官方格式夹具包含共享 DEM 的多 TX、不同频段标签、有效的 145 dB 像素、低于 20/高于 228 的有效 GT、空 mask 和非对称 DEM。
2. 对照官方 `normalize=False, augment=False` 检查 DEM/TX/PL/mask；检查 train/val 两频段首尾与切换位置，多 worker 的 spawn 和单进程结果一致。
3. 非法 hm_row、重复身份、跨 split terrain 重叠、错误 packed 长度、缺失标签与越界 TX 均应被拒绝。
4. 以超界标签识别错误截断评价；以不同误差、不同有效像素数和不同 batch size 验证 SSE/SAE 归约；空频段与完整性计数必须显式处理。
5. 同 seed/epoch/身份增强一致，变换后 TX 与 GT/mask 对齐；非对称 DEM、方向坐标、同像素距离和高程平移测试识别额外 flip 或单位错误。
6. 配置拒绝 GT/mask 输入、重复通道、未知方法及非法宽度；R0/G0 保持相同特征顺序和主干宽度。导入、审计、dry-run 和前向检查不得更新权重。

以上为待实现测试清单。本轮仅检查源码及 CSV：train/val 行数分别 25720/2330，地形数 6120/680，无重复 sample_id、无跨 split terrain 重叠，TX 均在边界内；没有扫描数组内容，也没有训练或性能实验结果。
