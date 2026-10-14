# 模型

日常只需选择下面两个模型之一，然后用同一个训练入口：

| 名称 | 含义 |
| --- | --- |
| `physics_unet` | 完整模型：自动构造物理/地形输入，用轻量 U-Net 预测相对 FSPL 的路径损耗残差 |
| `tiny_unet` | 基础模型：轻量 U-Net 根据高度、发射机位置、频段直接预测路径损耗 |

**只有 U-Net 网络有可训练参数，各模型均一次训练完成。**
固定特征计算和输出组合由数据/模型流程自动处理，不需要单独训练或加载权重。
完整模型使用 11 通道输入，输出 `FSPL + 208 × 网络输出`，单位为 dB。
本方案为项目自定义，不能称为官方推荐 baseline。

[network.py](network.py) 包含轻量 U-Net 和输出组合；[unet.py](unet.py) 是普通 U-Net 容量对照。
[presets.py](presets.py) 定义模型所需特征和输出方式。

## 可选对照

复现完整模型不需要先运行这些实验；需要分析各项贡献时再选用。

| 名称 | 含义 |
| --- | --- |
| `unet` | 普通 U-Net，直接预测 |
| `fspl_residual` | 基础输入加入 FSPL，预测残差 |
| `fspl_direct` | 与上项相同输入，改为直接预测 |
| `geometry_unet` | 残差模型加入相对位置、高差、坡度与曲率代理 |
| `physics_unet_no_los` | 完整模型去掉 LoS 输入 |
| `physics_unet_no_fresnel` | 完整模型去掉 Fresnel 输入 |
| `physics_unet_no_curvature` | 完整模型去掉曲率代理 |
| `physics_unet_direct` | 完整特征，改为直接预测 |
| `physics_unet_mse` | 完整模型改用 MSE 损失 |
| `physics_unet_alpha01` / `physics_unet_alpha05` | NLoS 损失加权系数 0.1 / 0.5 |
| `fspl` | 解析 FSPL，无需训练，只评价 |

`physics_unet_mae` 与 `physics_unet_alpha0` 均等同默认完整模型，无需重复运行。
老版本数字编号仅在读取已有配置/checkpoint 时兼容；新文档与输出使用描述性名称。

```bash
python evaluate.py --experiment fspl --device cpu --output reports/fspl-val.json
```
