# 数据加载

默认目录在 [paths.py](paths.py) 的 `DEFAULT_DATA_ROOT` 中设置；也可用训练参数 `--data-root` 覆盖。
数据位于仓库外，无需复制或重新预处理。所有数组以只读方式加载。

支持以下官方合并格式（train 和 val 各一套）：

```text
train_index.csv
train_hm.npy
train_pl415.npy
train_pl58.npy
train_mask415.npy
train_mask58.npy
```

也支持 `train/index.csv`、`train/hm.npy` 等分目录布局。
索引通过 `hm_row` 关联去重地形；mask 为打包布尔有效区域，不可用 `PL == 145` 替代。
数组需为已解压的 NPY；加载器不直接读取 ZIP。

数据包含 DEM、发射机索引、415/5775 MHz 路径损耗和有效 mask。
每图 256×256、1 m/pixel；train 每频 25,720 项，val 每频 2,330 项。
内部频段键为 `415` 和 `58`。当前没有独立 test。

[loader.py](loader.py) 负责读取与特征组装，自动按模型配置计算所需输入；
[features.py](features.py) 为固定物理/几何计算，无可训练参数。
训练和验证共用此流程，不需要单独运行特征脚本。GT 与有效 mask 仅参与损失/评价，不作为网络输入。
[transforms.py](transforms.py) 管理归一化与训练增强，验证不增强。
可通过 `--cache-dir cache/physics` 启用固定特征缓存。
