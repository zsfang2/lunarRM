# 配置

只保留两个常用配置：[完整模型](physics_unet.json) 与 [基础 U-Net](tiny_unet.json)。
JSON 展开了常用超参数，可直接编辑后运行：

```bash
python train.py --config configs/physics_unet.json -gpu 0
```

不传 `--config` 时不会读取这些文件，而使用内置默认设置。
命令行参数可覆盖 JSON 中的训练设置；模型名称必须与 JSON 的 `experiment` 一致。
其他对照无需新建 JSON，直接使用 `--experiment`，名称见 [模型说明](../models/README.md)。
