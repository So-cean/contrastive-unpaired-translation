# Accelerate 迁移状态

训练入口为 `train.py`（CUT）和 `train_sb.py`（实验 SB）。没有额外的 `train_accelerate.py`、`launch_train.sh` 或内置 Accelerate YAML 配置。

- `BaseModel` 不是 `nn.Module`；`prepare_training()` 在动态 F 初始化之后分别包装 G/D/F/E 和优化器。
- CUT / SB 反向传播通过 `BaseModel.backward()` 使用 `accelerator.backward()`。
- DataLoader 自带 DistributedSampler，不再传给 Accelerate 二次切分。
- 动态 F 优化器创建后统一构建学习率调度器。
- CPU 不再被强制设置为 CUDA 设备；仅主进程保存权重，保存函数内不放全员 barrier。
- 双数据流的 seed_offset 传递给数据集及采样器；worker 每个 epoch 重建以接收更新的 epoch。
- continue_train 恢复 G/D（SB 还有 E），F 首次初始化后恢复；不提供完整训练状态恢复。

单/双 NPU CUT 小模型回归通过，但不构成 SB、混合精度或完整多卡训练的结果验收。具体通过的检查见 [docs/VALIDATION.md](docs/VALIDATION.md)。
