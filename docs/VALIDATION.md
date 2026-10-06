# 验证记录

日期：2026-10-06。使用合成张量 / 合成 NIfTI，不使用或发布临床数据。

## 实际环境

- 节点：`bme-npu01`，Ascend 910B3；分配独占资源后，只使用单卡与双卡做短检查。
- 已有 Conda 环境：`vae`，Python 3.11。
- PyTorch `2.6.0+cpu` + torch_npu `2.6.0`，torchvision `0.21.0`。
- Accelerate `1.13.0`，MONAI `1.5.2`，nibabel `5.4.2`，Kornia `0.8.2`，pytorch-msssim `1.0.0`。
- 节点默认 CANN 路径指向 `8.0.RC3`；本次没有安装、升级或替换环境组件。
- 使用计算节点已配置的登录 shell，设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，由项目显式导入 torch_npu。以下通过结果仅代表这一实际环境和所列测试范围，不构成通用版本兼容矩阵。

## 已通过

| 检查 | 范围 | 结果 |
| --- | --- | --- |
| Python 语法 / shell 入口 | 修改后的 Python 源码、通用训练脚本 | 通过 |
| 单 NPU CUT | ResNet-6、1 通道、64×64、ngf/ndf=4、每配置 2 个训练步 | 3 个配置通过 |
| 双 NPU CUT | torchrun 2 进程，每个 rank 使用不同输入；检查 G/D/F 每个参数同步 | 每个 rank 的 3 个配置通过，共 6 个 PASS |
| 特征采样配置 | `mlp_sample + NCE=1`、`sample + NCE=1`、`mlp_sample + NCE=0`；开启 nce_idt | 损失有限、G/D 及适用时的 F 参数更新 |
| 权重往返 | 保存 G/D/F，重新构造模型并加载；动态 F 初始化后恢复 | state_dict 逐项完全相同 |
| NIfTI 端到端 | 合成 trainA/trainB 各 2 个体数据；实际 `train.py` 完成 1 epoch（4 个切片样本），实际 `predict_monai.py` 导出 2 个测试体数据 | 通过 |
| NIfTI 几何与掩膜 | 输出 256×256×2、spacing 1×1×2；生成图/预处理源图 affine 一致，有限值与 [0,1] 范围，源图为零处输出为零 | 通过 |
| DataLoader worker | num_threads=1、seed_offset=1、连续两个 epoch；worker 返回值与主进程同 epoch 样本对照 | 通过 |

运行命令（在已分配的计算节点、已配置的 vae 登录环境中）：

```bash
unset ACCELERATE_USE_CPU
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 CUT_SMOKE_DEVICE=npu OMP_NUM_THREADS=1
python tests/smoke_cut.py
python -m torch.distributed.run --standalone --nproc_per_node=2 tests/smoke_cut.py
python tests/smoke_monai.py
```

两次张量检查分别输出 3 / 6 个配置 PASS，端到端检查输出 `PASS synthetic NIfTI training, prediction, affine/mask and worker epochs (npu)`。测试在临时目录创建数据、权重和预测文件，退出时清理。

## 未验证 / 已知限制

- 登录节点的 vae CPU 检查遇到缺失 CANN 动态库；关闭设备插件自动加载后仍出现 `Illegal instruction`，因此不列为 CPU 通过。没有据此修改系统或 Python 环境。
- 不覆盖真实数据收敛、质量指标、速度提升、默认宽度的完整训练、8 卡、混合精度、所有生成器/判别器组合、SB 或完整 3D 模型。
- 保存检查只覆盖网络权重，不包含完整优化器/随机状态恢复。
- 测试范围较小，不能代替正式实验。新增损失的医学意义和效果需单独消融。
