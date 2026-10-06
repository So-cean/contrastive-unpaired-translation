# Ascend / 多进程检查

先在计算节点确认 PyTorch、torchvision、torch_npu、CANN 与驱动匹配，并加载集群规定的 CANN 环境。登录节点的 CPU 检查不能证明 NPU 可运行。

```bash
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__); print(torch.npu.is_available(), torch.npu.device_count())"
```

`train.py` 在检测到 `npu-smi` 和 torch_npu 时使用 `transfer_to_npu` 兼容层。`ACCELERATE_USE_CPU=true` 可跳过这个入口。若导入 torch 就报缺少 `libhccl.so`，应核对计算节点的 CANN 环境；仅运行 CPU 测试时可设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，不要把此开关当作 NPU 修复方案。

先用单卡和 README 的最小 CUT 配置验证，再在已经分配的计算资源中尝试以下设备诊断：

```bash
torchrun --standalone --nproc_per_node=2 test_npu_distributed.py
```

该工具只检查进程、设备、简单网络前向和同步，不验证 CUT 训练收敛。不要手工给每个进程设置相同的 LOCAL_RANK；由启动器提供 rank。多进程训练可使用同一 torchrun 前缀启动 `train.py`，并附加完整数据与模型参数。

遇到挂起时，记录每个 rank 的第一条错误、设备分配、进程数量和版本信息；确保数据足够分配到每个 rank。不要把关闭通信访问控制作为默认修复。原有 HCCL 超时值是否合适应按集群实际配置判断。

`CUT_DETECT_ANOMALY=1` 可开启训练异常检测；`ASCEND_LAUNCH_BLOCKING=1` 只在定位异步错误时临时使用，不能带着它报告正常性能。单/双 NPU 小模型的 FP32 回归已通过；完整多卡训练、bf16 / fp16 尚未验证。

## 本次验证用法

在 `bme-npu01` 已配置的 `vae` 登录 shell 中，本次使用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 并由训练入口显式导入 torch_npu。基础 NPU 运算和单/双进程 CUT 回归通过；其他启动环境曾在依赖导入阶段崩溃，未对根因作通用判断。请保留本机环境配置，并先运行小检查，完整版本与命令见 [验证记录](docs/VALIDATION.md)。
