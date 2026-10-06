#!/usr/bin/env python3
"""测试NPU分布式设备分配"""

import os
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

# NPU设置
if hasattr(torch, 'npu'):
    import torch_npu
    torch.npu.set_compile_mode(jit_compile=False)

# 在Accelerator初始化前设置设备（如果已知）
if 'LOCAL_RANK' in os.environ:
    local_rank = int(os.environ['LOCAL_RANK'])
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.set_device(local_rank)
        print(f"Pre-init: Set NPU device to {local_rank}")

# 初始化Accelerate
accelerator = Accelerator()
set_seed(42)

rank = accelerator.process_index
world_size = accelerator.num_processes
local_rank = accelerator.local_process_index
device = accelerator.device

# 显式设置NPU设备
if hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.set_device(local_rank)
    current_device = torch.npu.current_device()
    device_name = torch.npu.get_device_name(current_device)
else:
    current_device = "N/A"
    device_name = "N/A"

print(f"[Rank {rank}/{world_size}] Local rank: {local_rank}, "
      f"Device: {device}, NPU current: {current_device}, "
      f"NPU name: {device_name}, "
      f"Main: {accelerator.is_main_process}")

# 创建一个简单的模型并测试设备分配
model = torch.nn.Linear(10, 10)
model = model.to(device)

print(f"[Rank {rank}] Model device before prepare: {next(model.parameters()).device}")

# 使用accelerate prepare
model = accelerator.prepare(model)

print(f"[Rank {rank}] Model device after prepare: {next(model.parameters()).device}")

# 测试数据传输
x = torch.randn(4, 10).to(device)
print(f"[Rank {rank}] Input device: {x.device}")

y = model(x)
print(f"[Rank {rank}] Output device: {y.device}")

# 同步所有进程
accelerator.wait_for_everyone()

if accelerator.is_main_process:
    print("\n=== Test completed ===")
