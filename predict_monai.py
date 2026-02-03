#!/usr/bin/env python3
"""
predict_monai.py
对 train、val、test 三个 phase 依次进行推理，复用 test_monai.py 主流程。
自动统计每个 phase 的数据量，确保全部推理。
"""
import sys
import os
import torch
from pathlib import Path
from options.test_options import TestOptions
from test_monai import MonaiTester, MONAI_AVAILABLE
import importlib
import shutil
if shutil.which("npu-smi") and importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    os.environ['HCCL_EXEC_TIMEOUT'] = '120'  
    os.environ['HCCL_CONNECT_TIMEOUT'] = '120'
    
def count_nii_files(dataroot, phase, direction):
    # 根据 direction 选择主域
    if direction == 'AtoB':
        domain_dir = Path(dataroot) / f"{phase}A"
    else:
        domain_dir = Path(dataroot) / f"{phase}B"
    return len(list(domain_dir.glob("*.nii.gz")))

if __name__ == "__main__":
    # 解析参数
    opt = TestOptions().parse()
    # 支持 --phase all，或指定 train/val/test
    phases = []
    if hasattr(opt, 'phase'):
        phases = [opt.phase]
        if opt.phase == 'all':
            phases = ['train', 'val', 'test']
    else:
        phases = ['train', 'val', 'test']

    # 设备设置
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 强制 monai 数据集
    opt.dataset_mode = 'monai'
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    if not MONAI_AVAILABLE:
        print("MONAI is required for this script. Please install MONAI.")
        sys.exit(1)

    for phase in phases:
        print(f"\n===== Predicting phase: {phase} =====")
        opt.phase = phase
        # 自动统计主域 nii.gz 文件数量
        num_files = count_nii_files(opt.dataroot, phase, opt.direction)
        opt.num_test = num_files
        print(f"Found {num_files} volumes for phase '{phase}' (set as num_test)")
        tester = MonaiTester(opt)
        tester.test_volumes()
    print("\nAll phases completed.")
