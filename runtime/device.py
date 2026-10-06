"""Device setup shared by public entry points; no import-time configuration."""
import importlib.util
import os
import shutil


def configure_device():
    import torch
    cpu = os.environ.get('ACCELERATE_USE_CPU', '').lower() in ('true', '1')
    if not cpu and shutil.which('npu-smi') and importlib.util.find_spec('torch_npu'):
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False
        os.environ.setdefault('HCCL_EXEC_TIMEOUT', '120')
        os.environ.setdefault('HCCL_CONNECT_TIMEOUT', '120')
    torch.autograd.set_detect_anomaly(os.environ.get('CUT_DETECT_ANOMALY') == '1')
