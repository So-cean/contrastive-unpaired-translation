#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Schrödinger Bridge (SB) Model
Uses dual dataloaders for independent domain A and domain B sampling
"""

import time
import os
import torch
torch.autograd.set_detect_anomaly(os.environ.get("CUT_DETECT_ANOMALY") == "1")

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


# ============= Accelerate Imports =============
from accelerate import Accelerator
from accelerate.utils import set_seed
# ==============================================

import importlib
import shutil

# NPU/Ascend environment setup - torch_npu will automatically replace cuda with npu
if os.environ.get("ACCELERATE_USE_CPU", "").lower() not in ("true", "1") and shutil.which("npu-smi") and importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    os.environ['HCCL_EXEC_TIMEOUT'] = '120'
    os.environ['HCCL_CONNECT_TIMEOUT'] = '120'


if __name__ == '__main__':
    opt = TrainOptions().parse()

    # ============= Initialize Accelerate =============
    accelerator = Accelerator(cpu=not bool(opt.gpu_ids))
    set_seed(42)

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main_process = accelerator.is_main_process
    device = accelerator.device
    local_rank = accelerator.local_process_index

    print(f"[Rank {rank}/{world_size}] Local rank: {local_rank}, Device: {device}, Main: {is_main_process}", flush=True)

    opt.gpu_ids = [] if device.type == "cpu" else [local_rank]
    # ==================================================

    # ============= Create Dual Datasets =============
    # SB model needs two independent datasets with DIFFERENT random seeds
    # Both datasets load both domains, but with independent random sampling:
    #   - data['A'] from dataset 1  -> real_A (anchor)
    #   - data2['A'] from dataset 2 -> real_A2 (negative, for SB contrastive learning)
    #   - data2['B'] from dataset 2 -> real_B (target)
    #
    # Key: different seed_offset ensures data['A'] and data2['A'] are INDEPENDENT samples

    dataset = create_dataset(opt, accelerator=accelerator, seed_offset=0)   # Base seed
    dataset2 = create_dataset(opt, accelerator=accelerator, seed_offset=1)  # Different seed for independent sampling

    if len(dataset.dataloader) == 0:
        raise ValueError("No complete training batch: reduce batch_size/process count or add training volumes.")
    dataset_size = len(dataset)
    if is_main_process:
        print(f"Dataset size per rank: {dataset_size}, total: {dataset_size * world_size}")
        print(f"Using dual dataloaders for SB model with independent random seeds (seed_offset=0 and 1)")
        print(f"  - data['A'] from dataset 1 -> real_A (anchor)")
        print(f"  - data2['A'] from dataset 2 -> real_A2 (negative for SB contrastive)")
        print(f"  - data2['B'] from dataset 2 -> real_B (target)")
    # ================================================

    # ============= Create Model =============
    model = create_model(opt)
    model.device = device
    model.setup(opt)  # Setup schedulers and print networks
    # ==========================================

    # 只在主进程创建visualizer
    visualizer = None
    if is_main_process:
        visualizer = Visualizer(opt)
        opt.visualizer = visualizer

    total_iters = 0
    optimize_time = 0.1

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        if visualizer:
            visualizer.reset()

        dataset.set_epoch(epoch)
        dataset2.set_epoch(epoch)
        model.current_epoch = epoch

        # Use zip to iterate over two datasets independently
        for i, (data, data2) in enumerate(zip(dataset, dataset2)):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size

            optimize_start_time = time.time()

            # ============= Data Dependent Initialize =============
            if epoch == opt.epoch_count and i == 0:
                # Pass both data to data_dependent_initialize for SB model
                model.data_dependent_initialize(data, data2=data2, accelerator=accelerator)

                model.prepare_training(accelerator)

                accelerator.wait_for_everyone()

                if is_main_process:
                    print(f"Model prepared with Accelerate DDP")
            # ======================================================

            # Pass both data to set_input
            model.set_input(data, data2=data2)
            model.optimize_parameters()

            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.995 + 0.005 * optimize_time

            # -------------------- Visualization (仅主进程) --------------------
            if is_main_process and visualizer and total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # -------------------- Print losses (仅主进程) --------------------
            if is_main_process and total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                print(flush=True)

            # -------------------- Save latest (仅主进程) --------------------
            if is_main_process and total_iters % opt.save_latest_freq == 0:
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix, accelerator)

            iter_data_time = time.time()

        # -------------------- Save epoch (仅主进程) --------------------
        if is_main_process and epoch % opt.save_epoch_freq == 0:
            model.save_networks('latest', accelerator)
            model.save_networks(epoch, accelerator)

        if is_main_process:
            print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec", flush=True)

        model.update_learning_rate()
        accelerator.wait_for_everyone()
