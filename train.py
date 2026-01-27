#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import torch
torch.autograd.set_detect_anomaly(True)

import torch.distributed as dist
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

import importlib
import shutil
if shutil.which("npu-smi") and importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    os.environ['HCCL_EXEC_TIMEOUT'] = '120'  
    os.environ['HCCL_CONNECT_TIMEOUT'] = '120'
    
    # ========== 调试和日志 ==========
    os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '1'  # DEBUG级别
    os.environ['ASCEND_LAUNCH_BLOCKING'] = '1'  # 非阻塞模式
    os.environ['NPU_DEBUG'] = '1'

# -------------------- 0. 轮值杂货工具 --------------------
class RankRoller:
    def __init__(self, world_size):
        self.world_size = world_size
        self.counter = 0

    def on_shift(self):
        return self.counter % self.world_size

    def step(self):
        self.counter += 1
# -------------------------------------------------------

if __name__ == '__main__':
    opt = TrainOptions().parse()

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = world_size > 1

    if distributed:
        dist.init_process_group(backend='hccl', init_method='env://')
        torch.npu.set_device(local_rank)
        opt.gpu_ids = [local_rank]
    else:
        if len(opt.gpu_ids) > 0:
            torch.npu.set_device(opt.gpu_ids[0])

    roller = RankRoller(world_size)                       # ← 新增

    dataset = create_dataset(opt)
    dataset_size = len(dataset)

    model = create_model(opt)
    model.setup(opt)
    model.parallelize()

    visualizer = Visualizer(opt)
    opt.visualizer = visualizer
    total_iters = 0
    optimize_time = 0.1

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size

            optimize_start_time = time.time()

            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()

            model.set_input(data)
            model.optimize_parameters()

            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            # -------------------- 1. 可视化轮值 --------------------
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                on_duty = roller.on_shift()
                if rank == on_duty:
                    print(f'[rank{rank}] on duty → visualize iter {total_iters}', flush=True)   # ← 新增
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # -------------------- 2. 打印轮值 --------------------
            if total_iters % opt.print_freq == 0:
                on_duty = roller.on_shift()
                if rank == on_duty:
                    losses = model.get_current_losses()
                    print(f'[rank{rank}] on duty → losses @ iter {total_iters}: ', end='', flush=True)
                    for k, v in losses.items():
                        print(f'{k} {v:.3f} ', end='', flush=True)
                    print(flush=True)

            # -------------------- 3. 保存轮值 --------------------
            if total_iters % opt.save_latest_freq == 0:
                on_duty = roller.on_shift()
                if rank == on_duty:
                    print(f'[rank{rank}] on duty → saving iter {total_iters}', flush=True)
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

            iter_data_time = time.time()
            roller.step()                                           # 每 iter 轮换

        # -------------------- 4. epoch 保存也轮值 --------------------
        if epoch % opt.save_epoch_freq == 0:
            on_duty = roller.on_shift()
            if rank == on_duty:
                print(f'[rank{rank}] on duty → saving epoch {epoch}', flush=True)
                model.save_networks('latest')
                model.save_networks(epoch)

        print(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay}  Time: {time.time() - epoch_start_time:.0f} sec')
        model.update_learning_rate()
        torch.distributed.barrier()  

    if distributed:
        dist.destroy_process_group()
        
        # export HCCL_CONNECT_TIMEOUT= 120     
# export HCCL_EXEC_TIMEOUT= 120