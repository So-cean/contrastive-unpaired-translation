#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter
import time
import os
from numpy import diff
import torch
torch.autograd.set_detect_anomaly(True)

import torch.distributed as dist
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

import lightning as pl
pl.seed_everything(42, workers=True)

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
    # os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '1'  # DEBUG级别
    os.environ['ASCEND_LAUNCH_BLOCKING'] = '1'  # 非阻塞模式
    # os.environ['NPU_DEBUG'] = '1'

# -------------------- 0. 轮值杂货工具 --------------------
class RankRoller:
    def __init__(self, world_size):
        self.world_size = world_size
        self.counter = 0

    def on_shift(self):
        # return self.counter % self.world_size
        return 0 # 暂时禁用轮值，全部由 rank0 执行

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
        
        # # ========== DEBUG DATA DISTRIBUTION (START) ==========
        # if epoch == opt.epoch_count:  # Only check first epoch
        #     print(f"\n{'='*80}", flush=True)
        #     print(f"[Rank {rank}] DEBUG: Checking data distribution...", flush=True)
        #     print(f"{'='*80}\n", flush=True)
            
        #     # Collect paths from first 30 batches
        #     collected_paths_A = []
        #     collected_paths_B = []
            
        #     batch_count = 0
        #     for i, data in enumerate(dataset):
        #         if batch_count >= 30:
        #             break
                
        #         A_paths = data['A_paths'] if isinstance(data['A_paths'], list) else [data['A_paths']]
        #         B_paths = data['B_paths'] if isinstance(data['B_paths'], list) else [data['B_paths']]
                
        #         collected_paths_A.extend(A_paths)
        #         collected_paths_B.extend(B_paths)
                
        #         # Print first 3 batches
        #         if batch_count < 3:
        #             import os
        #             filenames_A = [os.path.basename(p) for p in A_paths]
        #             filenames_B = [os.path.basename(p) for p in B_paths]
        #             print(f"[Rank {rank}] Batch {batch_count}:", flush=True)
        #             print(f"  A: {filenames_A}", flush=True)
        #             print(f"  B: {filenames_B}", flush=True)
                
        #         batch_count += 1
            
        #     # Analyze
        #     unique_A = len(set(collected_paths_A))
        #     unique_B = len(set(collected_paths_B))
        #     total_A = len(collected_paths_A)
        #     total_B = len(collected_paths_B)
            
        #     counter_A = Counter(collected_paths_A)
        #     counter_B = Counter(collected_paths_B)
        #     max_repeat_A = max(counter_A.values()) if counter_A else 0
        #     max_repeat_B = max(counter_B.values()) if counter_B else 0
            
        #     print(f"\n[Rank {rank}] === SUMMARY after {batch_count} batches ===", flush=True)
        #     print(f"Domain A:", flush=True)
        #     print(f"  Total samples: {total_A}", flush=True)
        #     print(f"  Unique files: {unique_A}", flush=True)
        #     print(f"  Repetition ratio: {total_A/unique_A if unique_A > 0 else 0:.2f}x", flush=True)
        #     print(f"  Max repeats of one file: {max_repeat_A}", flush=True)
            
        #     print(f"Domain B:", flush=True)
        #     print(f"  Total samples: {total_B}", flush=True)
        #     print(f"  Unique files: {unique_B}", flush=True)
        #     print(f"  Repetition ratio: {total_B/unique_B if unique_B > 0 else 0:.2f}x", flush=True)
        #     print(f"  Max repeats of one file: {max_repeat_B}", flush=True)
            
        #     # Warning flags
        #     if total_A / unique_A > 3:
        #         print(f"\n⚠️  [Rank {rank}] HIGH REPETITION IN A! Likely index collision.", flush=True)
        #     if total_B / unique_B > 3:
        #         print(f"\n⚠️  [Rank {rank}] HIGH REPETITION IN B! Likely index collision.", flush=True)
            
        #     # Show most common files
        #     print(f"\n[Rank {rank}] Most frequently seen A files:", flush=True)
        #     for path, count in counter_A.most_common(5):
        #         print(f"  {count}x: {os.path.basename(path)}", flush=True)
            
        #     print(f"\n[Rank {rank}] Most frequently seen B files:", flush=True)
        #     for path, count in counter_B.most_common(5):
        #         print(f"  {count}x: {os.path.basename(path)}", flush=True)
            
        #     print(f"\n{'='*80}\n", flush=True)
            
        #     if distributed:
        #         torch.distributed.barrier()
            
        #     # IMPORTANT: Reset dataset iterator after debug collection
        #     # Need to break and restart the epoch properly
        #     print(f"[Rank {rank}] Debug complete. Restarting epoch...\n", flush=True)
            
        # # ========== DEBUG DATA DISTRIBUTION (END) ==========
        
        
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
                model.parallelize(force_rewrap=True)  # 强制重新包装，确保 netF 的 MLP 被 DDP 包装
                if distributed:
                    torch.distributed.barrier()  # ← ADD THIS

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
                    visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
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

        if rank == 0:
            print(f"[Rank {rank}] End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec", flush=True)
        #
        # print(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay}  Time: {time.time() - epoch_start_time:.0f} sec')
        model.update_learning_rate()
        if distributed:
            torch.distributed.barrier()  

    if distributed:
        dist.destroy_process_group()
        
