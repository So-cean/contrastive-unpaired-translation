#!/usr/bin/env python3
"""Single-process CUT timing; excludes dataset startup and does not save weights."""
import os
import statistics
import time

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

# Reuse the training entry's optional NPU compatibility setup.
from train import TrainOptions, create_dataset, create_model


def synchronize(device):
    if device.type == 'npu':
        torch.npu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def main():
    opt = TrainOptions().parse()
    accelerator = Accelerator(cpu=not bool(opt.gpu_ids))
    if accelerator.num_processes != 1 or opt.model != 'cut':
        raise ValueError('This diagnostic supports single-process CUT only.')
    set_seed(42)
    opt.gpu_ids = [] if accelerator.device.type == 'cpu' else [accelerator.local_process_index]
    start = time.perf_counter()
    dataset = create_dataset(opt, accelerator)
    try:
        batch = next(iter(dataset))
    except StopIteration as exc:
        raise ValueError('No complete batch: reduce batch_size or add data.') from exc
    print(f'Dataset construction + first batch: {time.perf_counter() - start:.3f} s')
    model = create_model(opt)
    model.setup(opt)
    model.data_dependent_initialize(batch, accelerator)
    model.prepare_training(accelerator)
    elapsed = []
    for step in range(12):
        synchronize(accelerator.device)
        start = time.perf_counter()
        model.set_input(batch)
        model.optimize_parameters()
        synchronize(accelerator.device)
        duration = time.perf_counter() - start
        if step >= 2:
            elapsed.append(duration)
    mean = statistics.mean(elapsed)
    print(f'Device: {accelerator.device}; precision: {accelerator.mixed_precision}')
    print(f'2 warmup + 10 measured steps on the same batch; mean={mean:.4f} s, '
          f'min={min(elapsed):.4f} s, max={max(elapsed):.4f} s')
    print(f'Training-step throughput (excludes data loading): {len(batch["A"]) / mean:.2f} samples/s')
    print('No weights saved. This is not end-to-end or multi-device throughput.')
    accelerator.end_training()


if __name__ == '__main__':
    main()
