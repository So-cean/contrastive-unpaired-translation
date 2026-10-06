"""One epoch/checkpoint loop, with an optional second data stream for SB."""
import time


def main():
    from runtime.device import configure_device
    configure_device()
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from options.train_options import TrainOptions
    from data import create_dataset
    from models import create_model
    from util.visualizer import Visualizer

    opt = TrainOptions().parse()
    if opt.dataset_mode == 'monai3d':
        raise ValueError('monai3d is an experimental loader; no complete 3D training model is provided.')
    if opt.phase != 'train':
        raise ValueError('Training requires --phase train; use inference.py for val/test.')
    accelerator = Accelerator(cpu=not bool(opt.gpu_ids))
    set_seed(42)
    opt.gpu_ids = [] if accelerator.device.type == 'cpu' else [accelerator.local_process_index]
    dual_stream = opt.model.lower() == 'sb'
    if dual_stream and accelerator.num_processes > 1:
        raise ValueError('The experimental SB model is currently supported in single-process training only.')
    loaders = [create_dataset(opt, accelerator, seed_offset=i) for i in range(2 if dual_stream else 1)]
    if any(len(loader.dataloader) == 0 for loader in loaders):
        raise ValueError('No complete training batch: reduce batch_size/process count or add training data.')
    if accelerator.is_main_process:
        print(f'Device: {accelerator.device}; processes: {accelerator.num_processes}; '
              f'data streams: {len(loaders)}; samples/rank: {len(loaders[0])}', flush=True)
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt) if accelerator.is_main_process else None
    if visualizer:
        opt.visualizer = visualizer
    total_iters = 0
    initialized = False

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start = time.perf_counter()
        data_start = epoch_start
        epoch_iter = 0
        model.current_epoch = epoch
        for loader in loaders:
            loader.set_epoch(epoch)
        if visualizer:
            visualizer.reset()
        for batches in zip(*loaders):
            data = batches[0]
            extra = {'data2': batches[1]} if dual_stream else {}
            data_time = time.perf_counter() - data_start
            batch_size = data['A'].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if not initialized:
                # F's parameters must exist before DDP and optimizer preparation.
                model.data_dependent_initialize(data, accelerator=accelerator, **extra)
                model.prepare_training(accelerator)
                accelerator.wait_for_everyone()
                initialized = True
            step_start = time.perf_counter()
            model.set_input(data, **extra)
            model.optimize_parameters()
            step_time = (time.perf_counter() - step_start) / batch_size
            if visualizer:
                if total_iters % opt.display_freq == 0:
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch,
                                                       total_iters % opt.update_html_freq == 0)
                if total_iters % opt.print_freq == 0:
                    visualizer.print_current_losses(epoch, epoch_iter, model.get_current_losses(),
                                                    step_time, data_time)
                if total_iters % opt.save_latest_freq == 0:
                    suffix = f'iter_{total_iters}' if opt.save_by_iter else 'latest'
                    model.save_networks(suffix, accelerator)
            data_start = time.perf_counter()
        if accelerator.is_main_process:
            if epoch % opt.save_epoch_freq == 0:
                model.save_networks('latest', accelerator)
                model.save_networks(epoch, accelerator)
            print(f'End of epoch {epoch}: {time.perf_counter() - epoch_start:.1f}s', flush=True)
        model.update_learning_rate()
        accelerator.wait_for_everyone()
    accelerator.end_training()
