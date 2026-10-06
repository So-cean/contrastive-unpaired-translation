"""A single inference entry for image datasets and medical volumes."""
from copy import copy
from pathlib import Path


def infer_images(opt, model):
    from data import create_dataset
    from util import html
    from util.visualizer import save_images
    dataset = create_dataset(opt)
    output = Path(opt.results_dir) / opt.name / f'{opt.phase}_{opt.epoch}'
    webpage = html.HTML(str(output), f'{opt.name}: {opt.phase}, epoch {opt.epoch}')
    count = 0
    for data in dataset:
        if opt.num_test > 0 and count >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        save_images(webpage, model.get_current_visuals(), model.get_image_paths(), width=opt.display_winsize)
        count += 1
    webpage.save()
    print(f'Saved {count} images to {output}', flush=True)
    return count


def main():
    from runtime.device import configure_device
    configure_device()
    from accelerate.utils import set_seed
    from options.inference_options import InferenceOptions
    from models import create_model
    set_seed(42)
    opt = InferenceOptions().parse()
    if opt.model == 'sb':
        raise ValueError('The experimental SB model does not yet provide a validated inference workflow.')
    if opt.num_test < 0:
        raise ValueError('--num_test must be 0 (all) or a positive limit.')
    if opt.dataset_mode == 'monai3d':
        raise ValueError('monai3d inference requires a 3D model, which is not provided.')
    if opt.dataset_mode == 'monai' and (opt.model != 'cut' or opt.input_nc != 1 or opt.output_nc != 1):
        raise ValueError('NIfTI inference currently requires --model cut --input_nc 1 --output_nc 1.')
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    model = create_model(opt)
    model.setup(opt)
    if opt.eval or opt.dataset_mode == 'monai':
        model.eval()
    phases = ['train', 'val', 'test'] if opt.phase == 'all' else [opt.phase]
    total = 0
    for phase in phases:
        phase_opt = copy(opt)
        phase_opt.phase = phase
        source = 'A' if opt.direction == 'AtoB' else 'B'
        source_dir = Path(opt.dataroot) / f'{phase}{source}'
        if opt.phase == 'all' and not source_dir.is_dir():
            print(f'Skipping missing phase: {source_dir}', flush=True)
            continue
        model.opt = phase_opt
        if opt.dataset_mode == 'monai':
            from runtime.medical import infer_volumes
            total += infer_volumes(phase_opt, model)
        else:
            total += infer_images(phase_opt, model)
    if total == 0:
        raise ValueError('No source images found for the requested phase(s).')
