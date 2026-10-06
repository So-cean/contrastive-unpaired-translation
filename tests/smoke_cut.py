"""Offline CUT regression: updates, netF variants, weights, DDP agreement.

Run with ACCELERATE_USE_CPU=true for CPU, or CUT_SMOKE_DEVICE=npu for NPU.
No clinical data or downloaded weights are needed.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
# Use the same optional NPU compatibility setup as the actual training entry.
from train import TrainOptions, create_model
from accelerate import Accelerator
from accelerate.utils import set_seed


def unwrapped(accelerator, net):
    return accelerator.unwrap_model(net)


def main():
    torch.set_num_threads(1)
    requested_device = os.environ.get('CUT_SMOKE_DEVICE', 'cpu')
    accelerator = Accelerator(cpu=requested_device == 'cpu')
    assert accelerator.device.type == requested_device, (accelerator.device, requested_device)
    # A separate temp directory on every rank avoids concurrent option-file writes.
    with tempfile.TemporaryDirectory(prefix='cut-smoke-') as work:
        for feature_mode, nce_weight in [('mlp_sample', 1), ('sample', 1), ('mlp_sample', 0)]:
            set_seed(42)
            args = (
                f'--gpu_ids -1 --checkpoints_dir {work} --name smoke '
                '--model cut --input_nc 1 --output_nc 1 --netG resnet_6blocks '
                '--ngf 4 --ndf 4 --netF_nc 8 --batch_size 1 --num_threads 0 '
                '--nce_layers 0,4,8 --num_patches 8 --nce_idt true '
                '--lambda_SSIM 0 --lambda_canny 0 --lambda_elastic 0 '
                f'--lambda_perceptual 0 --netF {feature_mode} --lambda_NCE {nce_weight}'
            )
            opt = TrainOptions(cmd_line=args).parse()
            opt.gpu_ids = [] if requested_device == 'cpu' else [accelerator.local_process_index]
            model = create_model(opt)
            assert not hasattr(model, 'vgg'), 'Disabled perceptual loss must not download VGG.'
            model.setup(opt)
            set_seed(100 + accelerator.process_index)
            batch = {
                'A': torch.rand(1, 1, 64, 64) * 2 - 1,
                'B': torch.rand(1, 1, 64, 64) * 2 - 1,
                'A_paths': ['synthetic-A'], 'B_paths': ['synthetic-B'],
            }
            model.data_dependent_initialize(batch, accelerator=accelerator)
            model.prepare_training(accelerator)
            assert len(model.schedulers) == len(model.optimizers)
            before = {n: next(unwrapped(accelerator, getattr(model, 'net' + n)).parameters()).detach().clone()
                      for n in ('G', 'D')}
            if hasattr(model, 'optimizer_F'):
                before['F'] = next(unwrapped(accelerator, model.netF).parameters()).detach().clone()
            for _ in range(2):
                model.set_input(batch)
                model.optimize_parameters()
                assert all(torch.isfinite(torch.as_tensor(v)) for v in model.get_current_losses().values())
            for name, previous in before.items():
                net = unwrapped(accelerator, getattr(model, 'net' + name))
                assert not torch.equal(previous, next(net.parameters())), f'net{name} did not update'
                for parameter in net.parameters():
                    if accelerator.num_processes > 1:
                        expected = parameter.detach().clone()
                        torch.distributed.broadcast(expected, src=0)
                        torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-6)
            model.update_learning_rate()
            if accelerator.is_main_process:
                model.save_networks('smoke', accelerator)
                opt.continue_train = True
                opt.epoch = 'smoke'
                restored = create_model(opt)
                restored.setup(opt)
                restored.data_dependent_initialize(batch)
                for name in model.model_names:
                    expected = unwrapped(accelerator, getattr(model, 'net' + name)).state_dict()
                    actual = getattr(restored, 'net' + name).state_dict()
                    assert expected.keys() == actual.keys()
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            accelerator.wait_for_everyone()
            accelerator.free_memory()
            print(f'PASS rank={accelerator.process_index} netF={feature_mode} lambda_NCE={nce_weight}', flush=True)
    accelerator.end_training()


if __name__ == '__main__':
    main()
