"""Synthetic NIfTI -> actual train CLI -> prediction -> geometry checks.

Run from the repository root. CUT_SMOKE_DEVICE=cpu (default) or npu.
All data, checkpoints and predictions live in a temporary directory.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import torch
from options.train_options import TrainOptions
from data import create_dataset


def main():
    torch.set_num_threads(1)
    requested = os.environ.get('CUT_SMOKE_DEVICE', 'cpu')
    with tempfile.TemporaryDirectory(prefix='cut-nifti-') as work:
        root = Path(work)
        rng = np.random.default_rng(42)
        affine = np.diag([1., 1., 2., 1.])
        for split in ('trainA', 'trainB', 'testA'):
            directory = root / 'data' / split
            directory.mkdir(parents=True)
            for subject in range(2):
                volume = rng.random((48, 48, 2), dtype=np.float32)
                nib.save(nib.Nifti1Image(volume, affine), directory / f'synthetic-{subject}.nii.gz')
        common = [
            '--dataroot', str(root / 'data'), '--name', 'smoke',
            '--checkpoints_dir', str(root / 'checkpoints'),
            '--model', 'cut', '--dataset_mode', 'monai', '--netG', 'resnet_6blocks',
            '--input_nc', '1', '--output_nc', '1', '--ngf', '4', '--ndf', '4',
            '--gpu_ids', '-1' if requested == 'cpu' else '0',
            '--num_threads', '0', '--lambda_perceptual', '0',
        ]
        train = common + [
            '--n_epochs', '1', '--n_epochs_decay', '0', '--batch_size', '1',
            '--nce_idt', 'false', '--nce_layers', '0,4,8', '--netF_nc', '8',
            '--num_patches', '8', '--lambda_SSIM', '0', '--lambda_canny', '0',
            '--lambda_elastic', '0', '--save_epoch_freq', '1', '--display_id', '0', '--no_html',
        ]
        subprocess.run([sys.executable, 'train.py', *train], check=True)
        for network in ('G', 'D', 'F'):
            assert (root / 'checkpoints' / 'smoke' / f'latest_net_{network}.pth').is_file()
        subprocess.run([sys.executable, 'predict_monai.py', *common,
                        '--phase', 'test', '--results_dir', str(root / 'results')], check=True)
        output = root / 'results' / 'smoke' / 'test_latest'
        generated = sorted(output.glob('*_fake_B.nii.gz'))
        assert len(generated) == 2
        for path in generated:
            predicted = nib.load(path)
            reference = nib.load(str(path).replace('_fake_B', '_real_A'))
            assert predicted.shape == (256, 256, 2)
            np.testing.assert_allclose(predicted.affine, reference.affine)
            np.testing.assert_allclose(predicted.header.get_zooms(), (1, 1, 2))
            values = predicted.get_fdata()
            assert np.isfinite(values).all()
            assert values.min() >= 0 and values.max() <= 1
            assert np.all(values[reference.get_fdata() == 0] == 0)
        # Worker copies must receive the current epoch, and dual data streams
        # must receive their requested offset (including on a single process).
        opt = TrainOptions(cmd_line=' '.join(train + ['--gpu_ids', '-1', '--num_threads', '1'])).parse()
        loader = create_dataset(opt, seed_offset=1)
        assert loader.dataset.seed_offset == 1
        loader.dataloader.batch_sampler.sampler = torch.utils.data.SequentialSampler(loader.dataset)
        for epoch in (1, 2):
            loader.set_epoch(epoch)
            batches = list(loader)
            torch.testing.assert_close(batches[0]['B'][0], loader.dataset[0]['B'])
        print(f'PASS synthetic NIfTI training, prediction, affine/mask and worker epochs ({requested})', flush=True)


if __name__ == '__main__':
    main()
