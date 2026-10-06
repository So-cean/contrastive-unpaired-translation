"""Slice-wise CUT prediction and NIfTI export in the preprocessed grid."""
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from data.medical_transforms import volume_transform


def save_volume(values, image, source_path, output_path):
    header = nib.load(source_path).header.copy()
    header.set_data_dtype(np.float32)
    affine = image.affine.detach().cpu().numpy()
    output = nib.Nifti1Image(np.asarray(values, dtype=np.float32), affine, header)
    nib.save(output, output_path)


def infer_volumes(opt, model):
    source_domain, target_domain = ('A', 'B') if opt.direction == 'AtoB' else ('B', 'A')
    root = Path(opt.dataroot)
    source_paths = sorted((root / f'{opt.phase}{source_domain}').glob('*.nii.gz'))
    target_paths = sorted((root / f'{opt.phase}{target_domain}').glob('*.nii.gz'))
    if opt.num_test > 0:
        source_paths, target_paths = source_paths[:opt.num_test], target_paths[:opt.num_test]
    if not source_paths:
        return 0
    output_dir = Path(opt.results_dir) / opt.name / f'{opt.phase}_{opt.epoch}'
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = volume_transform(opt.pixel_dim)
    for path in source_paths:
        image = transform({'image': str(path)})['image']
        volume = image.as_tensor()[0]
        fake = torch.empty_like(volume, device='cpu')
        with torch.inference_mode():
            for z in range(volume.shape[-1]):
                source = (volume[..., z] * 2 - 1)[None, None].to(model.device)
                generated = model.netG(source)
                fake[..., z] = ((generated[0, 0] + 1) / 2).cpu()
        real = volume.cpu().numpy()
        stem = path.name[:-7]
        save_volume(fake.numpy() * (real > 0), image, path, output_dir / f'{stem}_fake_{target_domain}.nii.gz')
        save_volume(real, image, path, output_dir / f'{stem}_real_{source_domain}.nii.gz')
    for path in target_paths:
        image = transform({'image': str(path)})['image']
        save_volume(image.as_tensor()[0].cpu().numpy(), image, path,
                    output_dir / f'{path.name[:-7]}_real_{target_domain}.nii.gz')
    print(f'Saved {len(source_paths)} predicted volumes to {output_dir}', flush=True)
    return len(source_paths)
