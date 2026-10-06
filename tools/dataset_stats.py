"""Summarize NIfTI volume counts, stored z spacing and slice counts per split."""
import argparse
import csv
from pathlib import Path


def analyze(root):
    import nibabel as nib
    import numpy as np
    if not root.is_dir():
        raise NotADirectoryError(root)
    rows = []
    for split in ('trainA', 'trainB', 'valA', 'valB', 'testA', 'testB'):
        paths = sorted(p for p in (root / split).glob('*') if p.name.endswith(('.nii', '.nii.gz')))
        spacings, slices = [], []
        for path in paths:
            image = nib.load(path)
            if len(image.shape) != 3:
                raise ValueError(f'Expected a 3D image: {path} has shape {image.shape}')
            spacings.append(float(image.header.get_zooms()[2]))
            slices.append(image.shape[2])
        if paths:
            rows.append(dict(split=split, volumes=len(paths), total_slices=sum(slices),
                             mean_slices=float(np.mean(slices)),
                             mean_z_spacing=float(np.mean(spacings)), std_z_spacing=float(np.std(spacings)),
                             min_z_spacing=min(spacings), max_z_spacing=max(spacings)))
    if not rows:
        raise ValueError(f'No NIfTI volumes found under {root}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataroot', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Optional CSV destination; prints only when omitted.')
    args = parser.parse_args()
    rows = analyze(args.dataroot.expanduser())
    for row in rows:
        print(row)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
