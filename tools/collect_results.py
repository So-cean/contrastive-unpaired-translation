"""Collect exported NIfTI images while retaining experiment/phase subfolders."""
import argparse
from pathlib import Path
import shutil


def category(name):
    lower = name.lower()
    stem = lower[:-7] if lower.endswith('.nii.gz') else lower[:-4] if lower.endswith('.nii') else ''
    for group in ('real_A', 'real_B', 'fake_A', 'fake_B'):
        if stem.endswith('_' + group.lower()):
            return group
    return None


def collect(root, output, dry_run=False, overwrite=False):
    root, output = root.expanduser().resolve(), output.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if root == output or root.is_relative_to(output):
        raise ValueError('Output must not equal or contain the input directory.')
    count = 0
    # Take a snapshot and exclude output, including a nested output directory.
    for path in sorted(root.rglob('*')):
        if path.is_relative_to(output) or not path.is_file():
            continue
        group = category(path.name)
        if group is None:
            continue
        destination = output / group / path.relative_to(root)
        if destination.exists() and not overwrite:
            continue
        print(f'{path} -> {destination}')
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        count += 1
    print(f'{"Planned" if dry_run else "Copied"} {count} files.')
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    collect(args.results_root, args.output, args.dry_run, args.overwrite)


if __name__ == '__main__':
    main()
