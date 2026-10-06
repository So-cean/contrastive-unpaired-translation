"""Mirror an existing split into another modality using explicit roots."""
import argparse
import os
from pathlib import Path

SPLITS = ('trainA', 'trainB', 'valA', 'valB', 'testA', 'testB')


def prepare(args):
    split_root = args.split_root.expanduser().resolve(strict=True)
    source_root = args.source_root.expanduser().resolve(strict=True)
    target_root = args.target_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output == split_root or output.is_relative_to(split_root):
        raise ValueError('Output must be separate from the input split.')
    plan = {}
    missing = []
    for split in SPLITS:
        for entry in sorted((split_root / split).glob('*')):
            if not entry.name.endswith(('.nii', '.nii.gz')):
                continue
            relative = entry.resolve(strict=True).relative_to(source_root)
            if args.source_token not in relative.name:
                raise ValueError(f'Source token {args.source_token!r} is absent from {entry.name}')
            target_name = relative.name.replace(args.source_token, args.target_token)
            target = target_root / relative.parent / target_name
            if not target.is_file():
                missing.append(target)
                continue
            destination = output / split / target_name
            if destination in plan and plan[destination] != target:
                raise ValueError(f'Two source files map to {destination}')
            if destination.exists() or destination.is_symlink():
                if destination.is_symlink() and destination.resolve() == target.resolve():
                    continue
                raise FileExistsError(f'Refusing to replace {destination}')
            plan[destination] = target
    if missing:
        raise FileNotFoundError('Missing target files; nothing written:\n' + '\n'.join(map(str, missing)))
    if not plan:
        print('No new links to create.')
    for destination, target in plan.items():
        print(f'{destination} -> {target}')
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(os.path.relpath(target, destination.parent))
    print(f'{"Planned" if args.dry_run else "Created"} {len(plan)} links.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--target-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-token', default='_t1')
    parser.add_argument('--target-token', default='_t2')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.source_token:
        parser.error('--source-token cannot be empty')
    prepare(args)


if __name__ == '__main__':
    main()
