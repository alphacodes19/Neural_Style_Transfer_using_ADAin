"""Dataset preparation for AdaIN training (Phase 2).

Subcommands:
  inspect  Show the real directory structure of a dataset (read-only, no validation).
  scan     Recursively discover + validate images, split train/val, write manifests.
  smoke    Build a small deterministic subset from already-written manifests.

Original dataset files are only ever read. COCO annotation files and Painter by
Numbers CSVs are never opened; they are just counted as ignored files.
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from utils.data_prep import (DEFAULT_MAX_PIXELS, MANIFEST_NAMES, discover_images,
                             read_manifest, sample_subset, split_paths,
                             validate_images, write_manifest)


def _fmt_counter(counter, limit=None):
    items = counter.most_common(limit)
    return ', '.join(f'{k}: {v}' for k, v in items) if items else '(none)'


def _evenly_spaced(items, k=5):
    if len(items) <= k:
        return list(items)
    step = (len(items) - 1) / (k - 1)
    return [items[round(i * step)] for i in range(k)]


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #
def cmd_inspect(args):
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f'Not a directory: {root}')

    all_ext = Counter()
    files_per_dir = Counter()      # directory -> number of image files directly inside
    n_dirs = 0
    max_depth = 0
    for dirpath, dirnames, filenames in os.walk(root):
        n_dirs += 1
        depth = 0 if dirpath == root else len(Path(dirpath).relative_to(root).parts)
        max_depth = max(max_depth, depth)
        for name in filenames:
            ext = os.path.splitext(name)[1].lower() or '(no extension)'
            all_ext[ext] += 1
            if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                files_per_dir[dirpath] += 1

    print(f'Root:                {root}')
    print(f'Directories:         {n_dirs}  (max depth below root: {max_depth})')
    print(f'Files (all types):   {sum(all_ext.values())}')
    print(f'Extensions (all):    {_fmt_counter(all_ext)}')
    print(f'Image files:         {sum(files_per_dir.values())}')
    print(f'Dirs containing images: {len(files_per_dir)}')
    print(f'\nTop {args.top} directories by number of images directly inside them:')
    for d, n in files_per_dir.most_common(args.top):
        rel = os.path.relpath(d, root)
        print(f'  {n:>8}  {"." if rel == "." else rel}')
    non_image = {e: c for e, c in all_ext.items()
                 if e not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}}
    if non_image:
        print('\nNon-image files present (will be ignored, never parsed): '
              + _fmt_counter(Counter(non_image)))


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
def cmd_scan(args):
    role = args.role
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print(f'[{role}] discovering images under {os.path.abspath(args.root)} ...')
    found, other_ext = discover_images(args.root, exclude=args.exclude)
    if not found:
        sys.exit(f'No images found under {args.root}. Run '
                 f'"python prepare_data.py inspect --root {args.root}" to see what is there.')
    image_ext = Counter(os.path.splitext(p)[1].lower() for p in found)

    results = validate_images(found, max_pixels=args.max_pixels, workers=args.workers)

    valid = [r for r in results if r[1] == 'ok']
    oversized = [r for r in results if r[1] == 'oversized']
    invalid = [r for r in results if r[1] == 'invalid']
    modes = Counter(r[3] for r in valid)
    valid_paths = [r[0] for r in valid]

    # Nothing is dropped silently: every excluded file goes in the log.
    invalid_log = manifest_dir / f'{role}_invalid.tsv'
    with open(invalid_log, 'w', encoding='utf-8', newline='\n') as f:
        f.write('status\tpath\tdetail\n')
        for path, status, detail, _ in invalid + oversized:
            f.write(f'{status}\t{path}\t{detail}\n')

    # Cap the validation size so a small dataset is never mostly validation.
    val_count = min(args.val_count, int(len(valid_paths) * 0.2))
    train, val = split_paths(valid_paths, val_count, args.seed)
    write_manifest(manifest_dir / f'{role}_train.txt', train)
    write_manifest(manifest_dir / f'{role}_val.txt', val)

    report = {
        'role': role, 'root': os.path.abspath(args.root), 'seed': args.seed,
        'max_pixels': args.max_pixels, 'discovered': len(found), 'valid': len(valid),
        'invalid': len(invalid), 'oversized': len(oversized),
        'train': len(train), 'val': len(val),
        'image_extensions': dict(image_ext), 'color_modes_of_valid': dict(modes),
        'ignored_non_image_files': dict(other_ext),
    }
    with open(manifest_dir / f'{role}_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f'\n=== {role.upper()} dataset report ===')
    print(f'Discovered images:       {len(found)}')
    print(f'Valid images:            {len(valid)}')
    print(f'Invalid/corrupt images:  {len(invalid)}')
    print(f'Oversized images:        {len(oversized)}  (> {args.max_pixels:,} pixels)')
    print(f'Image extensions:        {_fmt_counter(image_ext)}')
    print(f'Colour modes (valid):    {_fmt_counter(modes)}  (all converted to RGB at load time)')
    print(f'Ignored non-image files: {_fmt_counter(other_ext)}')
    print('Sample paths:')
    for p in _evenly_spaced(found):
        print(f'  {p}')
    if invalid or oversized:
        print(f'Excluded files are listed in: {invalid_log}')
        for path, status, detail, _ in (invalid + oversized)[:5]:
            print(f'  [{status}] {path}  ->  {detail}')
    if val_count < args.val_count:
        print(f'Note: val_count capped from {args.val_count} to {val_count} (20% of valid images).')
    print(f'Split (seed={args.seed}): train={len(train)}  val={len(val)}')
    print(f'Wrote: {manifest_dir / (role + "_train.txt")}')
    print(f'       {manifest_dir / (role + "_val.txt")}')
    print(f'       {manifest_dir / (role + "_report.json")}')


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def cmd_smoke(args):
    src = Path(args.manifest_dir)
    out = Path(args.out_dir)
    for name in MANIFEST_NAMES:
        paths = read_manifest(src / f'{name}.txt')
        n = args.n if name.endswith('_train') else args.val_n
        subset = sample_subset(paths, n, args.seed)
        write_manifest(out / f'{name}.txt', subset)
        print(f'{name:<14} {len(subset):>6} of {len(paths)}')
    print(f'Smoke manifests written to {out} (seed={args.seed}). '
          f'Train with: python train.py --manifest_dir {out}')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    ins = sub.add_parser('inspect', help='show the actual folder structure of a dataset')
    ins.add_argument('--root', required=True)
    ins.add_argument('--top', type=int, default=15, help='how many directories to list')
    ins.set_defaults(func=cmd_inspect)

    scan = sub.add_parser('scan', help='discover, validate, split, write manifests')
    scan.add_argument('--root', required=True, help='dataset root (searched recursively)')
    scan.add_argument('--role', required=True, choices=['content', 'style'],
                      help='content = COCO, style = Painter by Numbers')
    scan.add_argument('--manifest_dir', default=str(Path('data') / 'manifests'))
    scan.add_argument('--val_count', type=int, default=1000,
                      help='number of held-out validation images (capped at 20%% of valid)')
    scan.add_argument('--seed', type=int, default=42)
    scan.add_argument('--max_pixels', type=int, default=DEFAULT_MAX_PIXELS,
                      help='images above this pixel count are logged as oversized and excluded')
    scan.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    scan.add_argument('--exclude', nargs='*', default=[],
                      help='directory names to skip entirely (case-insensitive)')
    scan.set_defaults(func=cmd_scan)

    smoke = sub.add_parser('smoke', help='small deterministic subset from existing manifests')
    smoke.add_argument('--manifest_dir', default=str(Path('data') / 'manifests'))
    smoke.add_argument('--out_dir', default=str(Path('data') / 'manifests_smoke'))
    smoke.add_argument('--n', type=int, default=500, help='train images per role')
    smoke.add_argument('--val_n', type=int, default=50, help='val images per role')
    smoke.add_argument('--seed', type=int, default=42)
    smoke.set_defaults(func=cmd_smoke)
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    args.func(args)
