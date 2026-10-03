"""Dataset discovery, validation, splitting and manifest I/O.

Deliberately has no torch dependency (only Pillow) so that dataset preparation
can run on any machine, independently of the training environment.

Nothing in here ever modifies, moves or deletes the original dataset files.
"""
import os
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.webp'})

# Images above this many pixels are reported as "oversized" and kept out of the
# manifests (they are logged, not silently dropped). 50 MP is well under
# Pillow's own decompression-bomb warning threshold (~89 MP).
DEFAULT_MAX_PIXELS = 50_000_000

MANIFEST_NAMES = ('content_train', 'content_val', 'style_train', 'style_val')


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_images(root, exclude=()):
    """Recursively find images under ``root`` (any depth, case-insensitive ext).

    Returns ``(image_paths, other_ext_counts)``. ``image_paths`` is a sorted list
    of absolute path strings (sorted => deterministic across runs).
    ``other_ext_counts`` counts every non-image file by extension so ignored
    files (e.g. annotation .json, metadata .csv) are reported, not hidden.
    ``exclude`` is a collection of directory names to skip (case-insensitive).
    """
    root = os.path.abspath(str(root))
    if not os.path.isdir(root):
        raise NotADirectoryError(f'Not a directory: {root}')

    excluded = {e.lower() for e in exclude}
    images = []
    other = Counter()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in excluded)
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTS:
                images.append(os.path.join(dirpath, name))
            else:
                other[ext or '(no extension)'] += 1
    images.sort()
    return images, other


# --------------------------------------------------------------------------- #
# Safe loading / validation
# --------------------------------------------------------------------------- #
def load_rgb(path):
    """Open an image fully and return an RGB copy. The file is never modified.

    Images with transparency are composited onto white (a plain ``convert('RGB')``
    would turn transparent pixels into whatever colour is hidden under the alpha).
    Raises on corrupt/truncated files (Pillow's default strict behaviour).
    """
    with Image.open(path) as im:
        im.load()
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
            rgba = im.convert('RGBA')
            background = Image.new('RGB', rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel('A'))
            return background
        return im.convert('RGB')


def validate_image(args):
    """Check one image. Returns ``(path, status, detail, mode)``.

    status is 'ok', 'oversized' or 'invalid'. The size is read from the header
    first so oversized images are flagged without being decoded. Everything else
    is fully decoded, which is what catches truncated/corrupt files.
    """
    path, max_pixels = args
    try:
        with Image.open(path) as im:
            width, height = im.size
            mode = im.mode
            if width < 1 or height < 1:
                return path, 'invalid', f'degenerate size {width}x{height}', mode
            if max_pixels and width * height > max_pixels:
                return path, 'oversized', f'{width}x{height} = {width * height / 1e6:.1f} MP', mode
            im.load()
        return path, 'ok', f'{width}x{height}', mode
    except Exception as exc:  # noqa: BLE001 - we want to log every failure kind
        return path, 'invalid', f'{type(exc).__name__}: {exc}', ''


def validate_images(paths, max_pixels=DEFAULT_MAX_PIXELS, workers=1, progress=True):
    """Validate many images (optionally in parallel). Order follows ``paths``."""
    jobs = [(p, max_pixels) for p in paths]
    iterator = None
    pool = None
    if workers and workers > 1:
        pool = Pool(workers)
        iterator = pool.imap(validate_image, jobs, chunksize=64)
    else:
        iterator = map(validate_image, jobs)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(jobs), desc='Validating', unit='img')
        except ImportError:
            pass
    try:
        return list(iterator)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


# --------------------------------------------------------------------------- #
# Splitting, subsets, manifests
# --------------------------------------------------------------------------- #
def split_paths(paths, val_count, seed):
    """Seeded train/val split. Same inputs + seed => identical output."""
    ordered = sorted(paths)
    random.Random(seed).shuffle(ordered)
    val_count = max(0, min(val_count, len(ordered)))
    return sorted(ordered[val_count:]), sorted(ordered[:val_count])


def sample_subset(paths, n, seed):
    """Deterministic random subset of size ``n`` (all of them if fewer exist)."""
    ordered = sorted(paths)
    if n >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, n))


def write_manifest(path, image_paths):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for p in image_paths:
            f.write(p + '\n')


def read_manifest(path):
    """One image path per line; blank lines and '#' comments are ignored."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Manifest not found: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def load_split_manifests(manifest_dir, check_sample=200, seed=0):
    """Load content/style train/val manifests and sanity-check them.

    Raises if a train and val list overlap (leakage) or if the listed files
    no longer exist (e.g. the dataset was moved after preparation).
    """
    manifest_dir = Path(manifest_dir)
    splits = {name: read_manifest(manifest_dir / f'{name}.txt') for name in MANIFEST_NAMES}

    for role in ('content', 'style'):
        overlap = set(splits[f'{role}_train']) & set(splits[f'{role}_val'])
        if overlap:
            raise ValueError(f'{role}: {len(overlap)} images appear in both train and val '
                             f'manifests, e.g. {sorted(overlap)[0]}')

    rng = random.Random(seed)
    for name, paths in splits.items():
        probe = paths if len(paths) <= check_sample else rng.sample(paths, check_sample)
        missing = [p for p in probe if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f'{name}.txt lists files that do not exist (checked {len(probe)}), e.g. '
                f'{missing[0]}\nThe dataset was probably moved; re-run prepare_data.py scan.')
    return splits
