"""Reproducible evaluation and checkpoint selection for the AdaIN decoder.

What it does, per checkpoint:
  1. Loads VGG + decoder and a fixed evaluation set drawn ONLY from the validation
     manifests (content_val.txt / style_val.txt), chosen with a seeded random sample.
  2. Validation losses on the fixed pairs, computed by train.compute_losses, i.e. the
     exact function used for training, with the same preprocessing as training
     validation (Resize(content_size) + CenterCrop(final_size)), in fp32.
  3. Alpha sweep (0, .25, .5, .75, 1) using the same operation as app.py:
        t = AdaIN(f_c, f_s);  decoder(alpha * t + (1 - alpha) * f_c);  clamp(0, 1)
  4. Output sanity checks (NaN/Inf, all-black, constant, value range).
  5. Inference benchmark (time and peak GPU memory) at 256 / 512 / 1024.

Loss definitions (identical to training, nothing re-defined here):
  content_loss   = content_weight * MSE(encoder(g)[relu4_1], t)         weighted component
  style_loss     = style_weight * sum_l [MSE(mean_l(g), mean_l(s)) + MSE(std_l(g), std_l(s))]
                   over relu1_1..relu4_1                                  weighted component
  style_loss_raw = style_loss / style_weight                              unweighted
  total_loss     = content_loss + style_loss
So the 'style' column in train_log.csv / val_log.csv and in this script's CSVs is the
STYLE-WEIGHTED value, and total == content + style. Losses are for alpha = 1.
Weights default to the ones stored in the first checkpoint (override with
--content_weight / --style_weight) and are applied to every checkpoint so totals are
comparable. Only compare checkpoints trained with the same weights.

Best checkpoint: lowest mean total_loss on the fixed set among checkpoints that pass
the integrity gate (no non-finite values, no all-black or constant outputs).
There is no combined quality score. Validation loss is a proxy; look at the grids.
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from train import compute_losses, get_val_transform
from utils.data_prep import load_split_manifests, sample_subset
from utils.models import Decoder, VGGEncoder
from utils.utils import ImageFolderDataset, adaptive_instance_normalization

DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Heuristic thresholds for the output sanity checks (documented in the CSV headers).
BLACK_MAX = 2.0 / 255.0        # clamped max at or below this => "black"
CONSTANT_STD = 1e-3            # clamped pixel std below this => "constant"
RANGE_FRAC_WARN = 0.05         # more than 5% of raw values outside [0, 1] => warning
RANGE_ABS_WARN = (-1.0, 2.0)   # any raw value outside this => warning


def parse_arguments():
    p = argparse.ArgumentParser(description='Evaluate one or more AdaIN decoder checkpoints.',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument('--checkpoint', nargs='+', required=True,
                   help='checkpoint_*.pth or decoder_*.pth files; wildcards are expanded by this '
                        'script, so they also work in PowerShell')
    p.add_argument('--vgg', default=str(Path('weights') / 'vgg_normalised.pth'))
    p.add_argument('--manifest_dir', default=str(Path('data') / 'manifests_final'),
                   help='directory with content_val.txt / style_val.txt (and the train lists)')
    p.add_argument('--out_dir', default='evaluation')
    p.add_argument('--best_dir', default=None, help='default: <out_dir>/best')
    p.add_argument('--no_best', action='store_true', help='do not write best_validation files')
    p.add_argument('--replace_best', action='store_true',
                   help='overwrite an existing best_validation even if it is better or not comparable')

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_images', type=int, default=20,
                   help='content images and style images in the fixed set (paired by index)')
    p.add_argument('--content_size', type=int, default=512)
    p.add_argument('--final_size', type=int, default=256)
    p.add_argument('--alphas', type=float, nargs='+', default=DEFAULT_ALPHAS)
    p.add_argument('--content_weight', type=float, default=None)
    p.add_argument('--style_weight', type=float, default=None)

    p.add_argument('--grid_pairs', type=int, default=6, help='pairs (rows) in the sample grids')
    p.add_argument('--no_individual_images', action='store_true',
                   help='only write grids, not every alpha_*/pair_*.png')

    p.add_argument('--bench_sizes', type=int, nargs='+', default=[256, 512, 1024])
    p.add_argument('--bench_runs', type=int, default=10)
    p.add_argument('--skip_benchmark', action='store_true')
    return p.parse_args()


def safe_name(path):
    path = Path(path)
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', f'{path.parent.name}_{path.stem}')


def expand_paths(patterns):
    out = []
    for pat in patterns:
        hits = sorted(glob.glob(pat)) if any(c in pat for c in '*?[') else [pat]
        if not hits:
            sys.exit(f'No file matches: {pat}')
        out.extend(hits)
    seen, unique = set(), []
    for p in out:
        key = str(Path(p).resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    for p in unique:
        if not Path(p).is_file():
            sys.exit(f'Checkpoint not found: {p}')
    return unique


def load_checkpoint(path, device):
    obj = torch.load(path, map_location=device, weights_only=True)
    if isinstance(obj, dict) and 'decoder' in obj:
        meta = {'iteration': obj.get('iteration'), 'args': obj.get('args') or {},
                'seed': obj.get('seed'), 'full_checkpoint': True}
        return obj['decoder'], meta
    return obj, {'iteration': None, 'args': {}, 'seed': None, 'full_checkpoint': False}


def sha256_head(path, n=16):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:n]


def finite_or_none(x):
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float('nan')


def std(xs):
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def atomic_copy(src, dst):
    dst = Path(dst)
    tmp = dst.with_name(dst.name + '.tmp')
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def atomic_torch_save(obj, dst):
    dst = Path(dst)
    tmp = dst.with_name(dst.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, dst)


@torch.no_grad()
def stylize_alphas(encoder, decoder, content, style, alphas):
    """Same operation as app.py: interpolate in feature space between content and AdaIN features."""
    c = encoder(content, is_test=True)
    s = encoder(style, is_test=True)
    t = adaptive_instance_normalization(c, s)
    return [decoder(a * t + (1 - a) * c) for a in alphas]


def check_output(raw):
    """Sanity checks on a raw (unclamped) decoder output of shape (1, 3, H, W)."""
    finite = torch.isfinite(raw)
    n_bad = int((~finite).sum().item())
    clean = torch.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)
    clamped = clean.clamp(0, 1)
    vals = raw[finite]
    if vals.numel() > 0:
        raw_min, raw_max = vals.min().item(), vals.max().item()
        frac_out = ((vals < 0) | (vals > 1)).float().mean().item()
    else:
        raw_min = raw_max = frac_out = float('nan')
    is_black = bool(clamped.max().item() <= BLACK_MAX)
    is_constant = bool(clamped.std().item() < CONSTANT_STD)
    range_warn = bool(n_bad == 0 and (frac_out > RANGE_FRAC_WARN or raw_min < RANGE_ABS_WARN[0]
                                      or raw_max > RANGE_ABS_WARN[1]))
    return {'nonfinite_values': n_bad, 'raw_min': raw_min, 'raw_max': raw_max,
            'frac_out_of_01': frac_out, 'is_black': is_black, 'is_constant': is_constant,
            'range_warn': range_warn}


def to_pil(t):
    return TF.to_pil_image(torch.nan_to_num(t.detach().float().cpu(), nan=0.0, posinf=1.0, neginf=0.0)
                           .squeeze(0).clamp(0, 1))


def tile_grid(rows, labels, pad=4, header=16):
    w, h = rows[0][0].size
    n_cols = len(labels)
    canvas = Image.new('RGB', (n_cols * w + (n_cols + 1) * pad,
                               header + len(rows) * h + (len(rows) + 1) * pad), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    max_chars = max(4, w // 6)
    for j, label in enumerate(labels):
        draw.text((pad + j * (w + pad) + 2, 2), label[:max_chars], fill=(0, 0, 0))
    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            canvas.paste(img, (pad + j * (w + pad), header + pad + i * (h + pad)))
    return canvas


def benchmark(encoder, decoder, sizes, runs, device, seed):
    """Time one full stylization (encode x2, AdaIN, decode) on seeded random inputs.
    Peak memory is torch.cuda.max_memory_allocated (weights included, CUDA context excluded)."""
    results = {}
    cuda = device.type == 'cuda'
    warmup = 2 if cuda else 0
    if not cuda:
        runs = min(runs, 2)
    gen = torch.Generator().manual_seed(seed)
    failed = False
    for size in sorted(sizes):
        if failed:
            results[size] = {'status': 'skipped (smaller size ran out of memory)', 'ms': None, 'mem_mb': None}
            continue
        x = y = out = None
        try:
            x = torch.rand(1, 3, size, size, generator=gen).to(device)
            y = torch.rand(1, 3, size, size, generator=gen).to(device)
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            for _ in range(warmup):
                stylize_alphas(encoder, decoder, x, y, [1.0])
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                out = stylize_alphas(encoder, decoder, x, y, [1.0])
            if cuda:
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / runs * 1000.0
            mem = torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else None
            results[size] = {'status': 'ok', 'ms': ms, 'mem_mb': mem}
        except RuntimeError as exc:
            msg = str(exc).lower()
            if 'out of memory' in msg or 'allocate' in msg:
                failed = True
                results[size] = {'status': 'OOM', 'ms': None, 'mem_mb': None}
            else:
                raise
        finally:
            del x, y, out
            if cuda:
                torch.cuda.empty_cache()
    return results


def write_csv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    alphas = list(args.alphas)

    ckpt_paths = expand_paths(args.checkpoint)
    names, used = [], set()
    for p in ckpt_paths:
        base = safe_name(p)
        name, k = base, 2
        while name in used:
            name, k = f'{base}_{k}', k + 1
        used.add(name)
        names.append(name)

    if not Path(args.vgg).is_file():
        sys.exit(f'VGG weights not found: {args.vgg}')
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f'Device: {device}' + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''))

    splits = load_split_manifests(args.manifest_dir)
    cv_all, sv_all = splits['content_val'], splits['style_val']
    n = min(args.num_images, len(cv_all), len(sv_all))
    if n < args.num_images:
        print(f'NOTE: only {n} validation images available per role (requested {args.num_images}).')
    cv = sample_subset(cv_all, n, args.seed)
    sv = sample_subset(sv_all, n, args.seed)
    train_set = set(splits['content_train']) | set(splits['style_train'])
    leaked = (set(cv) | set(sv)) & train_set
    if leaked:
        sys.exit(f'{len(leaked)} evaluation images are also in the training manifests, e.g. {sorted(leaked)[0]}')
    eval_set_id = hashlib.sha1(('|'.join(cv) + '#' + '|'.join(sv) + f'#{args.seed}#{args.content_size}#{args.final_size}').encode()).hexdigest()[:12]

    val_tf = get_val_transform(SimpleNamespace(content_size=args.content_size,
                                               final_size=args.final_size, crop=True))
    c_ds, s_ds = ImageFolderDataset(cv, val_tf), ImageFolderDataset(sv, val_tf)
    contents = torch.stack([c_ds[i] for i in range(n)])
    styles = torch.stack([s_ds[i] for i in range(n)])
    print(f'Fixed evaluation set: {n} content x {n} style images (paired by index), seed {args.seed}, '
          f'id {eval_set_id}, manifests {args.manifest_dir}')

    loaded = {}
    first_state, first_meta = load_checkpoint(ckpt_paths[0], device)
    loaded[0] = (first_state, first_meta)
    saved_args = first_meta['args']
    cw = args.content_weight if args.content_weight is not None else float(saved_args.get('content_weight', 1.0))
    sw = args.style_weight if args.style_weight is not None else float(saved_args.get('style_weight', 5.0))
    weights = SimpleNamespace(content_weight=cw, style_weight=sw)
    print(f'Loss weights used for ALL checkpoints: content_weight={cw}, style_weight={sw}')

    encoder = VGGEncoder(args.vgg, map_location=device).to(device)
    encoder.eval()
    decoder = Decoder().to(device)
    decoder.eval()

    summary_rows, grid_cache, results = [], {}, []
    grid_n = min(args.grid_pairs, n)
    alpha_labels = [f'alpha {a:.2f}' for a in alphas]

    for idx, (ckpt_path, name) in enumerate(zip(ckpt_paths, names)):
        t_start = time.time()
        state, meta = loaded.pop(idx) if idx in loaded else load_checkpoint(ckpt_path, device)
        old = meta['args']
        if old and (float(old.get('content_weight', cw)) != cw or float(old.get('style_weight', sw)) != sw):
            print(f'WARNING: {name} was trained with content/style weights '
                  f'{old.get("content_weight")}/{old.get("style_weight")}; evaluating with {cw}/{sw}.')
        decoder.load_state_dict(state)
        decoder.eval()

        ck_dir = out_root / name
        ck_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_individual_images:
            for a in alphas:
                (ck_dir / f'alpha_{a:.2f}').mkdir(exist_ok=True)

        pair_rows, quality_rows, grid_rows = [], [], []
        consistency = 0.0
        for i in range(n):
            c, s = contents[i:i + 1].to(device), styles[i:i + 1].to(device)
            with torch.no_grad():
                g, loss_c, loss_s = compute_losses(encoder, decoder, c, s, weights, None, device)
                outs = stylize_alphas(encoder, decoder, c, s, alphas)
            lc, ls = loss_c.item(), loss_s.item()
            pair_rows.append([i, cv[i], sv[i], lc, ls, ls / sw if sw else float('nan'), lc + ls])

            if 1.0 in alphas:
                diff = (g - outs[alphas.index(1.0)]).abs().max().item()
                if math.isfinite(diff):
                    consistency = max(consistency, diff)

            row_imgs = []
            for a, out in zip(alphas, outs):
                q = check_output(out)
                quality_rows.append([i, f'{a:.2f}', q['nonfinite_values'], q['raw_min'], q['raw_max'],
                                     q['frac_out_of_01'], int(q['is_black']), int(q['is_constant']),
                                     int(q['range_warn'])])
                if not args.no_individual_images:
                    save_image(torch.nan_to_num(out.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1),
                               ck_dir / f'alpha_{a:.2f}' / f'pair_{i:02d}.png')
                if i < grid_n:
                    row_imgs.append(to_pil(out))
            if i < grid_n:
                grid_rows.append([to_pil(c), to_pil(s)] + row_imgs)
                grid_cache.setdefault(name, []).append(row_imgs[alphas.index(1.0)] if 1.0 in alphas else row_imgs[-1])

        grid_path = ck_dir / 'grid_alpha_sweep.png'
        tile_grid(grid_rows, ['content', 'style'] + alpha_labels).save(grid_path)

        write_csv(ck_dir / 'per_pair_losses.csv',
                  ['pair', 'content_path', 'style_path', 'content_loss', 'style_loss_weighted',
                   'style_loss_raw', 'total_loss'], pair_rows)
        write_csv(ck_dir / 'quality_checks.csv',
                  ['pair', 'alpha', 'nonfinite_values', 'raw_min', 'raw_max', 'frac_raw_outside_0_1',
                   'is_black', 'is_constant', 'range_warning'], quality_rows)

        bench = {}
        if not args.skip_benchmark:
            bench = benchmark(encoder, decoder, args.bench_sizes, args.bench_runs, device, args.seed)

        content_losses = [r[3] for r in pair_rows]
        style_losses = [r[4] for r in pair_rows]
        style_raw = [r[5] for r in pair_rows]
        totals = [r[6] for r in pair_rows]
        n_nonfinite = sum(1 for r in quality_rows if r[2] > 0)
        n_black = sum(r[6] for r in quality_rows)
        n_constant = sum(r[7] for r in quality_rows)
        n_range = sum(r[8] for r in quality_rows)
        total_mean = mean(totals)
        integrity_ok = (n_nonfinite == 0 and n_black == 0 and n_constant == 0 and math.isfinite(total_mean))

        metrics = {
            'checkpoint': str(Path(ckpt_path).resolve()), 'name': name,
            'checkpoint_sha256_prefix': sha256_head(ckpt_path), 'iteration': meta['iteration'],
            'trained_with_seed': meta['seed'], 'full_checkpoint': meta['full_checkpoint'],
            'eval_seed': args.seed, 'eval_set_id': eval_set_id, 'manifest_dir': args.manifest_dir,
            'num_pairs': n, 'alphas': alphas, 'content_size': args.content_size,
            'final_size': args.final_size, 'precision': 'fp32', 'device': str(device),
            'torch_version': str(torch.__version__),
            'weights': {'content_weight': cw, 'style_weight': sw},
            'definitions': {
                'content_loss': 'content_weight * MSE(encoder(g)[relu4_1], AdaIN target)  (weighted)',
                'style_loss': 'style_weight * sum over relu1_1..relu4_1 of MSE(mean)+MSE(std)  (WEIGHTED, same as logged in train/val CSVs)',
                'style_loss_raw': 'style_loss / style_weight  (unweighted)',
                'total_loss': 'content_loss + style_loss, at alpha = 1',
                'alpha_operation': 'decoder(alpha * AdaIN(fc, fs) + (1 - alpha) * fc), clamp(0,1)  (same as app.py)'},
            'content_loss': mean(content_losses), 'style_loss': mean(style_losses),
            'style_loss_raw': mean(style_raw), 'total_loss': total_mean,
            'total_loss_std_over_pairs': std(totals),
            'alpha1_vs_loss_path_max_abs_diff': consistency,
            'quality': {'outputs_checked': len(quality_rows), 'with_nonfinite': n_nonfinite,
                        'black': n_black, 'constant': n_constant, 'range_warnings': n_range,
                        'integrity_ok': integrity_ok},
            'benchmark': {str(k): v for k, v in bench.items()},
            'benchmark_note': 'seeded random inputs; one full stylization per run; GPU memory = peak allocated incl. weights',
            'pairs': [{'pair': r[0], 'content': r[1], 'style': r[2]} for r in pair_rows],
            'eval_wall_seconds': round(time.time() - t_start, 1),
            'finished': datetime.now().isoformat(timespec='seconds')}
        with open(ck_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump({k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                       for k, v in metrics.items()}, f, indent=2)

        row = {'checkpoint': name, 'iteration': meta['iteration'] if meta['iteration'] is not None else '',
               'content_loss': metrics['content_loss'], 'style_loss': metrics['style_loss'],
               'style_loss_raw': metrics['style_loss_raw'], 'total_loss': total_mean,
               'n_nonfinite_outputs': n_nonfinite, 'n_black_outputs': n_black,
               'n_constant_outputs': n_constant, 'n_range_warnings': n_range,
               'integrity_ok': int(integrity_ok), 'eval_set_id': eval_set_id}
        for size in sorted(args.bench_sizes):
            r = bench.get(size)
            row[f'inference_ms_{size}'] = '' if r is None else (round(r['ms'], 2) if r['ms'] is not None else r['status'])
            row[f'peak_mem_mb_{size}'] = '' if r is None else (round(r['mem_mb'], 1) if r['mem_mb'] is not None else (r['status'] if device.type == 'cuda' else 'n/a (cpu)'))
        row['_path'] = str(Path(ckpt_path).resolve())
        row['_has_full'] = meta['full_checkpoint']
        summary_rows.append(row)
        results.append((name, metrics, state, ckpt_path))
        print(f'[{idx + 1}/{len(ckpt_paths)}] {name}: content {metrics["content_loss"]:.4f}  '
              f'style(w) {metrics["style_loss"]:.4f}  total {total_mean:.4f}  '
              f'integrity_ok={integrity_ok}  ({metrics["eval_wall_seconds"]}s)')

    eligible = [r for r in summary_rows if r['integrity_ok']]
    eligible.sort(key=lambda r: (r['total_loss'], r['checkpoint']))
    ranks = {r['checkpoint']: k + 1 for k, r in enumerate(eligible)}
    for r in summary_rows:
        r['rank_by_total_loss'] = ranks.get(r['checkpoint'], '')

    cols = [k for k in summary_rows[0] if not k.startswith('_')]
    write_csv(out_root / 'summary.csv', cols, [[r[c] for c in cols] for r in summary_rows])

    if len(ckpt_paths) > 1:
        labels = ['content', 'style'] + [nm for nm in names]
        rows = []
        for i in range(grid_n):
            rows.append([to_pil(contents[i:i + 1]), to_pil(styles[i:i + 1])] + [grid_cache[nm][i] for nm in names])
        tile_grid(rows, labels).save(out_root / 'comparison_alpha1.png')

    best_info = None
    if not args.no_best:
        best_dir = Path(args.best_dir) if args.best_dir else out_root / 'best'
        if not eligible:
            print('No checkpoint passed the integrity gate; best_validation files NOT written.')
        else:
            best_dir.mkdir(parents=True, exist_ok=True)
            win = eligible[0]
            name, metrics, state, src = next(x for x in results if x[0] == win['checkpoint'])
            best_json = best_dir / 'best_validation.json'
            existing = None
            if best_json.is_file() and not args.replace_best:
                try:
                    with open(best_json, encoding='utf-8') as f:
                        existing = json.load(f)
                except (OSError, ValueError):
                    existing = None
            write_new, replaced = True, None
            if existing:
                old_total = existing.get('selected_total_loss')
                if old_total is None and existing.get('ranking'):
                    old_total = existing['ranking'][0]['total_loss']
                comparable = (existing.get('eval_set_id') == eval_set_id
                              and existing.get('weights') == metrics['weights'] and old_total is not None)
                if not comparable:
                    write_new = False
                    print(f'WARNING: existing best in {best_dir} was selected on a different evaluation set or '
                          f'loss weights, so it is NOT comparable and was left untouched. '
                          f'Use --best_dir <other folder> or --replace_best.')
                elif win['total_loss'] < old_total:
                    replaced = {'checkpoint': existing.get('selected'), 'total_loss': old_total}
                else:
                    write_new = False
                    print(f'Existing best {existing.get("selected")} (total {old_total:.6f}) is not beaten by '
                          f'{name} (total {win["total_loss"]:.6f}); best_validation left untouched.')
            if write_new:
                if win['_has_full']:
                    atomic_copy(src, best_dir / 'best_validation.pth')
                atomic_torch_save(state, best_dir / 'best_validation_decoder.pth')
                reason = (f'{name} has the lowest mean total validation loss ({win["total_loss"]:.6f}) among '
                          f'{len(eligible)} eligible of {len(summary_rows)} checkpoint(s) evaluated in this run')
                if replaced:
                    reason += (f' and beats the previous best {replaced["checkpoint"]} '
                               f'({replaced["total_loss"]:.6f}) on the same evaluation set and weights')
                best_info = {
                    'selected': name, 'source_path': win['_path'], 'iteration': metrics['iteration'],
                    'selected_total_loss': win['total_loss'],
                    'objective': 'lowest mean total_loss (content_loss + style_loss, alpha = 1, training loss '
                                 'definitions and weights) over the fixed evaluation set',
                    'eligibility': 'integrity gate: no non-finite outputs, no all-black outputs, no constant outputs',
                    'reason': reason + '.',
                    'replaced_previous_best': replaced,
                    'weights': metrics['weights'], 'eval_set_id': eval_set_id, 'eval_seed': args.seed,
                    'num_pairs': n, 'manifest_dir': args.manifest_dir,
                    'ranking': [{'rank': k + 1, 'checkpoint': r['checkpoint'], 'iteration': r['iteration'],
                                 'total_loss': r['total_loss'], 'content_loss': r['content_loss'],
                                 'style_loss_weighted': r['style_loss']} for k, r in enumerate(eligible)],
                    'excluded_by_integrity_gate': [r['checkpoint'] for r in summary_rows if not r['integrity_ok']],
                    'caveats': ['best_validation is persistent: a later run only replaces it if it is better on the '
                                'same evaluation set and loss weights. Nothing is ever deleted.',
                                'Validation loss is a proxy for quality. Inspect grid_alpha_sweep.png before final selection.',
                                'best_validation.pth is only written when the source is a full checkpoint '
                                '(decoder-only files give best_validation_decoder.pth only).'],
                    'selected_at': datetime.now().isoformat(timespec='seconds')}
                with open(best_json, 'w', encoding='utf-8') as f:
                    json.dump(best_info, f, indent=2)
                if len(summary_rows) == 1 and not replaced:
                    print('NOTE: only one checkpoint was evaluated, so it is trivially the best of one.')

    print('\n' + '=' * 78)
    print(f'{"checkpoint":<34}{"iter":>7}{"content":>10}{"style(w)":>10}{"total":>10}  ok  rank')
    for r in summary_rows:
        print(f'{r["checkpoint"][:33]:<34}{str(r["iteration"]):>7}{r["content_loss"]:>10.4f}'
              f'{r["style_loss"]:>10.4f}{r["total_loss"]:>10.4f}  {r["integrity_ok"]:>2}  {r["rank_by_total_loss"]}')
    print('=' * 78)
    print(f'Results: {out_root.resolve()}')
    print(f'Summary: {(out_root / "summary.csv").resolve()}')
    if best_info:
        print(f'Best:    {best_info["selected"]} -> {(Path(args.best_dir) if args.best_dir else out_root / "best").resolve()}')
        print(f'Reason:  {best_info["reason"]}')


if __name__ == '__main__':
    main()
