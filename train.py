import argparse
import contextlib
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from utils.data_prep import load_split_manifests
from utils.models import Decoder, VGGEncoder
from utils.utils import ImageFolderDataset, adaptive_instance_normalization, calc_mean_std, get_transform

RESUME_CHECK_KEYS = ['lr', 'lr_decay', 'content_weight', 'style_weight', 'batch_size',
                     'final_size', 'content_size', 'style_size', 'amp']


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--manifest_dir', type=str, default=None,
                        help='Directory with content_train/val.txt and style_train/val.txt '
                             '(from prepare_data.py). Preferred over --content_dir/--style_dir.')
    parser.add_argument('--content_dir', type=str, default=None,
                        help='Content image folder (searched recursively); used if no --manifest_dir')
    parser.add_argument('--style_dir', type=str, default=None,
                        help='Style image folder (searched recursively); used if no --manifest_dir')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker processes (use 0 if Windows multiprocessing misbehaves)')
    parser.add_argument('--vgg', type=str, default=str(Path('weights') / 'vgg_normalised.pth'),
                        help='Location of pre-trained VGG (default: weights/vgg_normalised.pth)')
    parser.add_argument('--experiment', type=str, default='experiment1',
                        help='Name of experiment (output goes to experiment/<name>/)')

    parser.add_argument('--final_size', type=int, default=256, help='Size of final image')
    parser.add_argument('--content_size', type=int, default=512, help='Size of content image')
    parser.add_argument('--style_size', type=int, default=512, help='Size of style image')
    parser.add_argument('--crop', action='store_true', default=True, help='Crop image')

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5,
                        help='Per-iteration decay: lr = lr0 / (1 + lr_decay * iteration)')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Total number of optimizer iterations (NOT epochs)')
    parser.add_argument('--content_weight', type=float, default=1.0, help='Content weight')
    parser.add_argument('--style_weight', type=float, default=5, help='Style weight')

    parser.add_argument('--log_interval', type=int, default=100, help='Print/log losses every N iterations')
    parser.add_argument('--save_interval', type=int, default=1000, help='Write *_latest.pth every N iterations')
    parser.add_argument('--val_interval', type=int, default=1000, help='Run validation every N iterations')
    parser.add_argument('--sample_interval', type=int, default=1000, help='Save sample grid every N iterations')
    parser.add_argument('--val_pairs', type=int, default=100,
                        help='Number of fixed content/style validation pairs (kept small on purpose)')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of fixed content/style pairs in each sample grid')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='cuDNN deterministic mode (slower; default is cudnn.benchmark=True)')
    parser.add_argument('--cudnn_benchmark', type=str, default='on', choices=['on', 'off'],
                        help='cudnn.benchmark autotuning (default on, unchanged). Autotuning probes '
                             'algorithms with large temporary workspaces; --deterministic forces it off.')
    parser.add_argument('--memory_debug', action='store_true', default=False,
                        help='Print CUDA memory (allocated/reserved/peak, allocator retries and OOMs) at '
                             'the main points of the first iterations, every log interval, and after '
                             'validation and samples')
    parser.add_argument('--amp', type=str, default='off', choices=['off', 'fp16', 'bf16'],
                        help='Mixed precision on CUDA (default off). AdaIN statistics and losses '
                             'are always computed in fp32.')

    parser.add_argument('--resume', action='store_true', default=False,
                        help='Continue from experiment/<name>/checkpoint_latest.pth')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='Resume from this checkpoint instead of checkpoint_latest.pth')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Allow a fresh run in an experiment directory that already has results')

    args = parser.parse_args()
    if not args.manifest_dir and not (args.content_dir and args.style_dir):
        parser.error('provide --manifest_dir, or both --content_dir and --style_dir')
    for name in ('iterations', 'log_interval', 'save_interval', 'val_interval', 'sample_interval', 'batch_size'):
        if getattr(args, name) < 1:
            parser.error(f'--{name} must be >= 1')
    return args


def seed_everything(seed, deterministic, benchmark='on'):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = (benchmark == 'on')


class MemoryProbe:
    def __init__(self, enabled, device):
        self.on = bool(enabled) and device.type == 'cuda'
        self.active = True
        self.last_retries = 0
        self.last_ooms = 0

    def __call__(self, tag, force=False):
        if not self.on or not (self.active or force):
            return
        mb = 2 ** 20
        stats = torch.cuda.memory_stats()
        retries, ooms = stats.get('num_alloc_retries', 0), stats.get('num_ooms', 0)
        print(f'[mem] {tag:<18} alloc {torch.cuda.memory_allocated() / mb:8.1f} MB | '
              f'reserved {torch.cuda.memory_reserved() / mb:8.1f} MB | '
              f'peak alloc {torch.cuda.max_memory_allocated() / mb:8.1f} MB | '
              f'allocator retries +{retries - self.last_retries} (total {retries}) | '
              f'OOMs +{ooms - self.last_ooms} (total {ooms})', flush=True)
        self.last_retries, self.last_ooms = retries, ooms


def print_memory_summary(device, iterations_run, seconds, speeds):
    if iterations_run > 0 and seconds > 0:
        speeds = sorted(speeds)
        median = speeds[len(speeds) // 2] if speeds else float('nan')
        print(f'Throughput: {iterations_run / seconds:.2f} it/s overall (includes validation, samples, '
              f'checkpoints); median over log windows {median:.2f} it/s')
    if device.type != 'cuda':
        return
    mb = 2 ** 20
    stats = torch.cuda.memory_stats()
    total = torch.cuda.mem_get_info()[1] / mb
    print(f'GPU memory: peak allocated {torch.cuda.max_memory_allocated() / mb:.0f} MB | '
          f'peak reserved {torch.cuda.max_memory_reserved() / mb:.0f} MB | device total {total:.0f} MB | '
          f'allocator retries {stats.get("num_alloc_retries", 0)} | allocator OOMs {stats.get("num_ooms", 0)} '
          f'| cudnn.benchmark={torch.backends.cudnn.benchmark}')


def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def evenly_spaced(n, k):
    k = min(k, n)
    return [int(i * n / k) for i in range(k)]


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def get_val_transform(args):
    steps = []
    if args.content_size > 0:
        steps.append(transforms.Resize(args.content_size))
    steps.append(transforms.CenterCrop(args.final_size) if args.crop else transforms.Resize(args.final_size))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


def setup_amp(mode, device):
    if mode == 'off':
        return None, None
    if device.type != 'cuda':
        print(f'WARNING: --amp {mode} needs CUDA; running in fp32 on {device.type}.')
        return None, None
    if mode == 'bf16':
        if not torch.cuda.is_bf16_supported():
            print('WARNING: this GPU does not support bf16; running in fp32.')
            return None, None
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler('cuda')


def autocast(amp_dtype, device):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def adain_target(encoder, content, style, amp_dtype, device):
    with torch.no_grad():
        with autocast(amp_dtype, device):
            c_feats = encoder(content)
            s_feats = encoder(style)
        s_feats = [f.float() for f in s_feats]
        t = adaptive_instance_normalization(c_feats[-1].float(), s_feats[-1])
    return t, s_feats


def compute_losses(encoder, decoder, content, style, args, amp_dtype, device, probe=None):
    t, s_feats = adain_target(encoder, content, style, amp_dtype, device)
    if probe is not None:
        probe('after_encode')
    with autocast(amp_dtype, device):
        g = decoder(t)
        if probe is not None:
            probe('after_decoder')
        g_feats = encoder(g)
    if probe is not None:
        probe('after_reencode')
    g_feats = [f.float() for f in g_feats]

    loss_c = F.mse_loss(g_feats[-1], t) * args.content_weight
    loss_s = 0
    for g_f, s_f in zip(g_feats, s_feats):
        g_mean, g_std = calc_mean_std(g_f)
        s_mean, s_std = calc_mean_std(s_f)
        loss_s = loss_s + F.mse_loss(g_mean, s_mean) + F.mse_loss(g_std, s_std)
    loss_s = loss_s * args.style_weight
    return g, loss_c, loss_s


@torch.no_grad()
def validate(encoder, decoder, content_loader, style_loader, args, amp_dtype, device):
    decoder.eval()
    sums = torch.zeros(2, device=device)
    n_images = 0
    for content, style in zip(content_loader, style_loader):
        content, style = content.to(device), style.to(device)
        _, loss_c, loss_s = compute_losses(encoder, decoder, content, style, args, amp_dtype, device)
        b = content.size(0)
        sums += torch.stack([loss_c, loss_s]) * b
        n_images += b
    decoder.train()
    c, s = (sums / n_images).tolist()
    return {'total': c + s, 'content': c, 'style': s}


@torch.no_grad()
def save_samples(encoder, decoder, fixed_content, fixed_style, path, amp_dtype, device):
    decoder.eval()
    content, style = fixed_content.to(device), fixed_style.to(device)
    t, _ = adain_target(encoder, content, style, amp_dtype, device)
    with autocast(amp_dtype, device):
        g = decoder(t)
    g = g.float().clamp(0, 1)
    save_image(torch.cat([content, style, g], dim=0), path, nrow=content.size(0))
    decoder.train()


def build_checkpoint(iteration, decoder, optimizer, scaler, lr, train_losses, val_losses, args,
                     best_validation=None):
    return {
        'iteration': iteration,
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'scheduler': None,
        'lr': lr,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_validation': best_validation,
        'best_iteration': best_validation['iteration'] if best_validation else None,
        'seed': args.seed,
        'args': dict(vars(args)),
        'torch_version': str(torch.__version__),
    }


def is_improvement(best, total):
    return math.isfinite(total) and (best is None or total < best['total'])


def load_best_from_disk(save_dir):
    try:
        with open(Path(save_dir) / 'best_validation.json', encoding='utf-8') as f:
            d = json.load(f)
        best = {'iteration': int(d['best_iteration']), 'total': float(d['validation_total']),
                'content': float(d['validation_content']), 'style': float(d['validation_style'])}
        return best if math.isfinite(best['total']) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_best_files(save_dir, ckpt, args, lr, best):
    atomic_save(ckpt, save_dir / 'best_checkpoint.pth')
    atomic_save(ckpt['decoder'], save_dir / 'best_decoder.pth')
    info = {
        'best_iteration': best['iteration'],
        'validation_total': best['total'],
        'validation_content': best['content'],
        'validation_style': best['style'],
        'content_weight': args.content_weight,
        'style_weight': args.style_weight,
        'learning_rate': lr,
        'learning_rate_initial': args.lr,
        'lr_decay': args.lr_decay,
        'seed': args.seed,
        'manifest_dir': args.manifest_dir,
        'experiment': args.experiment,
        'batch_size': args.batch_size,
        'val_pairs': args.val_pairs,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }
    tmp = save_dir / 'best_validation.json.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    os.replace(tmp, save_dir / 'best_validation.json')


def open_csv(path, header, append):
    exists = Path(path).is_file() and Path(path).stat().st_size > 0
    f = open(path, 'a' if append else 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    if not (append and exists):
        writer.writerow(header)
        f.flush()
    return f, writer


def main():
    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed, args.deterministic, args.cudnn_benchmark)

    save_dir = Path('experiment') / args.experiment
    samples_dir = save_dir / 'samples'
    latest_ckpt = save_dir / 'checkpoint_latest.pth'

    if not args.resume and not args.overwrite and (latest_ckpt.exists() or (save_dir / 'train_log.csv').exists()):
        sys.exit(f'{save_dir} already contains a run. Use --resume to continue it, '
                 f'--experiment <new name> for a new run, or --overwrite to start over.')
    save_dir.mkdir(exist_ok=True, parents=True)
    samples_dir.mkdir(exist_ok=True)

    print(f'Device: {device}' + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''))

    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)

    val_ready = False
    if args.manifest_dir:
        splits = load_split_manifests(args.manifest_dir)
        content_dataset = ImageFolderDataset(splits['content_train'], content_transform)
        style_dataset = ImageFolderDataset(splits['style_train'], style_transform)
        counts = {k: len(v) for k, v in splits.items()}
        print(f'Manifests: {args.manifest_dir}')
        val_ready = counts['content_val'] > 0 and counts['style_val'] > 0
    else:
        content_dataset = ImageFolderDataset(args.content_dir, content_transform)
        style_dataset = ImageFolderDataset(args.style_dir, style_transform)
        counts = {'content_train': len(content_dataset), 'content_val': 'n/a (no manifest)',
                  'style_train': len(style_dataset), 'style_val': 'n/a (no manifest)'}
        print('WARNING: no --manifest_dir, so there are no validation images; validation and '
              'fixed samples are skipped.')
    print(f"Content train: {counts['content_train']}")
    print(f"Content val:   {counts['content_val']}")
    print(f"Style train:   {counts['style_train']}")
    print(f"Style val:     {counts['style_val']}")

    start_iter_hint = 0
    resume_path = None
    ckpt = None
    if args.resume:
        resume_path = Path(args.resume_path) if args.resume_path else latest_ckpt
        if not resume_path.is_file():
            sys.exit(f'--resume was given but no checkpoint exists at {resume_path}')
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        start_iter_hint = int(ckpt['iteration'])

    pin = device.type == 'cuda'
    persistent = args.num_workers > 0
    content_gen = torch.Generator().manual_seed(args.seed + start_iter_hint)
    style_gen = torch.Generator().manual_seed(args.seed + 1_000_003 + start_iter_hint)
    common = dict(batch_size=args.batch_size, shuffle=True, pin_memory=pin, drop_last=True,
                  num_workers=args.num_workers, persistent_workers=persistent)
    content_loader = DataLoader(content_dataset, generator=content_gen, **common)
    style_loader = DataLoader(style_dataset, generator=style_gen, **common)
    if len(content_loader) == 0 or len(style_loader) == 0:
        sys.exit('A training set has fewer images than --batch_size.')
    print(f'Batches per pass: content {len(content_loader)}, style {len(style_loader)} '
          f'(each is cycled independently; training runs exactly {args.iterations} iterations)')

    val_content_loader = val_style_loader = None
    fixed_content = fixed_style = None
    if val_ready:
        val_tf = get_val_transform(args)
        cv, sv = splits['content_val'], splits['style_val']
        n_val = min(args.val_pairs, len(cv), len(sv))
        v_c = [cv[i] for i in evenly_spaced(len(cv), n_val)]
        v_s = [sv[i] for i in evenly_spaced(len(sv), n_val)]
        val_content_loader = DataLoader(ImageFolderDataset(v_c, val_tf), batch_size=args.batch_size,
                                        shuffle=False, num_workers=0)
        val_style_loader = DataLoader(ImageFolderDataset(v_s, val_tf), batch_size=args.batch_size,
                                      shuffle=False, num_workers=0)
        n_s = min(args.num_samples, len(cv), len(sv))
        s_c = ImageFolderDataset([cv[i] for i in evenly_spaced(len(cv), n_s)], val_tf)
        s_s = ImageFolderDataset([sv[i] for i in evenly_spaced(len(sv), n_s)], val_tf)
        fixed_content = torch.stack([s_c[i] for i in range(len(s_c))])
        fixed_style = torch.stack([s_s[i] for i in range(len(s_s))])
        print(f'Validation: {n_val} fixed pairs; sample grid: {n_s} fixed pairs')

    if not Path(args.vgg).is_file():
        raise FileNotFoundError(
            f'VGG weights not found: {args.vgg}\n'
            'Pass --vgg pointing at vgg_normalised.pth, or place it at weights/vgg_normalised.pth.')
    encoder = VGGEncoder(args.vgg, map_location=device).to(device)
    decoder = Decoder().to(device)
    encoder.eval()
    decoder.train()

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    amp_dtype, scaler = setup_amp(args.amp, device)
    mem = MemoryProbe(args.memory_debug, device)
    if args.memory_debug and not mem.on:
        print('NOTE: --memory_debug needs CUDA; ignored.')
    print(f'cudnn.benchmark={torch.backends.cudnn.benchmark}  deterministic={torch.backends.cudnn.deterministic}')
    if amp_dtype is not None:
        print(f'AMP enabled: {args.amp} (AdaIN statistics and losses stay fp32)')

    start_iter = 0
    last_train, last_val = None, None
    best = None
    if ckpt is not None:
        decoder.load_state_dict(ckpt['decoder'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scaler is not None and ckpt.get('scaler') is not None:
            scaler.load_state_dict(ckpt['scaler'])
        start_iter = int(ckpt['iteration'])
        last_train, last_val = ckpt.get('train_losses'), ckpt.get('val_losses')
        in_ckpt = ckpt.get('best_validation')
        on_disk = load_best_from_disk(save_dir)
        candidates = [(b, src) for b, src in ((in_ckpt, 'checkpoint'), (on_disk, 'best_validation.json'))
                      if b and math.isfinite(b['total'])]
        if candidates:
            best, best_src = min(candidates, key=lambda c: c[0]['total'])
        old = ckpt.get('args', {})
        changed = [f'{k}: {old[k]} -> {getattr(args, k)}' for k in RESUME_CHECK_KEYS
                   if k in old and old[k] != getattr(args, k)]
        print(f'Resuming from {resume_path} at iteration {start_iter} '
              f'(next lr = {args.lr / (1.0 + args.lr_decay * start_iter):.6g})')
        if changed:
            print('WARNING: settings differ from the checkpointed run: ' + '; '.join(changed))
        if best is not None:
            print(f'Best validation restored from {best_src}: total {best["total"]:.6f} '
                  f'@ iteration {best["iteration"]}')
            if not (save_dir / 'best_checkpoint.pth').is_file():
                print('WARNING: best_checkpoint.pth is missing; it will only be recreated when validation '
                      'beats the restored best.')
        elif val_ready:
            print('No previous best validation found; best tracking starts fresh.')
        del ckpt
        if start_iter >= args.iterations:
            print(f'Checkpoint is already at iteration {start_iter} >= --iterations {args.iterations}. '
                  'Nothing to do; raise --iterations to continue training.')
            return

    with open(save_dir / 'args.txt', 'a' if args.resume else 'w') as f:
        if args.resume:
            f.write(f'\n--- resumed at iteration {start_iter} ---\n')
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')

    train_f, train_csv = open_csv(save_dir / 'train_log.csv',
                                  ['iteration', 'total', 'content', 'style', 'lr', 'it_per_sec'],
                                  append=args.resume)
    val_f, val_csv = open_csv(save_dir / 'val_log.csv',
                              ['iteration', 'total', 'content', 'style'], append=args.resume)

    def save_latest(done, lr):
        ck = build_checkpoint(done, decoder, optimizer, scaler, lr, last_train, last_val, args, best)
        atomic_save(ck, save_dir / 'checkpoint_latest.pth')
        atomic_save(decoder.state_dict(), save_dir / 'decoder_latest.pth')
        return ck

    def run_validation(done, lr):
        nonlocal last_val, best
        last_val = validate(encoder, decoder, val_content_loader, val_style_loader, args, amp_dtype, device)
        mem('after_validation', force=True)
        val_csv.writerow([done, last_val['total'], last_val['content'], last_val['style']])
        val_f.flush()
        tqdm.write(f'Validation @ iteration {done}\n'
                   f'Val Total:   {last_val["total"]:.6f}\n'
                   f'Val Content: {last_val["content"]:.6f}\n'
                   f'Val Style:   {last_val["style"]:.6f}')
        if is_improvement(best, last_val['total']):
            previous = best
            best = {'iteration': done, 'total': last_val['total'],
                    'content': last_val['content'], 'style': last_val['style']}
            ck = build_checkpoint(done, decoder, optimizer, scaler, lr, last_train, last_val, args, best)
            save_best_files(save_dir, ck, args, lr, best)
            tqdm.write(f'NEW BEST @ iteration {done}:\n'
                       f'val_total={best["total"]:.6f} (content {best["content"]:.6f}, style {best["style"]:.6f})'
                       + (f'\nprevious best: {previous["total"]:.6f} @ iteration {previous["iteration"]}'
                          if previous else '')
                       + '\nSaved best_checkpoint.pth / best_decoder.pth / best_validation.json')
        elif not math.isfinite(last_val['total']):
            tqdm.write(f'WARNING: non-finite validation total at iteration {done}; best not updated.')
        else:
            tqdm.write(f'Validation did not improve.\n'
                       f'Current: {last_val["total"]:.6f}\n'
                       f'Best: {best["total"]:.6f} @ iteration {best["iteration"]}')

    def run_samples(done):
        save_samples(encoder, decoder, fixed_content, fixed_style,
                     samples_dir / f'iter_{done:07d}.png', amp_dtype, device)
        mem('after_samples', force=True)

    print(f'Training iterations {start_iter + 1}..{args.iterations}')
    content_iter, style_iter = cycle(content_loader), cycle(style_loader)
    window = torch.zeros(3, device=device)
    window_start = start_iter
    t_window = time.time()
    done = start_iter
    lr = args.lr / (1.0 + args.lr_decay * start_iter)
    pbar = tqdm(total=args.iterations, initial=start_iter, unit='it', dynamic_ncols=True)
    speeds = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    loop_start = time.time()

    try:
        for it in range(start_iter, args.iterations):
            lr = args.lr / (1.0 + args.lr_decay * it)
            for group in optimizer.param_groups:
                group['lr'] = lr

            content = next(content_iter).to(device, non_blocking=True)
            style = next(style_iter).to(device, non_blocking=True)

            mem.active = (it - start_iter) < 2 or (it + 1) % args.log_interval == 0
            mem('iter_start')
            _, loss_c, loss_s = compute_losses(encoder, decoder, content, style, args, amp_dtype, device, probe=mem)
            loss = loss_c + loss_s

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                mem('after_backward')
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                mem('after_backward')
                optimizer.step()

            mem('after_step')
            window += torch.stack([loss.detach(), loss_c.detach(), loss_s.detach()])
            done = it + 1
            pbar.update(1)

            if done % args.log_interval == 0 or done == args.iterations:
                n = done - window_start
                total, c_avg, s_avg = (window / n).tolist()
                if not all(math.isfinite(x) for x in (total, c_avg, s_avg)):
                    raise FloatingPointError(f'Non-finite loss at iteration {done}: '
                                             f'total={total}, content={c_avg}, style={s_avg}')
                speed = n / max(time.time() - t_window, 1e-9)
                last_train = {'total': total, 'content': c_avg, 'style': s_avg}
                speeds.append(speed)
                tqdm.write(f'Iteration {done}\n'
                           f'Total: {total:.6f}\n'
                           f'Content: {c_avg:.6f}\n'
                           f'Style: {s_avg:.6f}\n'
                           f'LR: {lr:.6e}\n'
                           f'Speed: {speed:.2f} it/s')
                train_csv.writerow([done, total, c_avg, s_avg, lr, speed])
                train_f.flush()
                window.zero_()
                window_start = done
                t_window = time.time()

            if val_ready and done % args.val_interval == 0:
                run_validation(done, lr)
            if val_ready and done % args.sample_interval == 0:
                run_samples(done)
            if done % args.save_interval == 0:
                save_latest(done, lr)
                tqdm.write(f'Saved checkpoint_latest.pth / decoder_latest.pth at iteration {done}')
    except KeyboardInterrupt:
        pbar.close()
        if done > start_iter:
            save_latest(done, lr)
            print(f'Interrupted at iteration {done}. Saved checkpoint_latest.pth; '
                  f'continue with --resume --experiment {args.experiment}')
        train_f.close()
        val_f.close()
        return
    pbar.close()

    if val_ready and args.iterations % args.val_interval != 0:
        run_validation(done, lr)
    if val_ready and args.iterations % args.sample_interval != 0:
        run_samples(done)
    loop_seconds = time.time() - loop_start
    ck = save_latest(done, lr)
    atomic_save(ck, save_dir / 'checkpoint_final.pth')
    atomic_save(decoder.state_dict(), save_dir / 'decoder_final.pth')
    train_f.close()
    val_f.close()
    print(f'Finished {done} iterations. Saved final checkpoint to {save_dir / "checkpoint_final.pth"} '
          f'and decoder to {save_dir / "decoder_final.pth"}')
    print_memory_summary(device, done - start_iter, loop_seconds, speeds)
    if best is not None:
        print(f'Best validation: total {best["total"]:.6f} @ iteration {best["iteration"]} '
              f'(see {save_dir / "best_checkpoint.pth"} and best_validation.json)')


if __name__ == '__main__':
    main()