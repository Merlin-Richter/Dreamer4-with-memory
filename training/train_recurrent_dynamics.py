"""Train bounded memory readers/writers on a frozen long-context state backbone.

Real cached trajectories are unrolled through successive blocks, without
detaching memory. The frozen parent supplies optional prediction targets only.
Loss terms are independently divided by detached absolute magnitude; logged
MSEs and checkpoint selection use raw, interpretable prediction errors.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_generation.episode_dataset import RandomClipBatches
from data_generation.episode_latent_dataset import EpisodeLatentClips
from models.long_context_dynamics_model import LongContextDynamicsConfig, LongContextDynamicsModel
from models.recurrent_dynamics_model import RecurrentDynamicsModel


def normalized_loss(value):
    return value / value.detach().abs().clamp_min(1e-8)


def losses(model, teacher, clean, actions, teacher_weight=.1, generator=None):
    """Same corruption for parent/student; clean frames are context, not targets."""
    shape, device = clean.shape[:2], clean.device
    mixture = torch.rand(shape, device=device, generator=generator)
    middle = torch.randint(1, 127, shape, device=device, generator=generator)
    tau = torch.where(mixture < .5, 127, torch.where(mixture < .75, 0, middle))
    noise = torch.randn(clean.shape, device=device, generator=generator)
    amount = tau.float()[..., None, None] / 127
    noisy = amount * clean + (1 - amount) * noise
    mask = tau < 127
    def mse(prediction, target):
        error = (prediction.float() - target.float()).square().mean((-1, -2))
        return (error * mask).sum() / mask.sum().clamp_min(1)
    # Only teacher targets are detached. Frozen student backbone operations
    # must remain in the autograd graph so future errors can train earlier writes.
    if teacher_weight:
        with torch.no_grad():
            target = teacher(noisy, tau, actions)
    prediction = model(noisy, tau, actions)
    observation = mse(prediction, clean)
    distillation = mse(prediction, target) if teacher_weight else observation.new_zeros(())
    objective = normalized_loss(observation) + teacher_weight * normalized_loss(distillation)
    return objective, observation, distillation


@torch.no_grad()
def validate(model, teacher, loader, device, teacher_weight):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(10002)
    total = np.zeros(2)
    count = 0
    for latents, actions in loader:
        latents = latents.to(device, non_blocking=device == 'cuda')
        actions = actions.to(device, torch.long, non_blocking=device == 'cuda')
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            _, observation, distillation = losses(model, teacher, latents, actions, teacher_weight, generator)
        total += np.array([float(observation), float(distillation)]) * latents.shape[0]
        count += latents.shape[0]
    return dict(validation_mse=total[0] / count, validation_teacher_mse=total[1] / count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--steps', type=int, default=10000, help='total updates, including before resume')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--context-size', type=int, default=256, help='training unroll; not an inference memory limit')
    parser.add_argument('--block-size', type=int, default=16)
    parser.add_argument('--memory-slots', type=int, default=16, help='persistent vectors per layer')
    parser.add_argument('--teacher-weight', type=float, default=.1, help='weight on separately normalized teacher MSE; 0 disables it')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--validation-batches', type=int, default=4)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.context_size, args.block_size, args.memory_slots,
           args.validation_batches, args.eval_every) < 1:
        parser.error('steps, batch/context/block sizes, slots and evaluation counts must be positive')
    if args.context_size % args.block_size or args.context_size < 2 * args.block_size:
        parser.error('training context must contain at least two complete blocks')
    if (args.workers < 0 or args.seed < 0 or not np.isfinite(args.lr) or args.lr <= 0
            or not np.isfinite(args.teacher_weight) or args.teacher_weight < 0):
        parser.error('need nonnegative workers/seed/teacher-weight and a finite positive learning rate')
    if args.device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        parser.error('CUDA training requires a BF16-capable GPU allocation')
    checkpoint = args.output / 'last.pt'
    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be empty for a new run; use --resume to continue')
    if args.resume and not checkpoint.exists():
        parser.error('--resume requires output/last.pt')
    parent = torch.load(args.parent, map_location='cpu', weights_only=False)
    if parent.get('architecture') != 'long_context_state_v1' or parent.get('objective') != 'denoising_x_mixture_v1':
        parser.error('parent must be a trained long-context state-history checkpoint')
    config = LongContextDynamicsConfig(**parent['config'])
    if args.context_size > config.max_temporal_length:
        parser.error('training unroll exceeds the parent context; use a parent trained on longer histories')
    train_data = EpisodeLatentClips(args.data, args.context_size, 'train')
    val_data = EpisodeLatentClips(args.data, args.context_size, 'validation')
    if tuple(train_data.latent_shape) != (config.n_latents, config.latent_dim):
        parser.error('cache latent shape differs from parent')
    if train_data.checkpoint_sha256 != parent['tokenizer_sha256'] or train_data.dataset_id != parent['dataset_id']:
        parser.error('use the parent training cache, with the same tokenizer and episode split')
    with h5py.File(args.data, 'r') as file:
        actions = file['actions'][:]
        if not np.issubdtype(actions.dtype, np.integer) or actions.min() < 0 or actions.max() >= config.n_actions:
            parser.error('cached actions fall outside the parent action space')
    del actions
    with args.parent.open('rb') as file:
        parent_sha256 = hashlib.file_digest(file, 'sha256').hexdigest()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    teacher = LongContextDynamicsModel(config, recompute=False).to(args.device).float().eval()
    teacher.load_state_dict(parent['model'])
    teacher.requires_grad_(False)
    model = RecurrentDynamicsModel(config, block_size=args.block_size, memory_slots=args.memory_slots).to(args.device).float()
    model.initialize(parent['model'])
    del parent
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=.01)
    start_step, history, best_mse, elapsed_before = 0, [], float('inf'), 0.
    if args.resume:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (saved.get('architecture') != 'recurrent_state_v1'
                or saved.get('objective') != 'normalized_denoising_distillation_v1'
                or saved.get('loss_normalization') != 'detached_abs_1e-8'
                or saved['config'] != asdict(config) or saved['parent_sha256'] != parent_sha256
                or saved['dataset_id'] != train_data.dataset_id):
            parser.error('resume model, objective, parent or dataset changed')
        for key in ('batch_size', 'context_size', 'block_size', 'memory_slots', 'teacher_weight',
                    'validation_batches', 'lr', 'seed', 'device'):
            if saved['arguments'][key] != vars(args)[key]:
                parser.error(f'{key} changed since the checkpoint')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_step, history, best_mse = saved['step'], saved['history'], saved['best_validation_mse']
        elapsed_before = history[-1]['elapsed_seconds']
        del saved
        if args.steps <= start_step:
            parser.error(f'checkpoint is already at step {start_step}; --steps must be larger')
    options = dict(num_workers=args.workers, pin_memory=args.device == 'cuda',
        persistent_workers=args.workers > 0, multiprocessing_context='spawn' if args.workers else None,
        prefetch_factor=2 if args.workers else None)
    batches = RandomClipBatches(len(train_data), train_data.starts_per_episode,
        args.batch_size, start_step, args.steps, args.seed)
    train_loader = DataLoader(train_data, batch_sampler=batches,
        generator=torch.Generator().manual_seed(args.seed), **options)
    windows = len(val_data) * val_data.starts_per_episode
    val_count = min(windows, args.validation_batches * args.batch_size)
    draws = np.random.default_rng(10001).choice(windows, val_count, replace=False)
    indices = [divmod(int(index), val_data.starts_per_episode) for index in draws]
    val_loader = DataLoader(Subset(val_data, indices), batch_size=args.batch_size,
        generator=torch.Generator().manual_seed(10001), **options)
    args.output.mkdir(parents=True, exist_ok=True)
    memory_bytes = config.depth * args.memory_slots * config.width * 4
    print(f'{args.device}: {args.batch_size} episodes × {args.context_size} frames; '
          f'{args.context_size // args.block_size} blocks of {args.block_size}; '
          f'{args.memory_slots} slots/layer; {memory_bytes / 1024:g} KiB persistent memory/episode (FP32).', flush=True)
    print(f'{sum(p.numel() for p in trainable):,} trainable reader/writer parameters; backbone and teacher frozen. '
          'Full recurrent gradients, block activation recomputation; no detached memory.', flush=True)
    print(f'Loss: normalize(prediction MSE) + {args.teacher_weight:g} × normalize(teacher MSE); '
          'normalize(x)=x/stopgrad(max(abs(x),1e-8)). Logs and best.pt use raw MSE.', flush=True)
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    started = time.perf_counter()
    for step, (latents, actions) in enumerate(train_loader, start_step + 1):
        model.train()
        latents = latents.to(args.device, non_blocking=args.device == 'cuda')
        actions = actions.to(args.device, torch.long, non_blocking=args.device == 'cuda')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=args.device == 'cuda'):
            loss, observation, distillation = losses(model, teacher, latents, actions, args.teacher_weight)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps or stop_requested:
            record = dict(step=step, training_mse=float(observation.detach()),
                training_teacher_mse=float(distillation.detach()), grad_norm=float(grad_norm),
                **validate(model, teacher, val_loader, args.device, args.teacher_weight),
                elapsed_seconds=elapsed_before + time.perf_counter() - started)
            if args.device == 'cuda':
                record['peak_vram_gib'] = torch.cuda.max_memory_allocated() / 2**30
            print(json.dumps(record, allow_nan=False), flush=True)
            history.append(record)
            improved = record['validation_mse'] < best_mse
            best_mse = min(best_mse, record['validation_mse'])
            saved = dict(config=asdict(config), arguments=vars(args), step=step,
                architecture='recurrent_state_v1', objective='normalized_denoising_distillation_v1',
                loss_normalization='detached_abs_1e-8', parent_sha256=parent_sha256,
                dataset_id=train_data.dataset_id, source_dataset_id=train_data.source_dataset_id,
                tokenizer_sha256=train_data.checkpoint_sha256, validation_clips=val_count,
                memory_bytes_fp32=memory_bytes, best_validation_mse=best_mse,
                torch_version=str(torch.__version__), precision='bf16' if args.device == 'cuda' else 'float32',
                model=model.state_dict(), optimizer=optimizer.state_dict(), history=history,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else None)
            for name in ('last', 'best') if improved else ('last',):
                temporary = args.output / f'{name}.tmp'
                torch.save(saved, temporary)
                temporary.replace(args.output / f'{name}.pt')
            (args.output / 'metrics.json').write_text(json.dumps(history, indent=2, allow_nan=False) + '\n')
        if stop_requested:
            print(f'Saved step {step}; continue with --resume.', flush=True)
            break


if __name__ == '__main__':
    main()
