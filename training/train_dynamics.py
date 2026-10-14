"""Train vanilla causal dynamics on cached latent episodes with flow matching.

python -m training.train_dynamics --data data/cards_latents.h5 \
    --output runs/dynamics_cards --num-actions 17 --device cuda \
    --batch-size 32 --context-size 16 --steps 20000

Batch size counts clips; context size includes the current frame, so T frames
allow at most T-1 past frames. action[t] produced latent[t]: do not shift actions.
One ramp-weighted clean-latent MSE objective, no shortcut distillation. Validation
uses fixed clips and noise. Logged losses measure denoising, not rollout accuracy.
The example is an initial training budget, not a measured convergence recipe.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_generation.episode_dataset import RandomClipBatches
from data_generation.episode_latent_dataset import EpisodeLatentClips
from models.dynamics_model import DynamicsModel, DynamicsModelConfig


@torch.no_grad()
def validate(model, loader, device):
    """Repeat the same corruption draws each evaluation, independently of training."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(10002)
    total, elements = 0., 0
    for latents, actions in loader:
        latents = latents.to(device, non_blocking=device == 'cuda')
        actions = actions.to(device, torch.long, non_blocking=device == 'cuda')
        loss = model.loss(latents, actions, generator=generator)
        total += float(loss) * latents.numel()
        elements += latents.numel()
    return total / elements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='latent HDF5 cache')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-actions', type=int, required=True, help='discrete action space size; cards: 17')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--steps', type=int, required=True, help='total optimizer updates, including before resume')
    parser.add_argument('--batch-size', type=int, default=1, help='clips per update')
    parser.add_argument('--context-size', type=int, default=16, help='frames per clip, including the current frame')
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--validation-batches', type=int, default=4)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.num_actions, args.validation_batches, args.eval_every) < 1:
        parser.error('steps, batch size, action count, validation batches and eval-every must be positive')
    if args.context_size < 2 or args.depth < 4 or args.width < 32 or args.width % 32:
        parser.error('need context-size >= 2, depth >= 4 and width divisible by 32 (16 attention heads)')
    if args.workers < 0 or args.seed < 0 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('need nonnegative workers/seed and a finite positive learning rate')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; run inside a GPU allocation or use --device cpu')
    checkpoint = args.output / 'last.pt'
    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be empty for a new run; use --resume to continue')
    if args.resume and not checkpoint.exists():
        parser.error('--resume requires output/last.pt')

    train_data = EpisodeLatentClips(args.data, args.context_size, 'train')
    val_data = EpisodeLatentClips(args.data, args.context_size, 'validation')
    with h5py.File(args.data, 'r') as file:
        actions = file['actions'][:]
        if not np.issubdtype(actions.dtype, np.integer) or actions.min() < 0 or actions.max() >= args.num_actions:
            parser.error('cached actions must be integer ids in [0, num-actions)')
    del actions
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = DynamicsModelConfig(dtype=torch.float32, embedding_dim=args.width,
        depth=args.depth, max_temporal_length=args.context_size, n_actions=args.num_actions,
        n_latents=train_data.latent_shape[0], bottleneck_dim=train_data.latent_shape[1])
    model = DynamicsModel(config).to(args.device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_step, history = 0, []
    if args.resume:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved.get('objective') != 'flow_matching_x_ramp_v1' or saved['config'] != asdict(config):
            parser.error('checkpoint model configuration or objective changed')
        if saved['dataset_id'] != train_data.dataset_id or saved['tokenizer_sha256'] != train_data.checkpoint_sha256:
            parser.error('checkpoint latent dataset or tokenizer changed')
        for key in ('batch_size', 'validation_batches', 'lr', 'seed', 'device'):
            if saved['arguments'][key] != vars(args)[key]:
                parser.error(f'{key} changed since the checkpoint')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_step, history = saved['step'], saved['history']
        del saved
        if args.steps <= start_step:
            parser.error(f'checkpoint is already at step {start_step}; --steps must be larger')

    # Loader RNGs are separate from training noise; each worker opens its own file.
    options = dict(num_workers=args.workers, pin_memory=args.device == 'cuda',
        persistent_workers=args.workers > 0,
        multiprocessing_context='spawn' if args.workers else None,
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
    print(f'{args.device}: {args.batch_size} clips × {args.context_size} frames; '
          f'latents {tuple(train_data.latent_shape)}; {args.num_actions} actions; '
          f'{sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
    print('Flow matching, float32, no shortcut distillation. Validation is denoising loss.', flush=True)
    for step, (latents, actions) in enumerate(train_loader, start_step + 1):
        model.train()
        latents = latents.to(args.device, non_blocking=args.device == 'cuda')
        actions = actions.to(args.device, torch.long, non_blocking=args.device == 'cuda')
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(latents, actions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            record = dict(step=step, training_loss=float(loss.detach()),
                          validation_loss=validate(model, val_loader, args.device))
            print(json.dumps(record, allow_nan=False), flush=True)
            history.append(record)
            saved = dict(config=asdict(config), arguments=vars(args), step=step,
                objective='flow_matching_x_ramp_v1', dataset_id=train_data.dataset_id,
                source_dataset_id=train_data.source_dataset_id,
                tokenizer_sha256=train_data.checkpoint_sha256, validation_clips=val_count,
                model=model.state_dict(), optimizer=optimizer.state_dict(), history=history,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else None)
            temporary = args.output / 'last.tmp'
            torch.save(saved, temporary)
            temporary.replace(checkpoint)
            (args.output / 'metrics.json').write_text(json.dumps(history, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
