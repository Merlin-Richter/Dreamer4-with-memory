"""Train an RGB tokenizer with separately normalized MSE, worst-patch MSE and LPIPS.

Objective: norm(MSE) + 0.1 norm(worst-patch MSE) + lpips_weight norm(LPIPS),
where norm(L) = L / max(stop_gradient(abs(L)), 1e-8). FP32 throughout.

Example: python -m training.train_tokenizer --data data/cards.h5 --output runs/tokenizer_cards
Batch size counts independently sampled frames. Each frame is encoded with T=1;
masking and dropout are off. No environment imports.
Resume with --resume; --steps is the total desired optimizer update count.
Logged losses are raw, before normalization. --lpips-weight 0 disables LPIPS.
LPIPS uses frozen VGG weights;
the first run downloads them to TORCH_HOME (PyTorch's default cache if unset).
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_generation.episode_dataset import EpisodeClips, RandomClipBatches
from models.tokenizer import AutoEncoder, AutoEncoderConfig


def tensor(frames, device):
    """Normalize the loader's (B,1,H,W,3) RGB bytes to floats in [0,1]."""
    return frames.to(device=device, dtype=torch.float32, non_blocking=(device == 'cuda')) / 255


def normalize_loss(loss):
    """Normalize a batch-mean scalar; only the numerator receives gradients."""
    return loss / loss.detach().abs().clamp_min(1e-8)


def reconstruction_losses(prediction, target, patch_size):
    """Return raw MSE and mean worst-patch MSE per frame, without object labels."""
    error = (prediction - target).square()
    h, w, channels = error.shape[-3:]
    patches = error.reshape(-1, h // patch_size, patch_size, w // patch_size, patch_size, channels)
    patch_mse = patches.mean(dim=(2, 4, 5))
    mse = error.mean()
    return mse, patch_mse.flatten(1).max(dim=1).values.mean()


def perceptual_loss(metric, prediction, target):
    """Evaluate LPIPS per frame: (B,T,H,W,3) [0,1] -> (B*T,3,H,W)."""
    prediction = prediction.flatten(0, 1).permute(0, 3, 1, 2)
    target = target.flatten(0, 1).permute(0, 3, 1, 2)
    return metric(prediction, target, normalize=True).mean()


def main():
    """Train on normalized terms; record raw losses for interpretable progress."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--steps', type=int, default=15000, help='total optimizer updates (default: 15000)')
    parser.add_argument('--batch-size', type=int, default=1, help='independent frames per update; cards recipe: 256')
    parser.add_argument('--workers', type=int, default=0, help='prefetch worker processes; 0 loads synchronously with less RAM')
    parser.add_argument('--validation-batches', type=int, default=4, help='maximum validation batches; fixed frames without replacement')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lpips-weight', type=float, default=0.2, help='frozen VGG LPIPS weight; 0 disables it')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.batch_size, args.validation_batches, args.steps, args.depth, args.eval_every) < 1:
        parser.error('batch/validation sizes, steps, depth and eval-every must be positive')
    if args.workers < 0 or args.seed < 0 or args.width < 8 or args.width % 8 or args.lr <= 0:
        parser.error('nonnegative workers/seed, positive lr and width divisible by 8 required')
    if not np.isfinite(args.lpips_weight) or args.lpips_weight < 0:
        parser.error('lpips-weight must be finite and nonnegative')
    checkpoint = args.output / 'last.pt'
    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be empty for a new run; use --resume to continue')
    if args.resume and not checkpoint.exists():
        parser.error('--resume requires output/last.pt')

    print(f'{args.device}: {args.batch_size} independent frames/update; workers={args.workers}', flush=True)

    # Entire episodes are disjoint between training and validation.
    train_data = EpisodeClips(args.data, 1, 'train')
    val_data = EpisodeClips(args.data, 1, 'validation')
    height, width, _ = train_data.image_shape
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = AutoEncoderConfig(dtype=torch.float32, embedding_dim=args.width,
        depth=args.depth, n_heads=4,
        max_temporal_length=1,
        img_input_H=height, img_input_W=width, patch_size=8,
        n_latents=4, bottleneck_dim=64, drop_rate=0., att_drop_rate=0.,
        mae_min_mask=0., mae_max_mask=0.)
    model = AutoEncoder(config).to(args.device).float()
    perceptual = None
    if args.lpips_weight:
        import lpips
        # Freeze the metric, not the gradient path from the loss to the tokenizer.
        perceptual = lpips.LPIPS(net='vgg', version='0.1', verbose=False).to(args.device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_step, history = 0, []
    if args.resume:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved.get('loss_normalization') == 'detached_abs_1e-8', 'Loss normalization changed; start a new run'
        assert saved['config'] == asdict(config), 'Model configuration changed'
        assert saved['arguments'].get('loss_patch_size', config.patch_size) == config.patch_size, 'Checkpoint loss patch size does not match the tokenizer patch size'
        assert saved['dataset_id'] == train_data.dataset_id, 'Dataset changed'
        assert saved['arguments'].get('lpips_weight', 0.) == args.lpips_weight, 'LPIPS weight changed; start a new run'
        assert saved['arguments'].get('context_size', 1) == 1, 'Checkpoint sampled multi-frame clips; start a new run'
        for key in ('validation_batches', 'batch_size', 'lr', 'seed', 'device'):
            assert saved['arguments'].get(key) == vars(args)[key], f'{key} changed'
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        if args.device == 'mps':
            torch.mps.set_rng_state(saved['mps_rng'])
        start_step, history = saved['step'], saved['history']
        del saved

    # Each worker opens its own HDF5 file. Spawn avoids inheriting CUDA/HDF5 state.
    # Separate loader RNGs leave the model's RNG untouched by worker startup.
    options = dict(num_workers=args.workers, pin_memory=args.device == 'cuda',
        persistent_workers=args.workers > 0,
        multiprocessing_context='spawn' if args.workers else None,
        prefetch_factor=2 if args.workers else None)
    batches = RandomClipBatches(len(train_data), train_data.starts_per_episode, args.batch_size, start_step, args.steps, args.seed)
    train_loader = DataLoader(train_data, batch_sampler=batches,
        generator=torch.Generator().manual_seed(args.seed), **options)
    available_frames = len(val_data) * val_data.starts_per_episode
    val_count = min(available_frames, args.validation_batches * args.batch_size)
    draws = np.random.default_rng(10001).choice(available_frames, val_count, replace=False)
    val_indices = [divmod(int(index), val_data.starts_per_episode) for index in draws]
    val_loader = DataLoader(Subset(val_data, val_indices), batch_size=args.batch_size,
        generator=torch.Generator().manual_seed(10001), **options)

    for step, (frames, _) in enumerate(train_loader, start_step + 1):
        model.train()
        target = tensor(frames, args.device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(target)
        mse, worst_patch = reconstruction_losses(prediction, target, config.patch_size)
        loss = normalize_loss(mse) + 0.1 * normalize_loss(worst_patch)
        raw_loss = mse.detach() + 0.1 * worst_patch.detach()
        if perceptual is not None:
            lpips_value = perceptual_loss(perceptual, prediction, target)
            loss = loss + args.lpips_weight * normalize_loss(lpips_value)
            raw_loss = raw_loss + args.lpips_weight * lpips_value.detach()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            # Fixed validation frames, loaded batchwise. Weight a short last batch
            # by its pixel count, so this is MSE over all validation frames.
            model.eval()
            squared_error, elements, perceptual_sum = 0., 0, 0.
            with torch.no_grad():
                for val_frames, _ in val_loader:
                    val_target = tensor(val_frames, args.device)
                    val_prediction = model(val_target)
                    squared_error += float((val_prediction - val_target).square().sum())
                    if perceptual is not None:
                        perceptual_sum += float(perceptual_loss(perceptual, val_prediction, val_target)) * val_target.numel()
                    elements += val_target.numel()
            record = dict(step=step, training_loss=float(raw_loss), training_mse=float(mse.detach()),
                training_worst_patch_mse=float(worst_patch.detach()), validation_mse=squared_error/elements)
            if perceptual is not None:
                record.update(training_lpips=float(lpips_value.detach()), validation_lpips=perceptual_sum/elements)
            history.append(record)
            print(json.dumps(record, allow_nan=False), flush=True)
            saved = dict(config=asdict(config), arguments=vars(args), step=step,
                loss_normalization='detached_abs_1e-8',
                dataset_id=train_data.dataset_id, validation_frames=val_count,
                model=model.state_dict(), optimizer=optimizer.state_dict(), history=history,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else None,
                mps_rng=torch.mps.get_rng_state() if args.device == 'mps' else None)
            # Only the last checkpoint is retained; publish it after a successful write.
            temporary = args.output / 'last.tmp'
            torch.save(saved, temporary)
            temporary.replace(checkpoint)
            (args.output / 'metrics.json').write_text(json.dumps(history, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
