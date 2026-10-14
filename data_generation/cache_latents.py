"""Encode every saved RGB frame into a float32 latent sequence.

python -m data_generation.cache_latents --data data/cards.h5 \
    --checkpoint runs/tokenizer/last.pt --output data/cards_latents.h5 --device cuda

Requires a frame-independent tokenizer. Each frame is encoded with T=1;
batch size only controls processing memory. Actions keep their original alignment:
action[t] produced frame[t], and latents[t] encodes that frame.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import uuid

import h5py
import numpy as np
import torch

from models.tokenizer import AutoEncoder, AutoEncoderConfig


@torch.inference_mode()
def cache_latents(data_path, checkpoint_path, output_path, batch_size=64, device='cpu'):
    """Write dense, uncompressed episode latents; publish only the completed file."""
    output_path = Path(output_path)
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    if output_path.exists() or temporary.exists():
        raise FileExistsError('Output or temporary file already exists; choose a new path')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    saved['config'].setdefault('output_sigmoid', False)
    saved['config'].setdefault('temporal_layout', 'ssst')
    config = AutoEncoderConfig(**saved['config'])
    if config.max_temporal_length != 1:
        raise ValueError('Latent caching requires a frame-independent tokenizer (max_temporal_length=1)')
    model = AutoEncoder(config).float().to(device).eval().requires_grad_(False)
    model.load_state_dict(saved['model'])
    step = saved['step']
    del saved
    with open(checkpoint_path, 'rb') as checkpoint:
        checkpoint_hash = hashlib.file_digest(checkpoint, 'sha256').hexdigest()

    with h5py.File(data_path, 'r') as source:
        if not source.attrs.get('complete', False) or source.attrs['format_version'] != 1:
            raise ValueError('Pixel dataset is incomplete or has an unsupported format')
        frames = source['frames']
        if frames.ndim != 5 or frames.dtype != np.uint8 or source.attrs['color_order'] != 'RGB':
            raise ValueError('Expected RGB uint8 frames with shape (episodes, time, H, W, 3)')
        n, length, h, w, channels = frames.shape
        if (h, w, channels) != (config.img_input_H, config.img_input_W, 3):
            raise ValueError('Pixel dimensions do not match the tokenizer')
        boundary = int(source.attrs['train_episodes'])
        if length < 1 or not 0 < boundary < n:
            raise ValueError('Need nonempty episodes and at least one episode per split')
        if source['actions'].shape != (n, length) or source['episode_seeds'].shape != (n,):
            raise ValueError('Actions or episode seeds do not match the frame dimensions')
        shape = (n, length, config.n_latents, config.bottleneck_dim)
        print(f'Checkpoint step {step} · {device} · {batch_size} frames/batch', flush=True)
        print(f'Pixels: {frames.shape} uint8 → latents: {shape} float32', flush=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(temporary, 'x') as target:
            target.attrs.update(format_version=1, dataset_id=str(uuid.uuid4()), complete=False,
                source_dataset_id=source.attrs['dataset_id'], train_episodes=boundary,
                checkpoint_sha256=checkpoint_hash, checkpoint_step=step,
                tokenizer_config=json.dumps({**asdict(config), 'dtype': str(config.dtype)}))
            source.copy('actions', target)
            source.copy('episode_seeds', target)
            latents = target.create_dataset('latents', shape, dtype='float32')
            for episode in range(n):
                for start in range(0, length, batch_size):
                    stop = min(start + batch_size, length)
                    pixels = torch.from_numpy(frames[episode, start:stop]).to(device, torch.float32) / 255
                    latents[episode, start:stop] = model.encoder(pixels[:, None])[:, 0].cpu().numpy()
                if (episode + 1) % max(1, n // 20) == 0 or episode + 1 == n:
                    print(f'{episode + 1}/{n} episodes written', flush=True)
            target.attrs['complete'] = True
    temporary.replace(output_path)
    print(f'Saved {output_path}: {output_path.stat().st_size / 2**20:.2f} MiB; '
          f'{boundary} train / {n - boundary} validation episodes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64, help='frames per encoder batch (default: 64)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cache_latents(args.data, args.checkpoint, args.output, args.batch_size, args.device)


if __name__ == '__main__':
    main()
