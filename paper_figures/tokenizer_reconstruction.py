"""Two-row sheet: saved ground-truth frames above tokenizer reconstructions.

Run from the repository root with python -m paper_figures.tokenizer_reconstruction.
Temporal tokenizers receive contiguous GT context ending at each displayed frame;
frame-independent tokenizers receive only that frame. This is reconstruction,
not an autoregressive rollout. Only the displayed pixels are clipped to [0, 1].
With --latents, decode clips from a saved latent cache instead of encoding pixels.
"""
import argparse
import hashlib
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.tokenizer import AutoEncoder, AutoEncoderConfig
from data_generation.episode_latent_dataset import EpisodeLatentClips


@torch.inference_mode()
def reconstruct(model, frames, times, context_size, device):
    """Reconstruct selected frames, one causal window at a time to bound memory."""
    predictions = []
    for time in times:
        start = max(0, time - context_size + 1)
        clip = torch.from_numpy(frames[start:time + 1]).to(device, torch.float32) / 255
        predictions.append(model(clip[None])[0, -1].cpu())
    return torch.stack(predictions).numpy()


@torch.inference_mode()
def decode_latents(model, latents, device):
    """Decode a cached (T,N,D) clip with a frame-independent tokenizer."""
    return torch.stack([model.decoder(z[None, None].to(device))[0, 0].cpu()
                        for z in latents]).numpy()


def save_sheet(frames, predictions, times, title, output, actions=None, cached=False):
    """Save aligned GT/reconstruction rows, with original episode frame indices."""
    image_width = 1.5  # inches; keep the row gap exactly one quarter of this width
    image_height = image_width * frames.shape[1] / frames.shape[2]
    row_gap = image_width / 4
    width = image_width * (len(times) + .08 * (len(times) - 1)) + 1
    height = 2 * image_height + row_gap + 1
    fig, axes = plt.subplots(2, len(times), figsize=(width, height), squeeze=False,
        gridspec_kw=dict(left=.7 / width, right=1 - .3 / width,
                         bottom=.35 / height, top=1 - .65 / height,
                         wspace=.08, hspace=row_gap / image_height))
    for column, time in enumerate(times):
        axes[0, column].imshow(frames[time], interpolation='nearest')
        axes[1, column].imshow(predictions[column].clip(0, 1), interpolation='nearest')
        heading = f't = {time}'
        if actions is not None:
            heading += f' · action {actions[time]}'
        axes[0, column].set_title(heading, fontsize=10)
        mse = np.mean((predictions[column] - frames[time].astype(np.float32) / 255) ** 2)
        axes[1, column].set_xlabel(f'MSE {mse:.3g}', fontsize=9)
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)
    axes[0, 0].set_ylabel('GT', fontsize=11)
    axes[1, 0].set_ylabel('Decoded from\nlatent cache' if cached else 'Reconstruction', fontsize=11)
    fig.suptitle(title, fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    """Load a trusted local training checkpoint and one saved RGB episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--latents', type=Path, help='decode this cache instead of encoding the original frames')
    parser.add_argument('--output', type=Path, required=True, help='high-resolution PNG file')
    parser.add_argument('--split', choices=['train', 'validation'], default='validation')
    parser.add_argument('--episode', type=int, default=0, help='zero-based index within split')
    parser.add_argument('--start', type=int, default=0, help='first displayed frame (zero-based)')
    parser.add_argument('--frames', type=int, default=8, help='number of columns')
    parser.add_argument('--stride', type=int, default=1, help='spacing of displayed frames; context stays contiguous')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    args = parser.parse_args()
    if min(args.episode, args.start) < 0 or min(args.frames, args.stride) < 1:
        parser.error('episode/start must be nonnegative; frames/stride must be positive')
    if args.output.suffix.lower() != '.png':
        parser.error('output must end in .png')
    times = args.start + np.arange(args.frames) * args.stride
    with h5py.File(args.data, 'r') as file:
        if not file.attrs.get('complete', False) or file.attrs['format_version'] != 1:
            parser.error('unsupported or incomplete dataset')
        if file.attrs['color_order'] != 'RGB' or file['frames'].dtype != np.uint8:
            parser.error('expected RGB uint8 frames')
        n, length = file['frames'].shape[:2]
        boundary = int(file.attrs['train_episodes'])
        first, stop = (0, boundary) if args.split == 'train' else (boundary, n)
        if args.episode >= stop - first or times[-1] >= length:
            parser.error('episode or displayed frames exceed the selected split/episode')
        frames = file['frames'][first + args.episode, :times[-1] + 1]
        actions = file['actions'][first + args.episode, :times[-1] + 1]
        source_id = file.attrs['dataset_id']
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    # Missing config fields denote an unbounded decoder and SSST layout.
    saved['config'].setdefault('output_sigmoid', False)
    saved['config'].setdefault('temporal_layout', 'ssst')
    config = AutoEncoderConfig(**saved['config'])
    if frames.shape[1:] != (config.img_input_H, config.img_input_W, 3):
        parser.error('dataset image dimensions do not match the tokenizer')
    context = config.max_temporal_length
    model = AutoEncoder(config).float().to(args.device).eval()
    model.load_state_dict(saved['model'])
    del saved
    if args.latents is None:
        predictions = reconstruct(model, frames, times, context, args.device)
    else:
        if config.max_temporal_length != 1:
            parser.error('latent caches require a frame-independent tokenizer')
        dataset = EpisodeLatentClips(args.latents, int(times[-1] - args.start + 1), args.split)
        with open(args.checkpoint, 'rb') as checkpoint:
            checkpoint_hash = hashlib.file_digest(checkpoint, 'sha256').hexdigest()
        if dataset.source_dataset_id != source_id or dataset.checkpoint_sha256 != checkpoint_hash:
            parser.error('latent cache does not match the supplied pixel dataset and tokenizer checkpoint')
        if (dataset.first, dataset.stop, dataset.starts_per_episode + dataset.context_size - 1) != (first, stop, length):
            parser.error('latent cache episode split or length does not match the pixel dataset')
        latents, cached_actions = dataset[args.episode, args.start]
        if not np.array_equal(cached_actions.numpy(), actions[args.start:times[-1] + 1]):
            parser.error('cached actions do not match the pixel clip')
        predictions = decode_latents(model, latents[::args.stride], args.device)
    mode = f'causal context ≤ {context}' if context > 1 else 'frame-independent'
    title = f'{args.split.capitalize()} episode {args.episode} · {mode}'
    save_sheet(frames, predictions, times, title, args.output, actions, cached=args.latents is not None)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
