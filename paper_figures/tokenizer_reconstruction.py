"""Two-row sheet: saved ground-truth frames above tokenizer reconstructions.

Run from the repository root with python -m paper_figures.tokenizer_reconstruction.
Temporal tokenizers receive contiguous GT context ending at each displayed frame;
frame-independent tokenizers receive only that frame. This is reconstruction,
not an autoregressive rollout. Only the displayed pixels are clipped to [0, 1].
"""
import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.tokenizer import AutoEncoder, AutoEncoderConfig


@torch.inference_mode()
def reconstruct(model, frames, times, context_size, device):
    """Reconstruct selected frames, one causal window at a time to bound memory."""
    predictions = []
    for time in times:
        start = max(0, time - context_size + 1)
        clip = torch.from_numpy(frames[start:time + 1]).to(device, torch.float32) / 255
        predictions.append(model(clip[None])[0, -1].cpu())
    return torch.stack(predictions).numpy()


def save_sheet(frames, predictions, times, title, output):
    """Save aligned GT/reconstruction rows, with original episode frame indices."""
    fig, axes = plt.subplots(2, len(times), figsize=(1.5 * len(times) + 1, 3.6), squeeze=False)
    for column, time in enumerate(times):
        axes[0, column].imshow(frames[time], interpolation='nearest')
        axes[1, column].imshow(predictions[column].clip(0, 1), interpolation='nearest')
        axes[0, column].set_title(f't = {time}', fontsize=10)
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)
    axes[0, 0].set_ylabel('GT', fontsize=11)
    axes[1, 0].set_ylabel('Reconstruction', fontsize=11)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(pad=0.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    """Load a trusted local training checkpoint and one saved RGB episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
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
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    # Missing config fields denote an unbounded decoder and SSST layout.
    saved['config'].setdefault('output_sigmoid', False)
    saved['config'].setdefault('temporal_layout', 'ssst')
    config = AutoEncoderConfig(**saved['config'])
    if frames.shape[1:] != (config.img_input_H, config.img_input_W, 3):
        parser.error('dataset image dimensions do not match the tokenizer')
    context = saved['arguments']['context_size'] if saved['arguments']['temporal_attention'] else 1
    model = AutoEncoder(config).float().to(args.device).eval()
    model.load_state_dict(saved['model'])
    del saved
    predictions = reconstruct(model, frames, times, context, args.device)
    mode = f'causal context ≤ {context}' if context > 1 else 'frame-independent'
    title = f'{args.split.capitalize()} episode {args.episode} · {mode}'
    save_sheet(frames, predictions, times, title, args.output)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
