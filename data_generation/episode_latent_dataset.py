"""Contiguous latent clips from cached episodes.

python -m data_generation.episode_latent_dataset data/cards_latents.h5 --clip-size 32
"""
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodeLatentClips(Dataset):
    """dataset[episode, start] -> float32 latents (T,N,D), aligned actions (T,).

    Episodes are relative to the chosen split; clips never cross its boundaries.
    len(dataset) counts episodes. RandomClipBatches from episode_dataset can sample
    these same two index axes. Each worker opens its own read-only HDF5 handle.
    """
    def __init__(self, path, context_size, split):
        self.path = str(Path(path).resolve())
        self.context_size = context_size
        self._file = None
        with h5py.File(self.path, 'r') as file:
            if not file.attrs.get('complete', False) or file.attrs['format_version'] != 1:
                raise ValueError('Latent dataset is incomplete or has an unsupported format')
            latents = file['latents']
            if latents.ndim != 4 or latents.dtype != np.float32:
                raise ValueError('Expected float32 latents with shape (episodes, time, tokens, dimensions)')
            n, length, *self.latent_shape = latents.shape
            if min(self.latent_shape) < 1 or file['actions'].shape != (n, length):
                raise ValueError('Invalid latent or action dimensions')
            if not 1 <= context_size <= length:
                raise ValueError('context_size must be between 1 and the saved episode length')
            boundary = int(file.attrs['train_episodes'])
            if not 0 < boundary < n or split not in ('train', 'validation'):
                raise ValueError('Invalid episode split')
            self.first, self.stop = (0, boundary) if split == 'train' else (boundary, n)
            self.starts_per_episode = length - context_size + 1
            self.dataset_id = file.attrs['dataset_id']
            self.source_dataset_id = file.attrs['source_dataset_id']
            self.checkpoint_sha256 = file.attrs['checkpoint_sha256']

    def __len__(self):
        return self.stop - self.first

    def __getitem__(self, index):
        episode, start = index
        if not (0 <= episode < len(self) and 0 <= start < self.starts_per_episode):
            raise IndexError(index)
        if self._file is None:
            self._file = h5py.File(self.path, 'r')
        episode += self.first
        clip = slice(start, start + self.context_size)
        return (torch.from_numpy(self._file['latents'][episode, clip]),
                torch.from_numpy(self._file['actions'][episode, clip]))

    def __getstate__(self):
        return {**self.__dict__, '_file': None}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_path', type=Path)
    parser.add_argument('--clip-size', type=int, default=32)
    parser.add_argument('--split', choices=['train', 'validation'], default='train')
    parser.add_argument('--episode', type=int, default=0, help='episode index within the split')
    parser.add_argument('--start', type=int, default=0)
    
    args = parser.parse_args()
    dataset = EpisodeLatentClips(args.dataset_path, args.clip_size, args.split)
    
    print()
    with h5py.File(args.dataset_path, 'r') as file:
        n = file['latents'].shape[0]
        boundary = int(file.attrs['train_episodes'])
        print(f'File: {args.dataset_path} ({args.dataset_path.stat().st_size / 2**20:.2f} MiB)')
        print(f'Latents: {file["latents"].shape} {file["latents"].dtype}')
        print(f'Episodes: {boundary} train / {n - boundary} validation')
    print(f'{args.split}: {len(dataset)} episodes × {dataset.starts_per_episode} clip starts')
    latents, actions = dataset[args.episode, args.start]

    print()
    print(f'Clip [{args.episode}, {args.start}]: latents {tuple(latents.shape)} {latents.dtype}; '
          f'actions {tuple(actions.shape)} {actions.dtype}')
    print()
    print(f'Latent range: [{latents.min().item():.6g}, {latents.max().item():.6g}]')
    print(f'Actions: {actions.tolist()}')
    print('action[t] produced the frame encoded by latents[t].')


if __name__ == '__main__':
    main()
