"""Contiguous clips from saved episodes. No environment dependencies."""
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class EpisodeClips(Dataset):
    """
    Index every length-T clip, without crossing episode or split boundaries.

    The file stores (episodes, time, height, width, RGB) uint8 frames and aligned
    integer actions. The first train_episodes episodes train; the rest validate.
    Access clips as dataset[episode, start] -> (frames, actions).

    The two index axes are (episodes, length - context_size + 1); len(dataset)
    counts episodes. Frames and actions remain separate tensors.
    Each loader worker opens its own read-only HDF5 handle on first access.
    """
    def __init__(self, path, context_size, split):
        self.path = str(Path(path).resolve())
        self.context_size = context_size
        self._file = None
        with h5py.File(self.path, 'r') as file:
            if not file.attrs.get('complete', False) or file.attrs['format_version'] != 1:
                raise ValueError('Dataset is incomplete or has an unsupported format')
            if file.attrs['color_order'] != 'RGB' or file['frames'].dtype != np.uint8:
                raise ValueError('Expected RGB uint8 frames')
            n, length, *self.image_shape = file['frames'].shape
            if self.image_shape[-1] != 3 or file['actions'].shape != (n, length):
                raise ValueError('Frame/action shapes do not match')
            if not 1 <= context_size <= length:
                raise ValueError('context_size must be between 1 and the saved episode length')
            boundary = int(file.attrs['train_episodes'])
            if not 0 < boundary < n or split not in ('train', 'validation'):
                raise ValueError('Invalid episode split')
            self.first, self.stop = (0, boundary) if split == 'train' else (boundary, n)
            self.starts_per_episode = length - context_size + 1
            self.dataset_id = file.attrs['dataset_id']

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
        return (torch.from_numpy(self._file['frames'][episode, clip]),
                torch.from_numpy(self._file['actions'][episode, clip]))

    def __getstate__(self):
        # Open handles are not transferable to spawned DataLoader workers.
        return {**self.__dict__, '_file': None}


class RandomClipBatches(Sampler):
    """
    Random clips with replacement, deterministic for each optimizer step.

    Prefetching can run ahead of training. Deriving indices from (seed, step)
    lets resume restart at the consumed step without saving worker/prefetch state.
    """
    def __init__(self, episodes, starts_per_episode, batch_size, start_step, steps, seed):
        self.episodes, self.starts_per_episode = episodes, starts_per_episode
        self.batch_size = batch_size
        self.start_step, self.steps, self.seed = start_step, steps, seed

    def __iter__(self):
        for step in range(self.start_step, self.steps):
            rng = np.random.default_rng([self.seed, step])
            # Unravel uniform draws into explicit (episode, start) indices.
            # Each optimizer step has a reproducible sequence of draws.
            indices = rng.integers(0, self.episodes * self.starts_per_episode, size=self.batch_size)
            yield [divmod(int(index), self.starts_per_episode) for index in indices]

    def __len__(self):
        return max(0, self.steps - self.start_step)



if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt


    parser = argparse.ArgumentParser(description="Inspect a dataset")
    parser.add_argument("dataset-path", help="dataset file to inspect")
    parser.add_argument("--clip-size", help="clip size the dataset loader should serve", default=32)

    args = parser.parse_args()

    dataset = EpisodeClips(args.dataset_path, 32, 'train')

    print()
    print(f"Dataset shape (each clip contains {args.clip_size} time steps; different starts are overlapping):")
    print("clips shape: (" + str(len(dataset)) + " episodes, " + str(dataset.starts_per_episode) + " starts per episode, 2(frames / actions) )")

    frames, actions = dataset[0, 0]

    print()
    print(f"frames shape of clip [0, 0]: " + str(frames.shape))
    print(f"actions shape of clip [0, 0]: " + str(actions.shape))

    for x in range(32):
        plt.imshow(frames[x])
        plt.title("Action: " + str(actions[x]))
        plt.pause(1)

