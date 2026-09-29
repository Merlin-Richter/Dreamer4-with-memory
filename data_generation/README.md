# Saved episodes

From the repository root, generate a dataset once:

```sh
python -m pip install -r requirements.txt
python -m data_generation.generate_episodes --task cards --episodes 1000 --length 128 --output data/cards.h5
python -m data_generation.generate_episodes --task ball --episodes 1000 --length 128 --output data/ball.h5
```

`--episodes N` is the total episode count. By default, the last 10% (at least 1) are validation episodes; set `--validation-episodes` explicitly if desired. Each split contains whole episodes. Episode seeds and action/environment RNG streams are independent of model training. No balancing or action resampling is performed. Existing files are never overwritten.

HDF5 layout:

- `frames`: `(N,L,H,W,3)`, RGB `uint8` in 0..255; lossless gzip compression, chunked within episodes.
- `actions`: `(N,L)`, `int64`; action[t] produced frame[t]. Reset frames are omitted.
- `episode_seeds`: `(N,)`, `uint64`; reproduce an episode with `generate_episode(task, seed, L)`.
- Attributes: dataset ID, completed flag, format version, RGB convention, train/validation boundary, generation settings, source hashes.

Only one episode is generated in RAM at a time. Pixels are stored as bytes; saving normalized float32 would require 4× the uncompressed pixel storage. Conversion to floats in 0..1 happens in the trainer. Keep completed files immutable; checkpoints record the dataset ID (an identifier, not a checksum of the contents).


Reading data:

Create `EpisodeClips(path, context_size, split)`, where `split` is `'train'` or `'validation'`.
Retrieve a clip with `frames, actions = dataset[episode, start]`. Increase `start` by 1 for the next overlapping window, or by `context_size` for the next non-overlapping clip, within the episode bounds.
The index axes are `(episodes, L-T+1)`.
Each clip returns frames `(T,H,W,3)` and actions `(T,)`; the pair is not a shared tensor axis because their shapes and types differ.

Using HDF5 for storing datasets. Documentation: https://docs.h5py.org/en/stable/high/dataset.html and https://docs.pytorch.org/docs/stable/data.html .
