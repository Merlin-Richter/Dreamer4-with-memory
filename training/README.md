# Tokenizer training details

Follow the [root walkthrough](../README.md) to generate data, train, inspect,
and cache latents. The current cards recipe is:

```sh
python -m training.train_tokenizer --data data/cards.h5 --output runs/tokenizer_cards \
  --device cuda --batch-size 256 --steps 15000 --workers 0 --seed 42
```

## Frames, sampling, and validation

Batch size counts independent frames. The pixel loader supplies length-one
clips, giving model inputs `(B, 1, H, W, 3)`. The tokenizer's temporal length is
always one. The trainer has no `--context-size` or `--temporal-attention` options.
Image dimensions come from the dataset.

Training samples frames uniformly with replacement from the training episodes.
Validation uses a fixed random subset without replacement, up to
`validation-batches * batch-size` frames. Small datasets may give fewer batches
or a short last batch. Validation MSE is weighted by pixel count. Entire episodes
are disjoint between training and validation.

`workers > 0` enables multiprocess prefetching (two batches per worker); each
worker opens its own read-only HDF5 handle. CUDA uses pinned host memory and
nonblocking transfers. `--workers 0` loads synchronously.

## Objective and saved progress

Training uses float32, a sigmoid decoder, width 128, depth 3, and AdamW with
learning rate `0.0003`. Masking and dropout are disabled.

The objective is:

```text
normalize(MSE) + 0.1 * normalize(worst_patch_MSE) + 0.2 * normalize(LPIPS)
normalize(L) = L / max(stop_gradient(abs(L)), 1e-8)
```

Worst-patch MSE is the average, over frames, of each frame's worst 8×8 patch.
LPIPS uses frozen VGG weights, while retaining gradients to the reconstruction.
Normalization is applied to each batch-mean scalar. These weights control
normalized loss terms; they do not guarantee percentages of the total gradient.

`training_loss` records the weighted sum of the **raw** terms. The individual
training and validation metrics are also raw. Inspect these and the generated
figures to follow progress. The normalized scalar itself is nearly constant for
positive losses above the normalization floor.

Evaluation runs every 500 updates by default. The output directory holds
`metrics.json` and `last.pt`, which also contains the optimizer, model settings,
dataset ID, update count, and RNG states. Only the latest checkpoint is retained.
Repeat the command with `--resume` to continue; `--steps` is the total update
count. Previously saved single-frame runs can resume under the simplified CLI.
Runs that sampled multi-frame clips require their original trainer to resume
with unchanged sampling; their checkpoints remain usable for inspection and caching.

## Recorded runs

Both setups below process 256 frames per update for 15,000 updates, or
3,840,000 frame samples. They differ in which frames are sampled together.

| Setup | Sampling | Reported A100 40 GB runtime | Results |
| --- | --- | --- | --- |
| Current recipe | 256 independently sampled frames | About 4 hours; run in progress when documented | Final metrics pending |
| Earlier reference | 32 contiguous clips of 8, flattened to 256 single-frame inputs | About 2.5 hours | Validation MSE `3.1275e-6`, LPIPS `1.4908e-4` |

The earlier run used seed 42 at commit
`43525d1f0b9cef813f4fec9cd573740ee6dc2bed` on an NVIDIA A100 PCIe 40 GB
(`galvani-cn211`, October 13, 2026). Its checkpoint confirms frame-independent
tokenization and the model/loss settings above. Validation sampled up to four
batches of 32 eight-frame clips. The inspected reconstructions preserved card
positions, colors, and symbols.

Its raw training loss plateaued around `0.04–0.05` while reconstructing card backs,
then began improving around step 2000. Later spikes recovered. Use the full
15,000-update budget and inspect the final reconstructions; robustness across
seeds has not yet been measured.

The pixel file stores compressed 16-frame chunks. Independent frame sampling
causes more separate reads than contiguous clips and may account for part of the
runtime difference; this has not been profiled. The earlier run's metrics and
runtime should not be attributed to the current sampling recipe.

Dependencies are not fully version-pinned, so these records are a reference for
settings and observed quality, not a claim of bitwise reproduction across machines.

## Vanilla dynamics

`train_dynamics.py` is a separate trainer for `models/dynamics_model.py`. It reads
only the latent cache; tokenizer weights and pixels are not needed for training.

```sh
python -m training.train_dynamics --data data/cards_latents.h5 \
  --output runs/dynamics_cards --num-actions 17 --device cuda \
  --batch-size 32 --context-size 16 --width 256 --depth 8 --steps 20000
```

This is an initial run configuration, not an established convergence budget.
The model has about 6.9 million parameters, 16 attention heads, temporal attention
every fourth layer, and no dropout. Training uses float32 and AdamW with learning
rate `0.0003`, weight decay `0.0001`, and gradient norm clipping at 1.

For clean latents `z`, sample a separate signal level `tau` for every frame,
uniformly from `{0, 1/128, ..., 127/128}`. Mix with Gaussian noise `eps`:

```text
noisy = (1 - tau) * eps + tau * z
prediction = model(noisy, tau, aligned_actions)
loss = mean((0.1 + 0.9 * tau) * (prediction - z)^2)
```

This retains the x-space flow term and ramp weighting from
[Dreamer 4, Section 3.2](https://arxiv.org/html/2509.24527v1#S3.SS2), with no
shortcut bootstrap loss or step-size conditioning. There is one scalar objective;
there are no independently weighted auxiliary losses to balance. Losses are logged
in their raw scale. Every frame is a training target, and temporal attention prevents
access to future frames or actions. With `--context-size 16`, the model sees up to
15 past frames and the current noisy target.

Validation uses a fixed subset of held-out clips and a separate, reset noise
generator. It does not consume the training RNG. The mean includes the correct
weight for a short last batch. Checkpoints contain the model, optimizer, RNG
states, settings, latent dataset ID, source pixel dataset ID, and tokenizer hash.
Only `last.pt` is retained. Resume with the same command plus `--resume`.

The existing `generate` method uses Euler integration with 64 steps per frame by
default. Supported step counts divide 128, including 8, 16, 32, and 64. These
choices still need evaluation; a lower denoising loss alone does not establish
rollout quality. The memory trainer and rollout figure scripts are separate work.
