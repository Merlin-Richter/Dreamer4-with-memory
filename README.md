# Dreamer4 with Memory

This bachelor-thesis project studies how a visual world model can remember
information that is no longer visible. A world model predicts what an environment
will look like after an action. When the relevant information disappeared many
steps ago, making a consistent prediction requires remembering it.

The walkthrough starts with an environment where you can see this requirement
for yourself. You will watch recorded episodes, train a model to compress and
reconstruct their images, and inspect the representations that will be used to
learn future predictions.

## 1. Set up the environment

Use Python 3.11 or newer. Run all commands from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```


## 2. See why the environment needs memory

The learning pipeline works with recorded trajectories: sequences of images and
the actions that produced them. You can use trajectories from your own environment
by saving them in the [same episode format](data_generation/README.md). For this
walkthrough, we use a memory-card environment because it makes both remembering
and forgetting easy to see.

There are 16 cards arranged in a 4×4 board. Each card has one of eight colored
symbols, chosen independently for each position at the start of an episode.
Their identities stay fixed for that episode. An action reveals one card and
turns the previously visible card face down; another action hides them all.

Suppose you see a red circle in the top-left corner, then reveal several other
cards. The top-left card now shows only its back, but you still know what is
underneath. When it is revealed again, a world model should predict the same red
circle. The current image alone cannot tell it that identity. This gives us a
concrete way to check whether information survives while it is hidden, and later
to test how long the model can retain it.

First, generate some trajectories and watch one. Each episode starts with a new
board and records randomly chosen reveal/hide actions. This command creates 900
training episodes and 100 held-out validation episodes, each 256 frames long:

```sh
python -m data_generation.generate_episodes \
  --task cards --episodes 1000 --length 256 \
  --validation-episodes 100 --seed 42 --output data/cards.h5
```

Play back the first 32 saved frames through the
[pixel loader](data_generation/episode_dataset.py) (a matplotlib window will open on desktop):

```sh
python -m data_generation.episode_dataset data/cards.h5 --clip-size 32
```

As you watch, remember the symbol at one position. If that position is revealed
again, you should see the same symbol. This is the consistency we will eventually
ask the world model to reproduce. Increase `--clip-size` to watch more of the
episode, up to all 256 frames.

The action shown above each image explains how that image was reached:

- **0–15:** reveal a position, numbered row by row from the top left.
- **16:** hide all cards.

In the saved data, **`action[t]` produced `frame[t]`**. The loader prints pixel
shape `(32, 128, 128, 3)` and action shape `(32,)`: 32 RGB board images and their
32 actions. Training and validation contain separate whole episodes, so we can
inspect reconstruction on boards held out from training.

Keep `data/cards.h5` for the following steps. The next stages use these same
images, first to learn a compact representation and then to store it for dynamics
training.

## 3. Prepare a clear way to evaluate predictions

Later, we want to measure how often a model predicts the correct card, and how
that accuracy changes the longer the card stays hidden. To do this consistently
across many generated images, we need a simple, inspectable way to turn each
predicted card into a label we can compare with the true identity.

The card checker assigns each tile one of the eight identities, the card back,
or **unrecognisable**. A recognisable card can still be the wrong prediction;
the last category lets us also account for outputs that do not clearly depict
any valid card. Together, these labels give us the basis for the later accuracy
evaluation.

Before using the checker to score model predictions, inspect its decisions on
examples where we know what the labels should be:

```sh
python -m paper_figures.card_checker --output figures/card_checker.png
```

Open `figures/card_checker.png` (or add `--show` on a desktop). The top row shows
the eight valid identities and the back. Other rows show small pixel errors,
mixtures, wrong colors, and invented symbols. Each tile has a predicted label
and its distance to the nearest valid template.

The [checker](evaluation/card_checker.py) measures RGB MSE inside the dotted
crop, excluding the shared border. It accepts the closest template only when
MSE ≤ `0.001`; otherwise the tile is **unrecognisable**. There is no classifier
to train. The example suite should pass; failures are printed and marked red.
These examples establish the intended behavior, not rejection of every possible
invalid image: small enough changes to a valid card can still be accepted.

## 4. Train the tokenizer

Before learning how the board changes over time, we learn a compact representation
of each image. The dynamics model will predict in the space of these compact representation (latent space), which we
can decode back into images to inspect. They must preserve the card identities
for the later memory experiment to be meaningful.

The tokenizer compresses each image into **four tokens of 64 numbers each**,
then learns to reconstruct the image from them. It processes each frame
independently. Batch size means frames per optimizer update. Temporal context
belongs to the later dynamics model; there is no tokenizer context-length option.

```sh
python -m training.train_tokenizer \
  --data data/cards.h5 --output runs/tokenizer_cards \
  --device cuda --batch-size 256 --steps 15000 --workers 0 --seed 42
```

This samples 256 independent frames per update: 3,840,000 frame samples over
15,000 updates. This takes roughly **4 hours on an A100
40 GB** for this setup.
The first run downloads the VGG weights used by LPIPS; later runs reuse the local cache.

See [training details](training/README.md)
for the objective, sampling, and recorded runs.

Training writes:

- `runs/tokenizer_cards/last.pt`: latest model, optimizer, settings, and progress.
- `runs/tokenizer_cards/metrics.json`: raw training and validation measurements.

To continue an interrupted run, repeat the command with `--resume`. `--steps`
is the total desired update count, not extra updates. Keep the dataset and
training settings the same.

## 5. Inspect what the tokenizer learned

```sh
python -m paper_figures.tokenizer_reconstruction \
  --data data/cards.h5 --checkpoint runs/tokenizer_cards/last.pt \
  --split validation --episode 0 --frames 8 --stride 4 \
  --output figures/tokenizer_cards.png
```

Open `figures/tokenizer_cards.png`. Ground truth is above reconstruction, with
aligned frame indices, actions, and MSE. Check revealed positions, colors, and
symbols as well as card backs. Use `--episode 1`, `--episode 2`, etc. to inspect
other validation episodes; indices are relative to the split. These are
reconstructions of observed frames. Future prediction is a later step.

Also run the card checker on actual tokenizer output:

```sh
python -m paper_figures.card_checker \
  --checkpoint runs/tokenizer_cards/last.pt \
  --output figures/card_checker_tokenizer.png
```

Its second row now shows reconstructed cards cropped from full boards: all eight
identities and the back. Each should be recognised as the corresponding reference
in the top row. This row checks the top-left position; episode sheets show
reconstructions at other positions.

In the earlier successful run, reconstructing only card backs produced a raw
training-loss plateau around `0.04–0.05`, with improvement beginning around
step 2000. At step 15,000 its validation MSE was `3.1275e-6` and LPIPS
`1.4908e-4`. That run sampled 32 contiguous eight-frame clips before flattening
them into 256 single-frame model inputs, and took about 2.5 hours. Those metrics
are a reference, not a guaranteed outcome for the current sampling recipe.
Check the final images; robustness across training seeds has not yet been measured.

## 6. Cache every episode's latents

Dynamics training will revisit the same recorded episodes many times. We can
encode their images once and save the resulting representations, called latents,
so each training update can read them directly. Both the baseline and the memory
model will learn from this same latent dataset.

Once the reconstructions look right, encode the saved pixel dataset:

```sh
python -m data_generation.cache_latents \
  --data data/cards.h5 --checkpoint runs/tokenizer_cards/last.pt \
  --output data/cards_latents.h5 --device cuda --batch-size 64
```

Here batch size controls how many frames are encoded at once. It does not change
the saved episode structure. The [generator](data_generation/cache_latents.py)
writes every frame's encoder output in order as uncompressed `float32` values,
and copies actions, seeds, and the split boundary. Repeated observations still
have their own stored latents.

Expect `(1000, 256, 4, 64)` latents: **250 MiB**, plus about 2 MiB of actions
and a little metadata. The cache records its source dataset ID and checkpoint
SHA-256. It is published only after writing completes; existing outputs are
not overwritten. Use a separate cache for each tokenizer checkpoint.

## 7. Inspect the tensors that dynamics training will receive

```sh
python -m data_generation.episode_latent_dataset data/cards_latents.h5 \
  --clip-size 32 --split validation --episode 0 --start 0
```

This prints file size, dimensions, split counts, latent range, and selected
actions. Expect 100 validation episodes, 225 possible starts per 32-frame clip,
latents shaped `(32, 4, 64)`, and actions shaped `(32,)`.

The [latent loader](data_generation/episode_latent_dataset.py) uses the same
episode/start indexing convention as the pixel loader:

```python
from data_generation.episode_latent_dataset import EpisodeLatentClips

dataset = EpisodeLatentClips("data/cards_latents.h5", context_size=32, split="validation")
latents, actions = dataset[0, 0]
```

Here context size is the **length of the retrieved episode clip**. Each latent
still represents a single frame. `actions[t]` produced the image encoded by
`latents[t]`.
Neighbouring clips (e.g., `dataset[0, 0]` and `dataset[0, 1]`) overlap. Use
`dataset[episode, start]` and `dataset[episode, start + clip_size]` for consecutive,
non-overlapping clips within the same episode.

## 8. Decode the saved cache and compare it with the original pixels

```sh
python -m paper_figures.tokenizer_reconstruction \
  --data data/cards.h5 --latents data/cards_latents.h5 \
  --checkpoint runs/tokenizer_cards/last.pt \
  --split validation --episode 0 --frames 8 --stride 4 \
  --output figures/latent_cache.png
```

Open `figures/latent_cache.png`. The bottom row, labelled **Decoded from latent
cache**, comes from latents retrieved through `EpisodeLatentClips` rather than
newly encoded pixels. Compare it with step 5: the same frames should reconstruct
consistently. The script checks the cache's source dataset and checkpoint, and
that actions align.

At this point, you have a visible pixel dataset, a trained tokenizer, and a latent dataset ready for dynamics training.

## 9. Train the vanilla dynamics baseline

The tokenizer learned to represent individual images. The dynamics model now
learns how those representations depend on earlier frames and the actions taken.
This baseline uses a fixed window of recent frames, which will give us a point
of comparison for the later model with memory.

```sh
python -m training.train_dynamics \
  --data data/cards_latents.h5 --output runs/dynamics_cards \
  --num-actions 17 --device cuda --batch-size 32 --context-size 16 \
  --width 256 --depth 8 --steps 20000 --workers 0 --seed 42
```

Run this inside a GPU allocation. Each update uses 32 clips of 16 frames.
The context length includes the current frame, so a prediction can use at most
15 earlier frames. The 17 actions are the 16 reveal positions and hide-all.
The trainer reads the latent dimensions from the cache and keeps actions aligned
with the frames they produced.

We train with ordinary flow matching: corrupt the latents with noise and predict
their clean values, with a different noise level for each frame and causal
attention across time. This is Dreamer 4-style dynamics without shortcut
distillation. The sampling-step count can be chosen later without retraining.

The trainer saves `last.pt` and `metrics.json` in `runs/dynamics_cards`, evaluating
and checkpointing every 500 updates and at the end. Validation reuses fixed clips
and noise. Both logged losses measure latent denoising; they do not yet tell us
whether generated cards retain the correct identities over long rollouts.
Repeat the command with `--resume` to continue; `--steps` is the total update count.

The settings above are an initial run, not yet a measured convergence or runtime
recipe. Rollout inspection and memory evaluation are the next separate steps.

## What comes next

These stages are still under development:

1. Implement and train the version with memory on the same cache as the baseline.
2. Construct scripted evaluation episodes with controlled periods of hidden information.
3. Compare generated rollouts visually and plot card-identity accuracy against rollout length.
4. Add the two-card matching test.

