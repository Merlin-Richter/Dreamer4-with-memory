# Dreamer4 with Memory

Code for memory world-model experiments with memory cards and a bouncing square behind a curtain. Currently supports saved episode generation, tokenizer training, and reconstruction sheets.

## Setup

Run commands from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Generate data

```sh
python -m data_generation.generate_episodes --task cards --episodes 1000 --length 128 --output data/cards.h5
```

Use `--task ball` for the bouncing square. Episodes are saved as RGB bytes in HDF5, with aligned actions and separate training/validation episodes. Existing datasets are not overwritten.

## Train a tokenizer

```sh
python -m training.train_tokenizer --data data/cards.h5 --output runs/tokenizer_cards --device cpu --batch-size 1 --context-size 8
```

Use `--device mps` on Apple Silicon or `--device cuda` on the cluster. Each update processes `batch-size × context-size` frames; increase sizes cautiously. Temporal attention is off by default; enable it with `--temporal-attention`.

The decoder uses sigmoid output. Training minimizes pixel MSE plus 0.1 times the worst patch MSE per frame (using the tokenizer’s 8×8 patch size for both environments).

Training records the combined training loss and training/validation MSE and retains `last.pt` in the output directory. Add `--resume` to continue the same run; `--steps` specifies the total update count.

## Make a reconstruction sheet

```sh
python -m paper_figures.tokenizer_reconstruction \
  --checkpoint runs/tokenizer_cards/last.pt --data data/cards.h5 \
  --episode 0 --frames 8 --stride 4 --output figures/tokenizer_cards.png
```

Shows validation GT above tokenizer reconstruction. Exports high-resolution PNG images. Each figure type has its own script in `paper_figures/`.

