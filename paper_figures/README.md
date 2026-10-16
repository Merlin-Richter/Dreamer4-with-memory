# Paper figures

One script per figure idea. Run modules from the repository root.

## Tokenizer reconstruction

```sh
python -m paper_figures.tokenizer_reconstruction \
  --checkpoint runs/tokenizer_cards/last.pt --data data/cards.h5 \
  --episode 0 --start 0 --frames 8 --stride 4 \
  --output figures/tokenizer_cards.png
```

Exports high-resolution PNG images at 300 DPI. Works with both environments.
The default selects validation episode 0 (indices are relative to the split).
GT is above reconstruction, with original zero-based frame indices. `--stride`
selects columns only: temporal context always includes the intervening frames.

The checkpoint determines whether attention is temporal. Each displayed frame
uses up to the trained context length of preceding GT frames, including itself;
near the episode start there is less history. Frame-independent checkpoints use
one frame at a time. These are reconstructions, not future predictions. Outputs
are clamped to [0, 1] only for display. Use trusted local training checkpoints.

## Imagined card rollouts

Reveal six positions, hide the cards, then revisit those positions in a different
order. The model generates every frame from noise and actions, without an observed
prefix. Its first reveals establish the identities to remember; the captions
compare each revisit with that position's first generated identity.

```sh
python -m paper_figures.card_rollout \
  --checkpoint runs/last.pt --tokenizer runs/cards_cluster_1000/last.pt \
  --output figures/card_rollout_gap0.png

python -m paper_figures.card_rollout \
  --checkpoint runs/last.pt --tokenizer runs/cards_cluster_1000/last.pt \
  --gap 16 --output figures/card_rollout_gap16.png
```

Use your dynamics checkpoint and the exact tokenizer used to create its training
cache. The script checks that they match. Both examples use sampling seed 0 and
16 denoising steps per frame, including when the checkpoint stores a different
inference budget. The same seed
and device give the same initial reveals in both examples. Use `--seed` to inspect
another sample, `--device cuda` for GPU inference, or `--sampling-steps` to change
the denoising budget.

Both rollout and quantitative scripts also accept state-stream checkpoints from
`training.train_state_dynamics`; the checkpoint identifies the architecture.
Use separate output filenames for the two models.

`--gap 16` adds 16 actions at other positions between hide and the first revisit.
These frames are generated and used as context, but omitted from the sheet with
a labelled gap. This number is the added delay, not each card's total hidden time.
Positions are numbered 1–16 in reading order; frame indices start at zero.

The outline marks the requested card. Labels use the existing card checker's
threshold on unmodified decoded pixels. A back or unrecognisable card never counts
as a matching identity. Other positions should remain face down: inspect the full
boards as well as the target captions. These sheets show individual imagined
episodes and self-consistency, not aggregate accuracy against a real board.

## Card memory across delays

Sample real boards and measure recall of their actual card identities. For each
board, encode six real reveals at positions 1–6 and a real hide frame as the
initial history. Then let the model imagine actions at other positions throughout
the delay, without supplying further real frames. At each measured delay, six
separate next-frame predictions branch from the same history, one for each target. They run as a batch;
no probe is allowed to influence another probe or a later delay.

```sh
python -m evaluation.card_memory \
  --checkpoint runs/last.pt --tokenizer runs/cards_cluster_1000/last.pt \
  --seeds 32 --device cuda --output figures/card_memory_accuracy.json

python -m paper_figures.card_memory_accuracy \
  --results figures/card_memory_accuracy.json \
  --output figures/card_memory_accuracy.png
```

Use `--device mps` for Apple GPUs, or omit the device flag to run on CPU.
Sampling defaults to 16 denoising steps.
`--seeds` counts independently sampled real boards, starting at `--start-seed`
(default 0). `--gaps 0 4 8 10 12 14 16 24 32` chooses the extra distractor actions
after hide. Each seven-frame real prefix is encoded once; the imagined history
through the longest delay is generated once and reused. Only probes need decoding.
The evaluator prints elapsed time and an estimate of the remaining runtime.

The JSON stores every probe, its seed, target position, actual and predicted
identities, prediction template MSE, exact hidden duration, and whether all other
cards are backs. It also records checkpoint hashes and sampling settings. The plotting
command reads this file without loading models or rerunning inference.

The blue curve counts correct identities among **all** target probes, using the
environment's board metadata as ground truth. That metadata never enters the
model. First reveals are inputs and are not scored; backs and unrecognisable
predictions count as failures. The orange curve measures recognisable faces on
revisits, separating memory failures from visual failures.
Shading is one standard error across episodes, computed after averaging the six
correlated probes within each episode. The 1/8 line is the chance reference for
the environment's uniformly sampled identities.

The x-axis counts added distractor actions. At gap `g`, the targets have actually
been hidden for `g+1` through `g+6` intervening frames because they were revealed
in sequence. With a 16-frame context (15 past frames), all six original reveals
have left the context at gap 14. This does not force forgetting: generated latents
could carry information forward. The qualitative rollout sheets above remain
fully imagined; this quantitative test measures recall of externally supplied
card identities.
