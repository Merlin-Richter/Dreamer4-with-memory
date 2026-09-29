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
