"""Show a scripted, fully imagined card episode and its revisits.

Reveal six cards, hide them, optionally insert distractor actions, then revisit
the same six positions. All frames are generated; only distractors are omitted
from the sheet. Positions are numbered 1..16 across rows, starting at top left.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from evaluation.card_checker import BACK, CardChecker, label_name
from evaluation.card_memory import generate_rollout, load_models, scripted_actions


def save_sheet(frames, gap, output, *, seed=0, sampling_steps=16, context_size=16):
    """Plot first reveals and revisits in order; classify unclipped decoder pixels."""
    actions = scripted_actions(gap)
    if len(frames) != len(actions):
        raise ValueError('Need one frame per scripted action, including distractors')
    checker = CardChecker(tile_size=frames.shape[1] // 4)
    labels, errors = checker.classify_boards(frames)
    labels, errors = labels.reshape(-1, 16), errors.reshape(-1, 16)
    # None reserves an explicit gap in the figure, without skipping generation.
    times = list(range(7)) + ([None] if gap else []) + list(range(7 + gap, 13 + gap))
    width = 1.55 * len(times)
    fig, axes = plt.subplots(1, len(times), figsize=(width, 3.65),
                             gridspec_kw=dict(left=.015, right=.985, bottom=.33,
                                              top=.76, wspace=.12))
    for ax, time in zip(axes, times):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if time is None:
            ax.text(.5, .56, '···', ha='center', fontsize=24, transform=ax.transAxes)
            ax.text(.5, .22, f'{gap} intervening\nactions omitted\n(positions 7–16)',
                    ha='center', va='center', fontsize=9, color='#555555', transform=ax.transAxes)
            continue
        action = actions[time]
        ax.imshow(frames[time], interpolation='nearest')
        ax.set_title(f't = {time}\n' + ('Hide all' if action == 16 else f'Reveal {action + 1}'),
                     fontsize=10)
        if action == 16:
            backs = int((labels[time] == BACK).sum())
            caption, color = f'{backs}/16 backs', '#555555'
        else:
            label = int(labels[time, action])
            error = errors[time, action]
            caption = f'{label_name(label)}\nMSE {error:.2g}'
            color = '#555555'
            if time < 6:
                caption += '\nfirst reveal'
            else:
                first = int(labels[action, action])
                if not 0 <= first < BACK:
                    status, color = 'no valid first reveal', '#9A6500'
                elif not 0 <= label < BACK:
                    status, color = 'no valid face', '#9A6500'
                elif label == first:
                    status, color = 'same identity', '#207345'
                else:
                    status, color = 'different identity', '#BD3737'
                caption += '\n' + status
            s = checker.tile_size
            ax.add_patch(Rectangle((action % 4 * s - .5, action // 4 * s - .5),
                                   s, s, fill=False, edgecolor='#555555', linewidth=.8))
        ax.set_xlabel(caption, fontsize=9, color=color, labelpad=8)
    fig.suptitle('First reveals → hide → ' + ('distractors → ' if gap else '') + 'revisits',
                 fontsize=14, y=.97)
    fig.text(.5, .855,
             f'Fully imagined · no observed prefix · seed {seed} · {sampling_steps} denoising steps/frame'
             f' · context: at most {context_size - 1} past frames', ha='center', fontsize=10)
    fig.text(.015, .12, 'Positions 1–16 run left to right, top to bottom. The outline marks the requested card;'
             ' other cards should be face down.', fontsize=10)
    fig.text(.015, .045, f'Labels use card-interior template MSE ≤ {checker.threshold:g}.'
             ' Revisits compare with the first generated reveal at that position; backs and unrecognisable cards never count as matches.',
             fontsize=10)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='trusted dynamics checkpoint')
    parser.add_argument('--tokenizer', type=Path, required=True, help='tokenizer used to build the training cache')
    parser.add_argument('--output', type=Path, required=True, help='PNG file')
    parser.add_argument('--gap', type=int, default=0, help='extra distractor actions after hide (try 16)')
    parser.add_argument('--seed', type=int, default=0, help='sampling seed; same seed gives the same initial reveals')
    parser.add_argument('--sampling-steps', type=int, default=16, help='default: 16; must divide the signal grid')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    args = parser.parse_args()
    if args.gap < 0 or args.output.suffix.lower() != '.png':
        parser.error('gap must be nonnegative and output must end in .png')
    torch.set_num_threads(4)
    actions = scripted_actions(args.gap)
    dynamics, tokenizer = load_models(args.checkpoint, args.tokenizer, args.device)
    steps = args.sampling_steps
    print(f'Generating all {len(actions)} frames from noise, {steps} denoising steps/frame, seed {args.seed}…', flush=True)
    frames = generate_rollout(dynamics, tokenizer, actions, args.seed, steps)
    save_sheet(frames, args.gap, args.output, seed=args.seed, sampling_steps=steps,
               context_size=dynamics.config.max_temporal_length)
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
