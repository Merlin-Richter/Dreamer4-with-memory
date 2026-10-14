"""Show valid cards, small pixel errors, mixtures, wrong colors and invented symbols.

Run: python -m paper_figures.card_checker [--checkpoint runs/.../last.pt] [--show]
With a checkpoint, row two shows actual reconstructions instead of small errors:
each card is cropped from the top-left of a full board reconstructed independently
with one frame of context. The other 15 cards are hidden.
Saves figures/card_checker.png and checks all 129 clean board observations,
all equal mixtures of 2..8 faces, 25/75 mixtures, and all wrong face colors.
The figure shows a selection; the console reports checks over the full suite.
Tests are synthetic examples, not a guarantee for every possible invalid image.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter

from envs.memory_cards_env import IDENTITIES
from evaluation.card_checker import BACK, NAMES, UNRECOGNISABLE, CardChecker, label_name


def recolor(checker, identity, color):
    """Preserve the rendered silhouette and antialiasing; replace only its color."""
    tile = checker.templates[identity].copy()
    b = checker.border
    background = np.array(ImageColor.getrgb('#FAF8F2')) / 255
    old = np.array(ImageColor.getrgb(IDENTITIES[identity][1])) / 255
    new = np.array(ImageColor.getrgb(IDENTITIES[color][1])) / 255
    alpha = ((tile[b:-b, b:-b] - background) * (old - background)).sum(-1)
    alpha = np.clip(alpha / ((old - background) ** 2).sum(), 0, 1)
    tile[b:-b, b:-b] = background + alpha[..., None] * (new - background)
    return tile


def invented_cards(checker):
    """Draw deliberately invalid symbols at 4x resolution, then antialias."""
    size = checker.tile_size
    background = '#FAF8F2'
    blank = np.rint(checker.templates[0] * 255).astype(np.uint8)
    b = checker.border
    blank[b:-b, b:-b] = ImageColor.getrgb(background)
    cards = []
    for name in ('purple oval', 'green hexagon', 'red X', 'cyan crescent',
                 'orange stripes', 'black zigzag', 'two colored dots', 'blue outline', 'empty face'):
        im = Image.fromarray(blank).resize((size*4, size*4), Image.Resampling.NEAREST)
        d = ImageDraw.Draw(im)
        s = size*4
        if name == 'purple oval':
            d.ellipse((.18*s, .32*s, .82*s, .68*s), fill='#9145CC')
        elif name == 'green hexagon':
            d.polygon([(s*x, s*y) for x, y in [(0.3,0.2),(0.7,0.2),(0.85,0.5),
                       (0.7,0.8),(0.3,0.8),(0.15,0.5)]], fill='#15965B')
        elif name == 'red X':
            d.line((.22*s, .22*s, .78*s, .78*s), fill='#E64045', width=round(.12*s))
            d.line((.22*s, .78*s, .78*s, .22*s), fill='#E64045', width=round(.12*s))
        elif name == 'cyan crescent':
            d.ellipse((.2*s, .2*s, .8*s, .8*s), fill='#009BAD')
            d.ellipse((.35*s, .16*s, .85*s, .7*s), fill=background)
        elif name == 'orange stripes':
            for y in (.25, .45, .65):
                d.rectangle((.2*s, y*s, .8*s, (y+.1)*s), fill='#EE7133')
        elif name == 'black zigzag':
            d.line([(s*x,s*y) for x,y in [(0.2,0.25),(0.8,0.25),(0.2,0.75),(0.8,0.75)]],
                   fill='#202020', width=round(.09*s))
        elif name == 'two colored dots':
            d.ellipse((.18*s,.35*s,.48*s,.65*s), fill='#E64045')
            d.ellipse((.52*s,.35*s,.82*s,.65*s), fill='#1877D2')
        elif name == 'blue outline':
            d.rectangle((.2*s,.2*s,.8*s,.8*s), outline='#1877D2', width=round(.08*s))
        cards.append((name, np.asarray(im.resize((size,size), Image.Resampling.LANCZOS)).astype(np.float32)/255,
                      UNRECOGNISABLE))
    return cards


def examples(checker):
    t = checker.templates
    valid = [(name, tile, i) for i, (name, tile) in enumerate(zip(NAMES, t))]
    perturbed = [(f'{name} + 0.02 RGB', np.clip(tile + .02, 0, 1), i)
                 for i, (name, tile) in enumerate(zip(NAMES, t))]
    blurred = [(f'{name}, blur 0.3 px', np.asarray(Image.fromarray(np.rint(tile*255).astype(np.uint8))
                .filter(ImageFilter.GaussianBlur(.3))).astype(np.float32)/255, i)
               for i, (name, tile) in enumerate(zip(NAMES, t))]
    mixtures = []
    for count in range(2, 9):
        for ids in combinations(range(8), count):
            mixtures.append(('+'.join(NAMES[i] for i in ids), t[list(ids)].mean(0), UNRECOGNISABLE))
    unequal = [(f'25% {NAMES[i]} + 75% {NAMES[j]}', .25*t[i] + .75*t[j], UNRECOGNISABLE)
               for i in range(8) for j in range(8) if i != j]
    face_back = [(f'50% {NAMES[i]} + back', .5*t[i] + .5*t[BACK], UNRECOGNISABLE) for i in range(8)]
    wrong_colors = [(f'{NAMES[i]} / {NAMES[j]} color', recolor(checker, i, j), UNRECOGNISABLE)
                    for i in range(8) for j in range(8) if i != j]
    invented = invented_cards(checker)
    displayed_mixtures = [
        ('50% circle + triangle', (t[0]+t[1])/2, UNRECOGNISABLE),
        ('50% circle + diamond', (t[0]+t[5])/2, UNRECOGNISABLE),
        unequal[0],
        ('3 faces: circle, triangle, square', t[:3].mean(0), UNRECOGNISABLE),
        ('3 faces: star, plus, diamond', t[3:6].mean(0), UNRECOGNISABLE),
        ('4 faces: first four', t[:4].mean(0), UNRECOGNISABLE),
        ('5 faces: first five', t[:5].mean(0), UNRECOGNISABLE),
        ('all 8 faces', t[:8].mean(0), UNRECOGNISABLE), face_back[0],
    ]
    groups = [('The eight valid identities, plus the hidden back', valid),
              ('Valid cards with small pixel errors', perturbed),
              ('Mixtures of two or more cards', displayed_mixtures),
              ('Real shapes with the wrong colors', wrong_colors[::7] + [wrong_colors[1]]),
              ('Invented symbols and an empty face', invented)]
    tests = valid + perturbed + blurred + mixtures + unequal + face_back + wrong_colors + invented
    return groups, tests


def check_boards(checker):
    """All-hidden plus each of 8 faces at each of 16 positions: 129 observations."""
    for position, identity in [(-1, BACK)] + list(np.ndindex(16, 8)):
        frame = np.tile(checker.templates[BACK], (4,4,1))
        expected = np.full((4,4), BACK)
        if position >= 0:
            row, col = divmod(position, 4)
            s = checker.tile_size
            frame[row*s:(row+1)*s, col*s:(col+1)*s] = checker.templates[identity]
            expected[row,col] = identity
        labels, _ = checker.classify_boards(np.rint(frame*255).astype(np.uint8))
        np.testing.assert_array_equal(labels, expected)


def tokenizer_examples(checker, checkpoint):
    """Reconstruct nine full boards, then crop their top-left cards for scoring."""
    import torch
    from models.tokenizer import AutoEncoder, AutoEncoderConfig

    torch.set_num_threads(4)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved['config'].setdefault('output_sigmoid', False)
    saved['config'].setdefault('temporal_layout', 'ssst')
    config = AutoEncoderConfig(**saved['config'])
    s = checker.tile_size
    if (config.img_input_H, config.img_input_W) != (4*s, 4*s):
        raise ValueError(f'The card checker needs a {4*s}×{4*s} tokenizer')
    model = AutoEncoder(config).float().eval()
    model.load_state_dict(saved['model'])
    cases = []
    with torch.inference_mode():
        for identity, name in enumerate(NAMES):
            board = np.tile(checker.templates[BACK], (4, 4, 1))
            board[:s, :s] = checker.templates[identity]
            reconstruction = model(torch.from_numpy(board)[None, None])[0, 0].numpy()
            cases.append((name, reconstruction[:s, :s].copy(), identity))
    title = (f'Tokenizer reconstruction · step {saved["step"]} · '
             'top-left card crops from full boards · single-frame context')
    return title, cases


def save_figure(checker, groups, summary, output, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from textwrap import fill

    fig = plt.figure(figsize=(19, 15), layout='constrained')
    fig.suptitle(f'Card checker: nearest-template pixel MSE\n'
                 f'Accept if MSE ≤ {checker.threshold:g} in the dotted crop; otherwise unrecognisable.\n'
                 f'{summary}', fontsize=14)
    for row, (title, cases) in zip(fig.subfigures(len(groups), 1), groups):
        row.suptitle(title, fontsize=13, fontweight='bold')
        for ax, (name, tile, expected) in zip(row.subplots(1, 9), cases):
            label, error = checker.classify_tiles(tile)
            label = int(label)
            ax.imshow(tile, interpolation='nearest')
            ax.set_title(fill(name, 21), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            correct = label == expected
            color = '#147D45' if label != UNRECOGNISABLE else '#975B12'
            if not correct:
                color = '#C62828'
            caption = f'{label_name(label)}\nMSE {float(error):.3g}'
            if not correct:
                caption += f'\nEXPECTED {label_name(expected)}'
            ax.set_xlabel(caption, fontsize=9, color=color)
            b = checker.border
            ax.add_patch(Rectangle((b-.5,b-.5), checker.tile_size-2*b, checker.tile_size-2*b,
                                  fill=False, edgecolor='#777777', linestyle=':', linewidth=.7))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor='white')
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        help='trained tokenizer checkpoint; replace row two with its reconstructions (CPU)')
    parser.add_argument('--threshold', type=float, default=.001, help='maximum accepted RGB MSE (default: 0.001)')
    parser.add_argument('--output', type=Path, default=Path('figures/card_checker.png'))
    parser.add_argument('--show', action='store_true', help='also open an interactive figure window')
    args = parser.parse_args()
    checker = CardChecker(threshold=args.threshold)
    groups, tests = examples(checker)
    if args.checkpoint is not None:
        groups[1] = tokenizer_examples(checker, args.checkpoint)
        tests += groups[1][1]
    labels, errors = checker.classify_tiles(np.stack([tile for _,tile,_ in tests]))
    expected = np.array([label for _,_,label in tests])
    correct = labels == expected
    valid = expected != UNRECOGNISABLE
    summary = (f'Valid tiles: {correct[valid].sum()}/{valid.sum()} · '
               f'Invalid tiles rejected: {correct[~valid].sum()}/{(~valid).sum()} · '
               '129 clean boards checked')
    check_boards(checker)
    print(summary)
    print(f'Largest valid MSE: {errors[valid].max():.6f}; smallest invalid MSE: {errors[~valid].min():.6f}')
    for i in np.flatnonzero(~correct):
        print(f'FAIL {tests[i][0]}: {label_name(labels[i])}, expected {label_name(expected[i])}, MSE={errors[i]:.6f}')
    save_figure(checker, groups, summary, args.output, args.show)
    print(f'Saved {args.output}')
    if not correct.all():
        raise SystemExit('Card checker examples failed; inspect the figure and reported cases.')


if __name__ == '__main__':
    main()
