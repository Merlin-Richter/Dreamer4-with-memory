"""Plot saved card-memory measurements without running model inference.

First run python -m evaluation.card_memory. Real reveals establish the identities
to remember. All probes count, including backs and unrecognisable predictions.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def summarize(result):
    """Aggregate six probes within each episode before measuring uncertainty.

    Episodes are independent sampling seeds. Cards within an episode share a
    history, so treating all card probes as independent would understate error.
    """
    if result['protocol'] != 'observed_six_card_branches_v1':
        raise ValueError('Unsupported card-memory evaluation protocol')
    settings = result['settings']
    gaps = settings['gaps']
    seeds = range(settings['start_seed'], settings['start_seed'] + settings['seeds'])
    rows = {(r['seed'], r['gap'], r['position']): r for r in result['records']}
    expected = {(s, g, p) for s in seeds for g in gaps for p in range(1, 7)}
    if set(rows) != expected or len(rows) != len(result['records']):
        raise ValueError('Expected exactly six target probes per episode and gap')
    summary = {}
    for metric in ['correct', 'predicted_valid', 'other_cards_back']:
        per_seed = np.array([[np.mean([rows[s, g, p][metric] for p in range(1, 7)])
                              for g in gaps] for s in seeds])
        error = per_seed.std(axis=0, ddof=1) / np.sqrt(len(seeds)) if len(seeds) > 1 else np.zeros(len(gaps))
        summary[metric] = dict(mean=per_seed.mean(axis=0), standard_error=error)
    return summary


def save_figure(result, output):
    settings = result['settings']
    summary = summarize(result)
    gaps = np.array(settings['gaps'])
    fig, ax = plt.subplots(figsize=(9, 5.7))
    fig.subplots_adjust(left=.095, right=.97, bottom=.27, top=.79)
    for metric, label, color in [('correct', 'Correct identity', '#245CB0'),
                                  ('predicted_valid', 'Recognisable face', '#BF751C')]:
        mean, se = summary[metric]['mean'], summary[metric]['standard_error']
        ax.plot(gaps, mean, 'o-', label=label, color=color, linewidth=2, markersize=4)
        if settings['seeds'] > 1:
            ax.fill_between(gaps, np.maximum(0, mean - se), np.minimum(1, mean + se),
                            color=color, alpha=.15, linewidth=0)
    ax.axhline(1 / 8, color='#888888', linestyle=':', linewidth=1, label='Uniform guess (1/8)')
    # Once even the sixth reveal leaves the finite context, all six are absent.
    outside = max(0, settings['context_size'] - 2)
    if gaps.min() <= outside <= gaps.max():
        ax.axvline(outside, color='#888888', linestyle='--', linewidth=1)
        ax.text(outside + .4, .48, 'All six original reveals\noutside the context',
                fontsize=9, color='#555555', va='center')
    ax.set_ylim(-.025, 1.055)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('Fraction of all target probes')
    ax.set_xlabel('Extra distractor actions after hide')
    ax.set_xticks(gaps)
    ax.grid(axis='y', alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Recall of observed cards across imagined delays', fontsize=15, y=.965)
    fig.text(.5, .875, f"{settings.get('model_name', 'Vanilla dynamics')} · {settings['seeds']} random boards · 6 probes per delay per board\n"
             f"Six real reveals + hide as prefix · {settings['sampling_steps']} denoising steps/frame",
             ha='center', fontsize=10)
    fig.text(.095, .09, 'Six independent next-frame branches share one history; probes never feed into later delays.\n'
             'Each target has been hidden for gap + 1 to gap + 6 steps. Backs and unrecognisable predictions are failures.',
             fontsize=9)
    uncertainty = ('Shading: ±1 standard error across episodes.' if settings['seeds'] > 1
                   else 'One episode; uncertainty not estimated.')
    fig.text(.095, .025, uncertainty + ' Predictions are scored against the real board; first reveals are not scored.', fontsize=9)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True, help='JSON from evaluation.card_memory')
    parser.add_argument('--output', type=Path, required=True, help='PNG figure')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.png':
        parser.error('output must end in .png')
    result = json.loads(args.results.read_text())
    save_figure(result, args.output)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
