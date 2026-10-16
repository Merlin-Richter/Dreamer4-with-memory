"""Scripted card-memory rollouts and quantitative recall evaluation.

Visual sheets imagine everything: python -m paper_figures.card_rollout.
Quantitative evaluation starts with six real reveals and a hide frame, then
imagines the delay and probes. Predictions are scored against the real board.
Run this module directly to save quantitative measurements across seeds/delays;
python -m paper_figures.card_memory_accuracy plots the saved JSON.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from models.dynamics_model import DynamicsModel, DynamicsModelConfig
from models.state_dynamics_model import StateDynamicsModel, StateDynamicsModelConfig
from models.tokenizer import AutoEncoder, AutoEncoderConfig
from evaluation.card_checker import BACK, CardChecker
from envs.memory_cards_env import MemoryCards


def scripted_actions(gap=0):
    """Reveal positions 1..6, hide, distract, then revisit 4,2,6,3,5,1.

    Return zero-based environment actions. Distractors cycle through positions
    7..16, never revealing a target. Every action produces one generated frame.
    ``gap`` counts extra actions after hide, not each target's total hidden time.
    """
    if not isinstance(gap, int) or gap < 0:
        raise ValueError('gap must be a nonnegative integer')
    return np.array([0, 1, 2, 3, 4, 5, 16]
                    + [6 + i % 10 for i in range(gap)]
                    + [3, 1, 5, 2, 4, 0], dtype=np.int64)


def load_models(checkpoint, tokenizer_checkpoint, device='cpu'):
    """Load trusted local checkpoints and require the tokenizer used in training."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    with open(tokenizer_checkpoint, 'rb') as file:
        tokenizer_hash = hashlib.file_digest(file, 'sha256').hexdigest()
    if saved['tokenizer_sha256'] != tokenizer_hash:
        raise ValueError('Tokenizer checkpoint does not match the dynamics training latents')
    if saved.get('objective') != 'flow_matching_x_ramp_v1':
        raise ValueError('Expected an ordinary flow-matching dynamics checkpoint')
    architecture = saved.get('architecture', 'vanilla')
    if architecture == 'state_stream_v1':
        dynamics = StateDynamicsModel(StateDynamicsModelConfig(**saved['config']))
    elif architecture == 'vanilla':
        dynamics = DynamicsModel(DynamicsModelConfig(**saved['config']))
    else:
        raise ValueError(f'Unknown dynamics architecture: {architecture}')
    dynamics = dynamics.float().to(device).eval()
    dynamics.load_state_dict(saved['model'])
    if dynamics.n_actions != 17:
        raise ValueError('Card rollouts require 16 reveal actions and one hide action')
    saved = torch.load(tokenizer_checkpoint, map_location='cpu', weights_only=False)
    saved['config'].setdefault('output_sigmoid', False)
    saved['config'].setdefault('temporal_layout', 'ssst')
    config = AutoEncoderConfig(**saved['config'])
    if config.max_temporal_length != 1:
        raise ValueError('Expected the frame-independent tokenizer used for latent caching')
    if (config.n_latents, config.bottleneck_dim) != (dynamics.n_latents, dynamics.bottleneck_dim):
        raise ValueError('Tokenizer and dynamics latent shapes differ')
    if config.img_input_H != config.img_input_W or config.img_input_H % 4:
        raise ValueError('Expected square images containing a 4x4 card board')
    tokenizer = AutoEncoder(config).float().to(device).eval()
    tokenizer.load_state_dict(saved['model'])
    return dynamics, tokenizer


@torch.inference_mode()
def generate_rollout(dynamics, tokenizer, actions, seed=0, sampling_steps=16):
    """Return decoded RGB frames (T,H,W,3); no ground-truth prefix or feedback.

    Sampling is seeded after loading models. The dynamics model manages its own
    sliding context; decoded pixels never feed back into generation.
    """
    device = next(dynamics.parameters()).device
    torch.manual_seed(seed)
    action_idx = torch.as_tensor(actions, dtype=torch.long, device=device)[None]
    empty = torch.empty((1, 0, dynamics.n_latents, dynamics.bottleneck_dim), device=device)
    latents = dynamics.generate(empty, len(actions), K=sampling_steps, action_idx=action_idx)
    return torch.stack([tokenizer.decoder(z[None, None])[0, 0].cpu()
                        for z in latents[0]]).numpy()


@torch.inference_mode()
def evaluate_seed(dynamics, tokenizer, gaps, seed=0, sampling_steps=16):
    """Encode real reveals, imagine the delay, then probe all six targets per gap.

    Each probe is a separate next-frame prediction with the same partly imagined
    history. Probes never become history for another probe or delay. Six branches
    form a batch, not six simultaneous actions in one environment. Generate the
    entire history first so adding probe points cannot change it.

    Board identities are uniformly sampled by the environment. Only the seven
    prefix images enter the model; privileged board metadata is used for scoring.
    Every probe counts: backs and unrecognisable predictions are failures.
    """
    if not gaps or any(not isinstance(g, int) or g < 0 for g in gaps):
        raise ValueError('Need nonnegative integer gaps')
    device = next(dynamics.parameters()).device
    action_ids = scripted_actions(max(gaps))[:-6]
    actions = torch.as_tensor(action_ids, device=device)[None]
    resolution = tokenizer.decoder.h_patches * tokenizer.decoder.patch_size
    env = MemoryCards(resolution=resolution, seed=seed)
    targets = env.evaluation_state()['board'][:6]
    frames = np.stack([env.step(int(a)) for a in action_ids[:7]])
    pixels = torch.as_tensor(frames, device=device, dtype=torch.float32) / 255
    # Seven independent images in a batch, reassembled as one temporal prefix.
    history = tokenizer.encoder(pixels[:, None])[:, 0][None]
    torch.manual_seed(seed)
    if max(gaps):
        delay = dynamics.generate(history, max(gaps), K=sampling_steps, action_idx=actions)
        history = torch.cat((history, delay), dim=1)
    checker = CardChecker(tile_size=resolution // 4)
    records = []
    for gap in gaps:
        # t is the time of the branched next-frame predictions, starting at zero.
        t = 7 + gap
        start = max(0, t - dynamics.config.max_temporal_length + 1)
        context = history[:, start:t].expand(6, -1, -1, -1)
        probe_actions = torch.cat((actions[:, start:t].expand(6, -1),
                                   torch.arange(6, device=device)[:, None]), dim=1)
        # Independent, reproducible probe noise; changing the gap list/order does
        # not change the noise for an existing gap (for a fixed history).
        torch.manual_seed(seed + 1_000_003 * (gap + 1))
        latents = dynamics.generate(context, 1, K=sampling_steps, action_idx=probe_actions)
        frames = tokenizer.decoder(latents)[:, 0].cpu().numpy()
        labels, errors = checker.classify_boards(frames)
        labels, errors = labels.reshape(6, 16), errors.reshape(6, 16)
        for target in range(6):
            expected, predicted = int(targets[target]), int(labels[target, target])
            records.append(dict(seed=seed, gap=gap, position=target + 1,
                                first_time=target, probe_time=t, hidden_steps=t - target - 1,
                                expected_label=expected, predicted_label=predicted,
                                predicted_mse=float(errors[target, target]),
                                predicted_valid=0 <= predicted < BACK,
                                correct=expected == predicted,
                                other_cards_back=bool(np.all(np.delete(labels[target], target) == BACK))))
    return records


def main():
    parser = argparse.ArgumentParser(description='Measure recall of real cards after imagined delays.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='JSON containing every probe and run settings')
    parser.add_argument('--seeds', type=int, default=32, help='number of independently sampled real boards (default: 32)')
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--gaps', type=int, nargs='+', default=[0, 4, 8, 10, 12, 14, 16, 24, 32])
    parser.add_argument('--sampling-steps', type=int, default=16)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    args = parser.parse_args()
    if args.seeds < 1 or args.start_seed < 0 or min(args.gaps) < 0 or args.output.suffix.lower() != '.json':
        parser.error('Need positive seeds, nonnegative start-seed/gaps, and a .json output')
    torch.set_num_threads(4)
    dynamics, tokenizer = load_models(args.checkpoint, args.tokenizer, args.device)
    gaps = sorted(set(args.gaps))
    settings = dict(checkpoint=str(args.checkpoint), tokenizer=str(args.tokenizer),
                    model_name='State stream' if isinstance(dynamics, StateDynamicsModel) else 'Vanilla dynamics',
                    seeds=args.seeds, start_seed=args.start_seed, gaps=gaps,
                    sampling_steps=args.sampling_steps, device=args.device,
                    context_size=dynamics.config.max_temporal_length, checker_threshold=.001,
                    torch_version=torch.__version__)
    for name, path in [('checkpoint_sha256', args.checkpoint), ('tokenizer_sha256', args.tokenizer)]:
        with open(path, 'rb') as file:
            settings[name] = hashlib.file_digest(file, 'sha256').hexdigest()
    records = []
    start = time.perf_counter()
    for i, seed in enumerate(range(args.start_seed, args.start_seed + args.seeds)):
        records.extend(evaluate_seed(dynamics, tokenizer, gaps, seed, args.sampling_steps))
        elapsed = time.perf_counter() - start
        remaining = elapsed / (i + 1) * (args.seeds - i - 1)
        print(f'{i + 1}/{args.seeds} episodes · {elapsed:.1f}s elapsed · ~{remaining:.0f}s remaining', flush=True)
    result = dict(protocol='observed_six_card_branches_v1', settings=settings,
                  elapsed_seconds=time.perf_counter() - start, records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved {len(records)} probes to {args.output}', flush=True)


if __name__ == '__main__':
    main()
