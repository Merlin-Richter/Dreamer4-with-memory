"""Write N fixed-length episodes to a losslessly compressed HDF5 file.

Run from the repository root:
python -m data_generation.generate_episodes --task cards --episodes 1000 --length 256 --output data/cards.h5
Pixels remain RGB uint8 (0..255); normalization belongs in the trainer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import uuid

import h5py
import numpy as np

from envs.bouncing_square_env import GridWorldEnv
from envs.memory_cards_env import MemoryCards


def generate_episode(task, seed, length):
    """Return RGB frames and aligned actions: action[t] produces frame[t].

    Cards: independent identities; uniform actions over 16 reveals and hide, no shifts/waits.
    Ball: initially revealed; curtain toggles with a per-episode probability drawn
    log-uniformly from [1/64, 1/4]. Physics advances on every step, hidden or not.
    Reset observations and privileged simulator states are not included.
    """
    # Separate streams prevent actions from accidentally encoding the board/state.
    env_stream, action_stream = np.random.SeedSequence(seed).spawn(2)
    env_seed = int(env_stream.generate_state(1, dtype=np.uint64)[0])
    rng = np.random.default_rng(action_stream)
    if task == 'cards':
        env = MemoryCards(seed=env_seed, shifts=False)
        actions = rng.integers(0, 17, size=length, dtype=np.int64)
        size = 128
    else:
        env = GridWorldEnv().reset(seed=env_seed)
        probability = np.exp(rng.uniform(np.log(1/64), np.log(1/4)))
        actions = np.zeros(length, dtype=np.int64)
        for t in range(1, length):
            actions[t] = actions[t-1] ^ int(rng.random() < probability)
        size = 64
    frames = np.empty((length, size, size, 3), dtype=np.uint8)
    for t, action in enumerate(actions):
        if task == 'cards':
            frames[t] = env.step(int(action))
        else:
            frames[t], _ = env.step(int(action))
    return frames, actions


def main():
    """Write one episode at a time, then publish the completed file."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['cards', 'ball'], required=True)
    parser.add_argument('--episodes', type=int, required=True)
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--validation-episodes', type=int, help='default: 10%% of episodes, at least one')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    n_val = max(1, args.episodes // 10) if args.validation_episodes is None else args.validation_episodes

    if args.episodes < 2 or not 0 < n_val < args.episodes or args.length < 1 or args.seed < 0:
        parser.error('need positive length, nonnegative seed, and at least one episode per split')
    if args.task == 'ball' and args.length < 2:
        parser.error('ball episodes need at least two frames to allow occlusion')
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    if args.output.exists() or temporary.exists():
        parser.error('output or temporary file already exists; choose a new path')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    size = 128 if args.task == 'cards' else 64
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), root/'envs/memory_cards_env.py', root/'envs/bouncing_square_env.py']
    
    with h5py.File(temporary, 'x') as file:
        file.attrs.update(format_version=1, dataset_id=str(uuid.uuid4()), complete=False,
            task=args.task, color_order='RGB', train_episodes=args.episodes-n_val,
            settings=json.dumps({**vars(args), 'output': str(args.output), 'validation_episodes': n_val}),
            source_sha256=json.dumps({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
        frames = file.create_dataset('frames', (args.episodes, args.length, size, size, 3),
            dtype='uint8', chunks=(1, min(16, args.length), size, size, 3), compression='gzip', compression_opts=1)
        actions = file.create_dataset('actions', (args.episodes, args.length), dtype='int64')
        seeds = file.create_dataset('episode_seeds', (args.episodes,), dtype='uint64')
        for episode in range(args.episodes):
            seed = int(rng.integers(0, 2**63))
            frames[episode], actions[episode] = generate_episode(args.task, seed, args.length)
            seeds[episode] = seed
            if (episode+1) % 100 == 0 or episode+1 == args.episodes:
                print(f'{episode+1}/{args.episodes} episodes written', flush=True)
        file.attrs['complete'] = True
    temporary.replace(args.output)
    print(f'Saved {args.output}: {args.episodes-n_val} train / {n_val} validation episodes')


if __name__ == '__main__':
    main()
