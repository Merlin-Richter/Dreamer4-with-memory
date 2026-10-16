#!/bin/bash
#SBATCH --job-name=cards-state
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-state-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

hostname
git rev-parse HEAD
date -Is
nvidia-smi

srun python -u -m training.train_state_dynamics \
  --data data/cards_latents.h5 \
  --output runs/state_dynamics_cards \
  --num-actions 17 \
  --device cuda \
  --batch-size 32 \
  --context-size 16 \
  --width 256 \
  --depth 12 \
  --state-tokens 4 \
  --steps 200000 \
  --workers 0 \
  --eval-every 500 \
  --seed 4 \
  "$@"
