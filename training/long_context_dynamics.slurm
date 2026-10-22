#!/bin/bash
#SBATCH --job-name=cards-long-state
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=slurm-long-state-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

hostname
git rev-parse HEAD
date -Is
nvidia-smi

srun python -u -m training.train_long_context_dynamics \
  --data data/cards_latents.h5 \
  --output runs/long_context_cards \
  --num-actions 17 \
  --device cuda \
  --batch-size 32 \
  --context-size 256 \
  --width 128 \
  --depth 8 \
  --heads 4 \
  --steps 40000 \
  --workers 0 \
  --eval-every 500 \
  --seed 4 \
  "$@"
