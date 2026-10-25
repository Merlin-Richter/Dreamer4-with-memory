#!/bin/bash
#SBATCH --job-name=cards-recurrent
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=slurm-recurrent-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

hostname
git rev-parse HEAD
date -Is
nvidia-smi

srun python -u -m training.train_recurrent_dynamics \
  --parent runs/long_context_cards/last.pt \
  --data data/cards_latents.h5 \
  --output runs/recurrent_cards \
  --device cuda \
  --batch-size 32 \
  --context-size 256 \
  --block-size 16 \
  --memory-slots 16 \
  --teacher-weight 0.1 \
  --steps 10000 \
  --workers 0 \
  --eval-every 250 \
  --seed 4 \
  "$@"
