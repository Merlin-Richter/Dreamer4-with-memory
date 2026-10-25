#!/bin/bash
#SBATCH --job-name=ball-tokenizer
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-ball-tokenizer-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

hostname
git rev-parse HEAD
date -Is
nvidia-smi

# Fresh training with normalized MSE, worst-patch MSE and VGG LPIPS.
# Batch size counts independent frames; temporal length is always one.
# The tokenizer trainer saves periodically; it does not handle Slurm USR1.
srun python -u -m training.train_tokenizer \
  --data data/ball.h5 \
  --output runs/tokenizer_ball_lpips \
  --device cuda \
  --batch-size 256 \
  --steps 30000 \
  --width 128 \
  --depth 3 \
  --lr 0.0003 \
  --lpips-weight 0.2 \
  --workers 0 \
  --validation-batches 4 \
  --eval-every 500 \
  --seed 42 \
  "$@"
