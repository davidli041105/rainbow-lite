#!/usr/bin/env bash
# Run all 4 Pong variants. Adjust GPU index if needed.
set -e
cd "$(dirname "$0")/../src"

ENV="ALE/Pong-v5"
FRAMES=2000000     # Pong solves quickly — 2M is plenty
SEED=0

python train.py --env $ENV --exp-name pong_dqn           --total-frames $FRAMES --seed $SEED
python train.py --env $ENV --exp-name pong_double        --total-frames $FRAMES --seed $SEED --double
python train.py --env $ENV --exp-name pong_dueling       --total-frames $FRAMES --seed $SEED           --dueling
python train.py --env $ENV --exp-name pong_double_duel   --total-frames $FRAMES --seed $SEED --double --dueling
