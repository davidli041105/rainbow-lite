#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../src"

ENV="ALE/Breakout-v5"
FRAMES=3000000
SEED=0

python train.py --env $ENV --exp-name breakout_dqn           --total-frames $FRAMES --seed $SEED
python train.py --env $ENV --exp-name breakout_double        --total-frames $FRAMES --seed $SEED --double
python train.py --env $ENV --exp-name breakout_dueling       --total-frames $FRAMES --seed $SEED           --dueling
python train.py --env $ENV --exp-name breakout_double_duel   --total-frames $FRAMES --seed $SEED --double --dueling
