#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../src"
ENV="ALE/Pong-v5"
FRAMES=2000000
SEED=0
mkdir -p ../logs
echo "Launching all 4 Pong variants in parallel..."
python train.py --env $ENV --exp-name pong_dqn         --total-frames $FRAMES --seed $SEED                      > ../logs/pong_dqn.log 2>&1 &
python train.py --env $ENV --exp-name pong_double      --total-frames $FRAMES --seed $SEED --double             > ../logs/pong_double.log 2>&1 &
python train.py --env $ENV --exp-name pong_dueling     --total-frames $FRAMES --seed $SEED --dueling            > ../logs/pong_dueling.log 2>&1 &
python train.py --env $ENV --exp-name pong_double_duel --total-frames $FRAMES --seed $SEED --double --dueling   > ../logs/pong_double_duel.log 2>&1 &
echo "PIDs: $(jobs -p)"
wait
echo "All 4 Pong variants done."
