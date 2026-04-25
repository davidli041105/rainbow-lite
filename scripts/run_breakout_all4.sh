#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../src"
ENV="ALE/Breakout-v5"
FRAMES=3000000
SEED=0
mkdir -p ../logs
echo "Launching all 4 Breakout variants in parallel..."
python train.py --env $ENV --exp-name breakout_dqn         --total-frames $FRAMES --seed $SEED                      > ../logs/breakout_dqn.log 2>&1 &
python train.py --env $ENV --exp-name breakout_double      --total-frames $FRAMES --seed $SEED --double             > ../logs/breakout_double.log 2>&1 &
python train.py --env $ENV --exp-name breakout_dueling     --total-frames $FRAMES --seed $SEED --dueling            > ../logs/breakout_dueling.log 2>&1 &
python train.py --env $ENV --exp-name breakout_double_duel --total-frames $FRAMES --seed $SEED --double --dueling   > ../logs/breakout_double_duel.log 2>&1 &
echo "PIDs: $(jobs -p)"
wait
echo "All 4 Breakout variants done."
