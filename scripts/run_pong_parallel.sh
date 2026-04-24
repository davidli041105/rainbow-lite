#!/usr/bin/env bash
# Run 2 variants simultaneously on a single GPU (each uses ~2-3GB VRAM).
# Use tmux or nohup for long runs.
set -e
cd "$(dirname "$0")/../src"

ENV="ALE/Pong-v5"
FRAMES=2000000
SEED=0

mkdir -p ../logs
python train.py --env $ENV --exp-name pong_dqn         --total-frames $FRAMES --seed $SEED                      > ../logs/pong_dqn.log 2>&1 &
PID1=$!
python train.py --env $ENV --exp-name pong_double      --total-frames $FRAMES --seed $SEED --double             > ../logs/pong_double.log 2>&1 &
PID2=$!
wait $PID1 $PID2

python train.py --env $ENV --exp-name pong_dueling     --total-frames $FRAMES --seed $SEED --dueling            > ../logs/pong_dueling.log 2>&1 &
PID3=$!
python train.py --env $ENV --exp-name pong_double_duel --total-frames $FRAMES --seed $SEED --double --dueling   > ../logs/pong_double_duel.log 2>&1 &
PID4=$!
wait $PID3 $PID4

echo "All Pong variants done."
