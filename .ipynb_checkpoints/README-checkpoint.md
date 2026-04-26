# Rainbow-Lite: Double DQN + Dueling Ablation

A minimal reproduction of two Rainbow improvements for a course assignment.
Baseline: Nature DQN. Ablations: +Double, +Dueling, +Both.

## Setup (on your RTX 4090D server)

```bash
# 1. Create env
conda create -n rainbow python=3.10 -y
conda activate rainbow

# 2. Install PyTorch matching your CUDA (check nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install the rest
pip install -r requirements.txt

# 4. Download Atari ROMs (one-time)
AutoROM --accept-license

# 5. Smoke test
cd src
python -c "from atari_wrappers import make_atari_env; env = make_atari_env('ALE/Pong-v5'); o,_ = env.reset(); print('obs shape:', o.shape)"
# Expected: obs shape: (4, 84, 84)
```

## Quick sanity check (5-min smoke test)

```bash
cd src
python train.py --env ALE/Pong-v5 --exp-name smoke \
    --total-frames 100000 --learning-starts 5000 --eval-freq 20000 --eval-episodes 2
```
If this runs without errors and you see the `step=` log lines, you're good.

## Full run (the actual experiment)

Sequential (simple, 4 × ~2h each ≈ 8h for Pong):
```bash
bash scripts/run_pong.sh
bash scripts/run_breakout.sh
```

Parallel (recommended — 2 at a time, ~4h for all 4 Pong variants):
```bash
bash scripts/run_pong_parallel.sh
```

Monitor with tensorboard:
```bash
tensorboard --logdir runs --port 6006
```

## Plot results

```bash
cd analysis
python plot_curves.py --runs-dir ../src/runs --game pong --out pong.png
python plot_curves.py --runs-dir ../src/runs --game breakout --out breakout.png
```

## Project layout

- `src/atari_wrappers.py` — DeepMind preprocessing
- `src/replay_buffer.py` — uint8 replay (stores single frames)
- `src/networks.py` — Nature CNN + DQN/Dueling heads
- `src/agent.py` — agent with `--double` / `--dueling` toggles
- `src/train.py` — main loop
- `src/evaluate.py` — eval protocol (epsilon=0.001, 5 episodes)

## Key implementation notes for the report

**Double DQN** (one-line change in `agent.py::compute_target`):
- Vanilla: `max_a Q_target(s', a)`
- Double: `Q_target(s', argmax_a Q_online(s', a))`
- Decouples action selection from value estimation → reduces overestimation bias.

**Dueling DQN** (architectural change in `networks.py::DuelingDQN`):
- `Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)`
- Learns state value separately; helps when the choice of action has small effect on return.
- Mean-subtraction (vs. argmax) is the identifiability fix from Wang et al. 2016.

## Hyperparameters (see `train.py` defaults)

| Param | Value | Notes |
|---|---|---|
| Batch size | 32 | Nature DQN default |
| Replay buffer | 300k | Smaller than paper's 1M — RAM-conservative |
| Learning starts | 20k | After this step, start grad updates |
| Train freq | every 4 env steps | 1 grad step per 4 env steps |
| Target update | every 8k grad steps | Hard update |
| Learning rate | 6.25e-5 | Adam, eps=1.5e-4 |
| Gamma | 0.99 | |
| Epsilon | 1.0 → 0.01 over 250k frames | Short decay for short run |
| Eval | every 100k frames, 5 eps, eps=0.001 | |
