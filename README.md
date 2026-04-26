# Rainbow-Lite: A Small-Scale Reproduction of Two Rainbow Improvements

**Final Project — Reinforcement Learning, Spring 2026**
School of Artificial Intelligence, Xi'an Jiaotong University

This repository contains the code and experimental results for a course final project that reproduces two of the six improvements unified by the Rainbow agent (Hessel et al., AAAI 2018):

- **Double DQN** (van Hasselt et al., AAAI 2016) — decouples action selection from value estimation in the TD target.
- **Dueling Network** (Wang et al., ICML 2016) — factors `Q(s,a)` into a state value `V(s)` and an advantage `A(s,a)`.

The experiment is a 2×2 ablation over `{Double, no-Double} × {Dueling, no-Dueling}`, trained from scratch on **Pong** (2M frames) and **Breakout** (3M frames) on a single RTX 4090 (24GB).

---

## TL;DR — what we found

| Variant | Pong (2M frames) | Breakout (3M frames) |
| --- | --- | --- |
| DQN (baseline) | −15.4 | 13.5 |
| + Double | −17.1 | **19.7** |
| + Dueling | **−5.9** | 7.3 |
| + Double & Dueling | −15.9 | 5.2 |

Three findings worth highlighting:

1. **Game-dependent policy gains.** Dueling helps Pong but hurts Breakout;
   Double DQN is the opposite. Neither is a universal win at this training scale.
2. **Double DQN's overestimation suppression is robust across both games**, even
   on Pong where its policy benefit is invisible. The mean predicted Q-value
   for Double-equipped variants stays consistently below their non-Double
   counterparts throughout training (see `pong_qvalues.png`, `breakout_qvalues.png`).
3. **No additive composition** at this scale. Double + Dueling never beats the
   better single improvement on either game, contrary to the Rainbow paper's
   composability claim.

These results are consistent with the per-game variance reported in the
original Double DQN and Dueling papers. The lack of composition is at
odds with Rainbow, which we attribute primarily to our reduced training
budget (~1% of the paper's 200M frames per game) and single seed.

---

## Repository layout

```
rainbow-lite/
├── src/
│   ├── atari_wrappers.py   # DeepMind-style preprocessing (frame skip, grayscale, 84×84, frame stack)
│   ├── replay_buffer.py    # uniform replay buffer storing uint8 frames individually
│   ├── networks.py         # Nature CNN + standard Q-head + Dueling head
│   ├── agent.py            # DQN agent with --double / --dueling toggles
│   ├── train.py            # main training loop (one variant per process)
│   ├── evaluate.py         # online eval (5 episodes during training)
│   └── reeval.py           # post-training re-evaluation with more episodes
├── analysis/
│   ├── plot_curves.py      # original learning-curve plotter
│   ├── plot_curves_v2.py   # plots from reeval.csv with std bands
│   └── plot_q_values.py    # mean training Q-value comparison from TensorBoard
├── scripts/
│   ├── run_pong.sh         # sequential 4-variant launcher
│   ├── run_pong_parallel.sh
│   ├── run_pong_all4.sh    # all 4 variants in parallel (recommended on 4090)
│   ├── run_breakout.sh
│   └── run_breakout_all4.sh
├── pong_curves.png         # final learning curves, Pong
├── pong_qvalues.png        # mean Q-value comparison, Pong
├── breakout_curves.png     # final learning curves, Breakout
├── breakout_qvalues.png    # mean Q-value comparison, Breakout
├── requirements.txt
└── README.md
```

---

## Setup

Tested on Linux + RTX 4090 (CUDA 12.1) with Python 3.10.

```bash
# 1. Create env
conda create -n rainbow python=3.10 -y
conda activate rainbow

# 2. Install PyTorch (adjust the CUDA tag to your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install everything else
pip install -r requirements.txt

# 4. Download Atari ROMs (one-time)
AutoROM --accept-license

# 5. Sanity check the env pipeline
cd src
python -c "from atari_wrappers import make_atari_env; \
           env = make_atari_env('ALE/Pong-v5'); \
           o, _ = env.reset(); print('obs shape:', o.shape)"
# Expected: obs shape: (4, 84, 84)
```

A 5-minute smoke test before launching real runs:

```bash
cd src
python train.py --env ALE/Pong-v5 --exp-name smoke \
    --total-frames 100000 --learning-starts 5000 \
    --eval-freq 20000 --eval-episodes 2
```

---

## Reproducing the experiments

### 1. Train all four variants on each game

Each variant is a separate process. With 24GB VRAM and enough RAM, all four
fit on a single 4090 simultaneously.

```bash
# All 4 Pong variants in parallel — ~2.5h on RTX 4090
bash scripts/run_pong_all4.sh

# All 4 Breakout variants in parallel — ~5h on RTX 4090
bash scripts/run_breakout_all4.sh
```

Sequential alternatives (`run_pong.sh`, `run_breakout.sh`) are also provided.

### 2. Re-evaluate checkpoints with more episodes

The training-time evaluation uses only 5 episodes per checkpoint, which is
too noisy for clean curves. After training, re-evaluate every checkpoint
with more episodes:

```bash
cd src
# Pong: dense (every 100k frames) × 20 episodes — ~30 min
python reeval.py --runs-dir runs --pattern "pong_*" \
    --env ALE/Pong-v5 --n-episodes 20

# Breakout: sparse (every 250k frames) × 15 episodes — ~2h
python reeval.py --runs-dir runs --pattern "breakout_*" \
    --env ALE/Breakout-v5 --n-episodes 15 \
    --step-stride 250000 --max-steps-per-ep 8000
```

Each variant directory ends up with a `reeval.csv` consumed by the plotters.

### 3. Generate plots

```bash
cd analysis
python plot_curves_v2.py --runs-dir ../src/runs --game pong --out ../pong_curves.png
python plot_curves_v2.py --runs-dir ../src/runs --game breakout --out ../breakout_curves.png
python plot_q_values.py  --runs-dir ../src/runs --game pong     --out ../pong_qvalues.png
python plot_q_values.py  --runs-dir ../src/runs --game breakout --out ../breakout_qvalues.png
```

`plot_curves_v2.py` prefers `reeval.csv` and falls back to the noisier
`metrics.csv` if reeval hasn't been run for a variant. `plot_q_values.py`
reads `train/q_mean` from the TensorBoard event files.

---

## Implementation notes

The two improvements are deliberately implemented as small, surgical changes
to a clean Nature DQN baseline.

### Double DQN — `src/agent.py::compute_target`

A one-line change. The online network selects the next action; the target
network evaluates it.

```python
if self.double:
    next_actions = self.online(next_obs).argmax(dim=1, keepdim=True)
    next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
else:
    next_q = self.target(next_obs).max(dim=1).values
```

### Dueling DQN — `src/networks.py::DuelingDQN`

Architectural change. After the Nature CNN backbone, two 512-unit streams
produce `V(s)` and `A(s, ·)`, aggregated with mean-subtraction:

```python
q = value + advantage - advantage.mean(dim=1, keepdim=True)
```

Mean-subtraction (rather than the alternative `max`-subtraction in the
original Dueling paper) is the identifiability fix that allows `V` and `A`
to be uniquely recoverable.

The Dueling head roughly doubles the FC parameter count (1.69M → 3.29M); a
matched-capacity baseline ablation would be a useful follow-up.

---

## Hyperparameters

Defaults in `src/train.py`. These largely follow Nature DQN, with the replay
buffer reduced from 1M to 300k for RAM-conservative parallel runs.

| Parameter | Value |
| --- | --- |
| Batch size | 32 |
| Replay buffer capacity | 300,000 |
| Learning starts | 20,000 frames |
| Train frequency | every 4 env steps |
| Target update | every 8,000 gradient updates |
| Learning rate | 6.25 × 10⁻⁵ |
| Optimizer | Adam, eps = 1.5 × 10⁻⁴ |
| Discount γ | 0.99 |
| ε schedule | 1.0 → 0.01 linearly over 250k frames |
| Eval (training-time) | every 100k frames, 5 episodes, ε = 0.001 |
| Eval (post-training reeval) | 15–20 episodes per checkpoint, ε = 0.001 |
| Loss | Huber (Smooth L1) |
| Gradient clipping | L2-norm clipped at 10 |
| Total frames | Pong: 2M; Breakout: 3M |
| Seed | 0 (single seed) |

---

## Limitations

- **Single seed.** RL is high-variance; results are illustrative rather than
  statistically established.
- **Two games rather than 57.** No human-normalized median across the
  benchmark suite — only per-game scores.
- **Reduced training horizon (2–3M vs. paper's 200M frames).** Absolute scores
  are far below the paper's, and improvements that mostly help in the
  long-horizon regime would be undetectable here.
- **300k replay buffer.** Recycles experience faster than the paper's 1M and
  may interact with the improvements in ways we cannot disentangle.
- **Capacity mismatch in Dueling.** Dueling vs. baseline differ in parameter
  count; a width-matched baseline would isolate architecture from capacity.

---

## References

- M. Hessel et al., "Rainbow: Combining improvements in deep reinforcement learning," *AAAI* 2018.
- H. van Hasselt, A. Guez, D. Silver, "Deep reinforcement learning with Double Q-learning," *AAAI* 2016.
- Z. Wang et al., "Dueling network architectures for deep reinforcement learning," *ICML* 2016.
- V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature* 518, 2015.

---

## Acknowledgments

This is course work for *Reinforcement Learning*, Spring 2026, AI Honors Program, Xi'an Jiaotong University. Code is written from scratch for this assignment; no external RL framework is used.