"""
Re-evaluate trained checkpoints with more episodes for lower-variance curves.

For each variant directory, loads every ckpt_*.pt and runs N evaluation episodes,
writing results to reeval.csv inside the variant directory.

Usage:
    python reeval.py --runs-dir runs --pattern "pong_*" --env ALE/Pong-v5 --n-episodes 20
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from atari_wrappers import make_eval_env
from agent import DQNAgent
from evaluate import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--pattern", required=True, help="e.g. pong_* or breakout_*")
    p.add_argument("--env", required=True, help="e.g. ALE/Pong-v5")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--epsilon", type=float, default=0.001)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def get_step_from_ckpt_name(name: str):
    """ckpt_100000.pt -> 100000;  ckpt_final.pt -> None (skip)."""
    m = re.match(r"ckpt_(\d+)\.pt$", name)
    return int(m.group(1)) if m else None


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    variant_dirs = sorted([d for d in runs_dir.glob(args.pattern) if d.is_dir()])
    if not variant_dirs:
        print(f"No directories matched {args.pattern} under {runs_dir}")
        sys.exit(1)

    print(f"Found {len(variant_dirs)} variants:")
    for d in variant_dirs:
        print(f"  {d.name}")

    # We can reuse one eval env across all checkpoints/variants
    eval_env = make_eval_env(args.env, seed=12345)
    n_actions = eval_env.action_space.n

    for vdir in variant_dirs:
        # Read config to know whether double/dueling were used
        with open(vdir / "config.json") as f:
            cfg = json.load(f)
        print(f"\n=== {vdir.name}  (double={cfg['double']}, dueling={cfg['dueling']}) ===")

        ckpts = sorted(
            [(get_step_from_ckpt_name(p.name), p) for p in vdir.glob("ckpt_*.pt")
             if get_step_from_ckpt_name(p.name) is not None],
            key=lambda x: x[0],
        )
        if not ckpts:
            print(f"  No numbered checkpoints. Skipping.")
            continue

        agent = DQNAgent(
            n_actions=n_actions, device=args.device, lr=1e-4, gamma=0.99,
            double=cfg["double"], dueling=cfg["dueling"],
        )

        out_csv = vdir / "reeval.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "n_episodes", "return_mean", "return_std",
                        "return_min", "return_max"])
            for step, ckpt_path in ckpts:
                ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
                agent.online.load_state_dict(ckpt["online"])
                agent.online.eval()

                returns = []
                t0 = time.time()
                for _ in range(args.n_episodes):
                    obs, _ = eval_env.reset()
                    ep_ret = 0.0
                    for _ in range(27000):  # max episode steps
                        a = agent.act(np.asarray(obs, dtype=np.uint8), args.epsilon)
                        obs, r, term, trunc, _ = eval_env.step(a)
                        ep_ret += r
                        if term or trunc:
                            break
                    returns.append(ep_ret)
                returns = np.array(returns)
                w.writerow([step, args.n_episodes, returns.mean(), returns.std(),
                            returns.min(), returns.max()])
                f.flush()
                print(f"  step={step:>8d}  mean={returns.mean():>7.2f}  "
                      f"std={returns.std():>5.2f}  "
                      f"min={returns.min():>4.0f}  max={returns.max():>4.0f}  "
                      f"({time.time()-t0:.1f}s)")
        print(f"  -> {out_csv}")

    eval_env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()