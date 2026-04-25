"""Re-evaluate trained checkpoints with N episodes for low-variance curves."""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from atari_wrappers import make_eval_env
from agent import DQNAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--pattern", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--epsilon", type=float, default=0.001)
    p.add_argument("--device", default="cuda")
    p.add_argument("--step-stride", type=int, default=0,
                   help="Only evaluate checkpoints at multiples. 0 = all. Always includes last.")
    p.add_argument("--max-steps-per-ep", type=int, default=27000)
    return p.parse_args()


def get_step(name):
    m = re.match(r"ckpt_(\d+)\.pt$", name)
    return int(m.group(1)) if m else None


def select_ckpts(ckpts, stride):
    if stride <= 0:
        return ckpts
    last = ckpts[-1]
    selected = [c for c in ckpts if c[0] % stride == 0]
    if last not in selected:
        selected.append(last)
    return selected


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    variant_dirs = sorted([d for d in runs_dir.glob(args.pattern) if d.is_dir()])
    if not variant_dirs:
        print(f"No match {args.pattern} under {runs_dir}")
        sys.exit(1)

    print(f"Found {len(variant_dirs)} variants:", [d.name for d in variant_dirs])
    print(f"Settings: n_eps={args.n_episodes}, stride={args.step_stride}, "
          f"max_steps={args.max_steps_per_ep}")

    eval_env = make_eval_env(args.env, seed=12345)
    n_actions = eval_env.action_space.n

    for vdir in variant_dirs:
        with open(vdir / "config.json") as f:
            cfg = json.load(f)
        print(f"\n=== {vdir.name}  (double={cfg['double']}, dueling={cfg['dueling']}) ===")

        ckpts = sorted(
            [(get_step(p.name), p) for p in vdir.glob("ckpt_*.pt") if get_step(p.name) is not None],
            key=lambda x: x[0],
        )
        if not ckpts:
            print("  No numbered checkpoints. Skipping.")
            continue

        ckpts = select_ckpts(ckpts, args.step_stride)
        print(f"  Will evaluate {len(ckpts)} checkpoints: {[c[0] for c in ckpts]}")

        agent = DQNAgent(
            n_actions=n_actions, device=args.device, lr=1e-4, gamma=0.99,
            double=cfg["double"], dueling=cfg["dueling"],
        )

        out_csv = vdir / "reeval.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "n_episodes", "return_mean", "return_std", "return_min", "return_max"])
            for step, ckpt_path in ckpts:
                ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
                agent.online.load_state_dict(ckpt["online"])
                agent.online.eval()

                returns = []
                t0 = time.time()
                for _ in range(args.n_episodes):
                    obs, _ = eval_env.reset()
                    ep_ret = 0.0
                    for _ in range(args.max_steps_per_ep):
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

    eval_env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
