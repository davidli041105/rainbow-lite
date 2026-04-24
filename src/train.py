"""
Training loop for DQN / Double DQN / Dueling DQN / Double+Dueling DQN.

Usage:
    python train.py --env ALE/Pong-v5 --total-frames 3000000 \
                    --double --dueling --exp-name pong_both --seed 0
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from atari_wrappers import make_atari_env, make_eval_env
from replay_buffer import ReplayBuffer
from agent import DQNAgent
from evaluate import evaluate


def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    frac = min(step / duration, 1.0)
    return start + frac * (end - start)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="ALE/Pong-v5")
    p.add_argument("--exp-name", type=str, required=True)
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=0)
    # algorithm toggles
    p.add_argument("--double", action="store_true")
    p.add_argument("--dueling", action="store_true")
    # hyperparameters (defaults tuned for ~3M frame budget)
    p.add_argument("--total-frames", type=int, default=3_000_000,
                   help="Total env frames (after frame-skip of 4).")
    p.add_argument("--buffer-size", type=int, default=300_000,
                   help="Smaller than paper's 1M to save RAM on short runs.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=6.25e-5)
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--train-freq", type=int, default=4,
                   help="One gradient update every N env steps.")
    p.add_argument("--target-update-freq", type=int, default=8_000,
                   help="Hard target update every N gradient steps.")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay-frames", type=int, default=250_000)
    p.add_argument("--eval-freq", type=int, default=100_000,
                   help="Evaluate every N env steps.")
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--log-freq", type=int, default=1_000)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: double={args.double}, dueling={args.dueling}, env={args.env}")

    log_dir = Path(args.log_dir) / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(log_dir=str(log_dir))
    csv_path = log_dir / "metrics.csv"
    csv_file = open(csv_path, "w")
    csv_file.write("step,episode_return,eval_return,loss,q_mean,epsilon\n")

    # Envs
    env = make_atari_env(args.env, seed=args.seed)
    eval_env = make_eval_env(args.env, seed=args.seed + 1000)
    n_actions = env.action_space.n
    print(f"Action space: {n_actions} actions — {env.unwrapped.get_action_meanings()}")

    # Agent + buffer
    agent = DQNAgent(
        n_actions=n_actions, device=device, lr=args.lr, gamma=args.gamma,
        double=args.double, dueling=args.dueling,
    )
    n_params = sum(p.numel() for p in agent.online.parameters())
    print(f"Online net params: {n_params:,}")

    buffer = ReplayBuffer(
        capacity=args.buffer_size, obs_shape=(84, 84), frame_stack=4, device=device,
    )

    # Training
    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    recent_returns = []
    grad_steps = 0
    t0 = time.time()

    for step in range(1, args.total_frames + 1):
        epsilon = linear_schedule(args.eps_start, args.eps_end,
                                  args.eps_decay_frames, step)

        # obs is (4, 84, 84) from FrameStackObservation
        action = agent.act(np.asarray(obs, dtype=np.uint8), epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store only the NEWEST frame (last in stack) — buffer rebuilds stacks at sample time
        newest_frame = np.asarray(next_obs, dtype=np.uint8)[-1]
        buffer.add(newest_frame, action, float(reward), bool(terminated),
                   episode_start=(episode_length == 0))

        episode_return += reward
        episode_length += 1

        if done:
            recent_returns.append(episode_return)
            if len(recent_returns) > 100:
                recent_returns.pop(0)
            writer.add_scalar("train/episode_return", episode_return, step)
            writer.add_scalar("train/episode_length", episode_length, step)
            if len(recent_returns) >= 10:
                writer.add_scalar("train/return_mean_100",
                                  float(np.mean(recent_returns)), step)
            csv_file.write(f"{step},{episode_return},,,,{epsilon:.4f}\n")
            csv_file.flush()
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
        else:
            obs = next_obs

        # Learning
        if step > args.learning_starts and step % args.train_freq == 0:
            batch = buffer.sample(args.batch_size)
            stats = agent.update(batch)
            grad_steps += 1

            if grad_steps % args.target_update_freq == 0:
                agent.sync_target()

            if step % args.log_freq == 0:
                writer.add_scalar("train/loss", stats["loss"], step)
                writer.add_scalar("train/q_mean", stats["q_mean"], step)
                writer.add_scalar("train/target_mean", stats["target_mean"], step)
                writer.add_scalar("train/epsilon", epsilon, step)
                sps = step / (time.time() - t0)
                writer.add_scalar("train/steps_per_sec", sps, step)
                avg_ret = float(np.mean(recent_returns)) if recent_returns else 0.0
                print(f"step={step:>8d}  eps={epsilon:.3f}  "
                      f"loss={stats['loss']:.4f}  qmean={stats['q_mean']:.3f}  "
                      f"avg_ret100={avg_ret:.2f}  sps={sps:.0f}")

        # Evaluation
        if step % args.eval_freq == 0 and step > args.learning_starts:
            eval_stats = evaluate(agent, eval_env, n_episodes=args.eval_episodes)
            for k, v in eval_stats.items():
                writer.add_scalar(k, v, step)
            print(f"[EVAL @ step {step}] "
                  f"return={eval_stats['eval/return_mean']:.2f} "
                  f"± {eval_stats['eval/return_std']:.2f}")
            csv_file.write(f"{step},,{eval_stats['eval/return_mean']},,,\n")
            csv_file.flush()
            agent.save(log_dir / f"ckpt_{step}.pt")

    # Final
    agent.save(log_dir / "ckpt_final.pt")
    csv_file.close()
    writer.close()
    env.close()
    eval_env.close()
    print(f"Done. Total wall-clock: {(time.time() - t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
