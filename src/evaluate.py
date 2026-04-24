"""Evaluation: run N episodes with epsilon=0.001 (Nature DQN protocol)."""
import numpy as np


def evaluate(agent, env, n_episodes: int = 5, epsilon: float = 0.001,
             max_steps_per_ep: int = 27000) -> dict:
    returns = []
    lengths = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        # obs from FrameStackObservation is (stack, H, W)
        ep_ret = 0.0
        ep_len = 0
        for _ in range(max_steps_per_ep):
            action = agent.act(np.asarray(obs, dtype=np.uint8), epsilon)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            ep_len += 1
            if terminated or truncated:
                break
        returns.append(ep_ret)
        lengths.append(ep_len)
    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/length_mean": float(np.mean(lengths)),
    }
