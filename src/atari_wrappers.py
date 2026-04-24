"""
Atari preprocessing wrappers following Mnih et al. 2015.
Built on top of gymnasium + ale-py.
"""
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

gym.register_envs(ale_py)


class FireOnResetWrapper(gym.Wrapper):
    """Press FIRE on reset for games that require it (Breakout etc.)."""

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)  # some games need this
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ClipRewardWrapper(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1} — standard in DQN for stability across games."""

    def reward(self, reward):
        return float(np.sign(reward))


def make_atari_env(env_id: str, seed: int = 0, clip_rewards: bool = True,
                   episode_life: bool = True, frame_stack: int = 4):
    """
    Build an Atari env with DeepMind-style preprocessing.

    env_id example: "ALE/Pong-v5", "ALE/Breakout-v5"
    """
    # frameskip=1 because AtariPreprocessing handles frame skipping itself
    env = gym.make(env_id, frameskip=1, repeat_action_probability=0.0,
                   full_action_space=False)

    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=episode_life,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,  # keep uint8 for memory efficiency in replay buffer
    )

    # Fire-on-reset for games that need it
    action_meanings = env.unwrapped.get_action_meanings()
    if "FIRE" in action_meanings and len(action_meanings) >= 3:
        try:
            env = FireOnResetWrapper(env)
        except AssertionError:
            pass

    if clip_rewards:
        env = ClipRewardWrapper(env)

    env = FrameStackObservation(env, stack_size=frame_stack)
    env.action_space.seed(seed)
    return env


def make_eval_env(env_id: str, seed: int = 0):
    """Eval env: no reward clipping, no episode-on-life-loss (for true episode returns)."""
    return make_atari_env(env_id, seed=seed, clip_rewards=False,
                          episode_life=False, frame_stack=4)
