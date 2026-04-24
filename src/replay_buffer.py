"""
Uniform replay buffer. Stores uint8 frames individually and reconstructs
stacks at sample time to save memory (4x vs storing stacked frames).
"""
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple, frame_stack: int = 4,
                 device: str = "cuda"):
        """
        obs_shape: shape of a SINGLE frame, e.g. (84, 84) after preprocessing.
        We store individual frames and rebuild the stacked observation at sample time.
        """
        self.capacity = capacity
        self.frame_stack = frame_stack
        self.device = device
        self.ptr = 0
        self.size = 0

        # uint8 storage — critical for memory
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        # `episode_start` flags let us avoid stacking across episode boundaries
        self.episode_starts = np.zeros(capacity, dtype=np.float32)

    def add(self, obs_single_frame: np.ndarray, action: int, reward: float,
            done: bool, episode_start: bool):
        """
        obs_single_frame: uint8, shape (84, 84) — the newest frame only
        """
        self.obs[self.ptr] = obs_single_frame
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.episode_starts[self.ptr] = float(episode_start)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _stack_frames(self, idx: int) -> np.ndarray:
        """Reconstruct the 4-frame stack ending at idx."""
        frames = []
        for offset in range(self.frame_stack - 1, -1, -1):
            i = (idx - offset) % self.capacity
            # If we cross an episode boundary going back, repeat the earliest valid frame
            frames.append(self.obs[i])
        return np.stack(frames, axis=0)  # (4, 84, 84)

    def sample(self, batch_size: int):
        # Avoid sampling the most recent frame_stack-1 transitions (no valid next_state)
        # and avoid boundary issues with the write pointer.
        high = self.size - 1
        low = self.frame_stack - 1
        idx = np.random.randint(low, high, size=batch_size)

        obs_batch = np.stack([self._stack_frames(i) for i in idx])
        next_obs_batch = np.stack([self._stack_frames((i + 1) % self.capacity) for i in idx])
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]

        # Convert to torch on device, normalize obs to [0,1] here
        obs_t = torch.from_numpy(obs_batch).to(self.device, non_blocking=True).float() / 255.0
        next_obs_t = torch.from_numpy(next_obs_batch).to(self.device, non_blocking=True).float() / 255.0
        actions_t = torch.from_numpy(actions).to(self.device, non_blocking=True)
        rewards_t = torch.from_numpy(rewards).to(self.device, non_blocking=True)
        dones_t = torch.from_numpy(dones).to(self.device, non_blocking=True)

        return obs_t, actions_t, rewards_t, next_obs_t, dones_t

    def __len__(self):
        return self.size
