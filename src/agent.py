"""
DQN Agent with toggles for:
  - double: Double DQN target (van Hasselt et al. 2016)
  - dueling: Dueling network head (Wang et al. 2016)

The 2x2 combination reproduces a slice of the Rainbow ablation study.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import build_network


class DQNAgent:
    def __init__(self, n_actions: int, device: str = "cuda",
                 lr: float = 6.25e-5, gamma: float = 0.99,
                 double: bool = False, dueling: bool = False,
                 grad_clip: float = 10.0):
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.double = double
        self.dueling = dueling
        self.grad_clip = grad_clip

        self.online = build_network(n_actions, dueling).to(device)
        self.target = build_network(n_actions, dueling).to(device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

        # Adam with eps=1.5e-4 is the Rainbow/Nature-DQN setting
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr, eps=1.5e-4)

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float) -> int:
        """obs: (4, 84, 84) uint8 or already float in [0,1]."""
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        if obs.dtype == np.uint8:
            obs = obs.astype(np.float32) / 255.0
        x = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        q = self.online(x)
        return int(q.argmax(dim=1).item())

    def compute_target(self, rewards, next_obs, dones):
        """
        Standard DQN target:  y = r + gamma * (1 - done) * max_a' Q_target(s', a')
        Double DQN target:    y = r + gamma * (1 - done) * Q_target(s', argmax_a' Q_online(s', a'))
        """
        with torch.no_grad():
            if self.double:
                # Action selection by online net, evaluation by target net
                next_actions = self.online(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            else:
                # Vanilla DQN: both selection and evaluation by target net
                next_q = self.target(next_obs).max(dim=1).values
            target = rewards + self.gamma * (1.0 - dones) * next_q
        return target

    def update(self, batch) -> dict:
        obs, actions, rewards, next_obs, dones = batch
        target = self.compute_target(rewards, next_obs, dones)

        q_values = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Huber loss (smooth L1) — standard in DQN
        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "q_mean": q_values.mean().item(),
            "target_mean": target.mean().item(),
        }

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path):
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "double": self.double,
            "dueling": self.dueling,
        }, path)
