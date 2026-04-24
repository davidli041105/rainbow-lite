"""
Network definitions: Nature DQN conv backbone, with either
  - a standard Q-head (single linear stream), or
  - a dueling Q-head (value stream + advantage stream).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NatureCNN(nn.Module):
    """Conv backbone from Mnih et al. 2015. Input: (B, 4, 84, 84) in [0,1]."""

    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = 64 * 7 * 7  # 3136

    def forward(self, x):
        return self.conv(x).flatten(start_dim=1)


class DQN(nn.Module):
    """Standard DQN: conv features -> Linear(512) -> Linear(n_actions)."""

    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.backbone = NatureCNN(in_channels)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)  # (B, n_actions)


class DuelingDQN(nn.Module):
    """
    Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean_a' A(s,a')

    Subtracting the mean advantage is the "identifiability fix" from
    Wang et al. 2016 — without it, V and A are not uniquely recoverable.
    """

    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.backbone = NatureCNN(in_channels)
        # Value stream: scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )
        # Advantage stream: A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        feats = self.backbone(x)
        value = self.value_stream(feats)           # (B, 1)
        advantage = self.advantage_stream(feats)   # (B, n_actions)
        # Aggregation: mean-subtracted advantage
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


def build_network(n_actions: int, dueling: bool) -> nn.Module:
    return DuelingDQN(n_actions) if dueling else DQN(n_actions)
