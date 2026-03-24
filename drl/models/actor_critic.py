"""Actor-Critic networks for the DRL portfolio optimizer."""

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Per-asset actor network.
    Input: [asset_features] (dim=F)
    Output: one logit per asset → softmax across all assets = portfolio weights.
    """

    def __init__(self, feat_dim: int = 11, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape [batch, N_assets, F] or [N_assets, F]
        Returns:
            logits: Shape [batch, N_assets] or [N_assets]
        """
        return self.net(x).squeeze(-1)


class Critic(nn.Module):
    """Shared critic network.
    Input: global state = concat(mean, std, max of all asset features)
    Output: state value estimate (scalar).
    """

    def __init__(self, feat_dim: int = 11, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        # Global state: mean + std + max of features = 3 * feat_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape [batch, N_assets, F] or [N_assets, F]
        Returns:
            value: Shape [batch] or scalar
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        global_state = torch.cat([
            x.mean(dim=1),
            x.std(dim=1),
            x.max(dim=1).values,
        ], dim=-1)
        return self.net(global_state).squeeze(-1)


class ActorCritic(nn.Module):
    """Combined Actor-Critic for a single PPO agent."""

    def __init__(self, feat_dim: int = 11):
        super().__init__()
        self.actor = Actor(feat_dim)
        self.critic = Critic(feat_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get portfolio weights (softmax of logits)."""
        logits = self.actor(x)
        return torch.softmax(logits, dim=-1)
