"""CNN Fusion layer for RA-DRL.
Combines outputs from 3 PPO agents into final portfolio weights.
"""

import torch
import torch.nn as nn


class CNNFusion(nn.Module):
    """CNN fusion layer that combines 3 agents' weight proposals.

    Input: stacked actions from 3 agents → shape [batch, N_assets, 3]
    Output: final portfolio weights → shape [batch, N_assets]
    """

    def __init__(self, n_assets: int, n_agents: int = 3):
        super().__init__()
        self.n_assets = n_assets

        # Conv layer: learns which agent to trust for which asset
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=n_agents, out_channels=8, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=1),
            nn.ReLU(),
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(n_assets * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_assets),
        )

    def forward(self, agent_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_weights: Shape [batch, N_assets, 3] — stacked weights from 3 agents.
        Returns:
            weights: Shape [batch, N_assets] — final portfolio weights (softmax).
        """
        if agent_weights.dim() == 2:
            agent_weights = agent_weights.unsqueeze(0)

        # Conv expects [batch, channels, length] → transpose to [batch, 3, N_assets]
        x = agent_weights.permute(0, 2, 1)
        x = self.conv(x)  # [batch, 4, N_assets]
        x = x.reshape(x.size(0), -1)  # [batch, 4 * N_assets]
        x = self.fc(x)  # [batch, N_assets]

        return torch.softmax(x, dim=-1)
