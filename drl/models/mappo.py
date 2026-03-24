"""Multi-Agent PPO (MAPPO) implementation.
Wraps 3 Actor-Critic pairs + CNN fusion layer.
"""

import torch

from .actor_critic import ActorCritic
from .cnn_fusion import CNNFusion


class MAPPO:
    """Multi-Agent PPO with 3 agents + CNN fusion.

    Agent 1: Optimizes log returns
    Agent 2: Optimizes Differential Sharpe Ratio
    Agent 3: Optimizes negative max drawdown
    """

    def __init__(self, n_assets: int, feat_dim: int = 11, lr: float = 1e-3):
        self.n_assets = n_assets
        self.feat_dim = feat_dim

        # 3 independent actor-critic agents
        self.agent1 = ActorCritic(feat_dim)  # Return agent
        self.agent2 = ActorCritic(feat_dim)  # Sharpe agent
        self.agent3 = ActorCritic(feat_dim)  # Drawdown agent

        # CNN fusion layer
        self.fusion = CNNFusion(n_assets, n_agents=3)

        # Optimizers
        self.opt1 = torch.optim.Adam(self.agent1.parameters(), lr=lr)
        self.opt2 = torch.optim.Adam(self.agent2.parameters(), lr=lr)
        self.opt3 = torch.optim.Adam(self.agent3.parameters(), lr=lr)
        self.opt_fusion = torch.optim.Adam(self.fusion.parameters(), lr=lr)

    def get_weights(self, features: torch.Tensor) -> torch.Tensor:
        """Get final portfolio weights by fusing 3 agents' proposals.

        Args:
            features: Shape [N_assets, F] or [batch, N_assets, F]
        Returns:
            weights: Shape [N_assets] or [batch, N_assets]
        """
        with torch.no_grad():
            w1 = self.agent1.get_weights(features)
            w2 = self.agent2.get_weights(features)
            w3 = self.agent3.get_weights(features)

            # Stack: [batch, N_assets, 3]
            stacked = torch.stack([w1, w2, w3], dim=-1)
            final_weights = self.fusion(stacked)

        return final_weights.squeeze(0) if features.dim() == 2 else final_weights

    def save(self, path: str):
        """Save all model checkpoints."""
        torch.save({
            "agent1": self.agent1.state_dict(),
            "agent2": self.agent2.state_dict(),
            "agent3": self.agent3.state_dict(),
            "fusion": self.fusion.state_dict(),
            "n_assets": self.n_assets,
            "feat_dim": self.feat_dim,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MAPPO":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            n_assets=checkpoint["n_assets"],
            feat_dim=checkpoint["feat_dim"],
        )
        model.agent1.load_state_dict(checkpoint["agent1"])
        model.agent2.load_state_dict(checkpoint["agent2"])
        model.agent3.load_state_dict(checkpoint["agent3"])
        model.fusion.load_state_dict(checkpoint["fusion"])
        return model
