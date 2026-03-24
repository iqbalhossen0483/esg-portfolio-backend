"""Gym-style portfolio trading environment for DRL training."""

import numpy as np

from .constraints import project_with_constraints
from .reward import (
    combined_reward,
    reward_log_return,
    reward_max_drawdown,
    DifferentialSharpeRatio,
)


class PortfolioEnv:
    """Portfolio trading environment.

    State: feature tensor at time t → [N_assets, F]
    Action: raw logits → projected to constrained weights
    Reward: multi-objective (return, risk, ESG, drawdown)
    """

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        sector_ids: np.ndarray,
        esg_scores: np.ndarray,
        esg_lambda: float = 0.5,
    ):
        """
        Args:
            features: Shape [T, N, F] — feature tensor.
            returns: Shape [T, N] — daily returns.
            sector_ids: Shape [N] — integer sector ID per asset.
            esg_scores: Shape [N] — normalized ESG scores per asset.
            esg_lambda: ESG importance weight [0, 1].
        """
        self.features = features
        self.returns = returns
        self.sector_ids = sector_ids
        self.esg_scores = esg_scores
        self.esg_lambda = esg_lambda

        self.T, self.N, self.F = features.shape
        self.t = 0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.prev_weights = np.ones(self.N) / self.N
        self.return_history = []
        self.dsr = DifferentialSharpeRatio()

    def reset(self) -> np.ndarray:
        """Reset environment. Returns initial observation."""
        self.t = 0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.prev_weights = np.ones(self.N) / self.N
        self.return_history = []
        self.dsr.reset()
        return self.features[0]

    def step(self, raw_weights: np.ndarray) -> tuple[np.ndarray, dict, bool]:
        """Take one step.

        Args:
            raw_weights: Raw weight array [N]. Will be projected to satisfy constraints.

        Returns:
            next_obs: Shape [N, F]
            rewards: Dict with 'combined', 'return', 'sharpe', 'drawdown' rewards.
            done: True if episode finished.
        """
        # Project weights
        weights = project_with_constraints(raw_weights, self.sector_ids)

        # Compute portfolio return
        if self.t + 1 >= self.T:
            return self.features[self.t], {}, True

        port_return = float(np.dot(weights, self.returns[self.t + 1]))

        # Update portfolio value
        prev_value = self.portfolio_value
        self.portfolio_value *= (1 + port_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # Compute metrics
        turnover = float(np.abs(weights - self.prev_weights).sum())
        self.return_history.append(port_return)
        rolling_vol = float(np.std(self.return_history[-20:])) if len(self.return_history) > 1 else 0.0
        mdd = (self.portfolio_value - self.peak_value) / self.peak_value if self.peak_value > 0 else 0.0
        esg_bonus = float(np.dot(weights, self.esg_scores))

        # Compute rewards
        r_combined = combined_reward(
            portfolio_return=port_return,
            rolling_volatility=rolling_vol,
            turnover=turnover,
            esg_bonus=esg_bonus,
            mdd=mdd,
            lam=self.esg_lambda,
        )
        r_return = reward_log_return(self.portfolio_value, prev_value)
        r_sharpe = self.dsr(port_return)
        r_drawdown = reward_max_drawdown(mdd)

        rewards = {
            "combined": r_combined,
            "return": r_return,
            "sharpe": r_sharpe,
            "drawdown": r_drawdown,
            "portfolio_return": port_return,
            "portfolio_value": self.portfolio_value,
            "turnover": turnover,
            "mdd": mdd,
            "esg": esg_bonus,
        }

        self.prev_weights = weights
        self.t += 1
        done = self.t >= self.T - 1
        next_obs = self.features[min(self.t, self.T - 1)]

        return next_obs, rewards, done
