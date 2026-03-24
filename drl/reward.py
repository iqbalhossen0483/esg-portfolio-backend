"""Multi-objective reward functions for DRL portfolio optimization."""

import numpy as np


def reward_log_return(portfolio_value_t: float, portfolio_value_prev: float) -> float:
    """Agent 1: r1 = log(V_t / V_{t-1})"""
    if portfolio_value_prev <= 0:
        return 0.0
    return float(np.log(portfolio_value_t / portfolio_value_prev))


class DifferentialSharpeRatio:
    """Agent 2: Differential Sharpe Ratio (DSR).
    Incrementally estimates Sharpe ratio at each timestep.

    DSR_t = (B_{t-1} * r_t - 0.5 * A_{t-1} * r_t^2) / (B_{t-1} - A_{t-1}^2)^(3/2)
    where A_t = A_{t-1} + η(r_t - A_{t-1}), B_t = B_{t-1} + η(r_t^2 - B_{t-1})
    """

    def __init__(self, eta: float = 0.01):
        self.eta = eta
        self.A = 0.0  # EMA of returns
        self.B = 1e-8  # EMA of squared returns (small init to avoid div by zero)

    def __call__(self, r_t: float) -> float:
        denom = (self.B - self.A ** 2) ** 1.5
        if abs(denom) < 1e-12:
            dsr = 0.0
        else:
            dsr = (self.B * r_t - 0.5 * self.A * r_t ** 2) / denom

        # Update running averages
        self.A = self.A + self.eta * (r_t - self.A)
        self.B = self.B + self.eta * (r_t ** 2 - self.B)

        return float(dsr)

    def reset(self):
        self.A = 0.0
        self.B = 1e-8


def reward_max_drawdown(mdd: float) -> float:
    """Agent 3: r3 = -MDD (penalize drawdown)."""
    return float(-abs(mdd))


def combined_reward(
    portfolio_return: float,
    rolling_volatility: float,
    turnover: float,
    esg_bonus: float,
    mdd: float,
    mdd_threshold: float = 0.1,
    alpha: float = 0.05,
    beta: float = 0.001,
    gamma: float = 0.1,
    lam: float = 0.5,
) -> float:
    """Combined multi-objective reward.

    reward = R_portfolio - α*σ - β*turnover + λ*ESG - γ*max(0, MDD - threshold)

    Args:
        portfolio_return: Daily portfolio return.
        rolling_volatility: Rolling 20-day volatility.
        turnover: Sum of absolute weight changes.
        esg_bonus: Weighted ESG score of portfolio.
        mdd: Current max drawdown (negative value).
        mdd_threshold: Drawdown threshold before penalty kicks in.
        alpha: Risk penalty weight (default 0.05).
        beta: Turnover penalty weight (default 0.001).
        gamma: Drawdown penalty weight (default 0.1).
        lam: ESG importance weight (default 0.5, range [0,1]).

    Returns:
        Combined reward value.
    """
    excess_dd = max(0, abs(mdd) - mdd_threshold)
    reward = (
        portfolio_return
        - alpha * rolling_volatility
        - beta * turnover
        + lam * esg_bonus
        - gamma * excess_dd
    )
    return float(reward)
