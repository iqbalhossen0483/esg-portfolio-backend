"""Backtest and evaluate trained DRL models."""

import numpy as np
import pandas as pd

from .constraints import project_with_constraints
from .models.mappo import MAPPO
from core.metrics import (
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_annualized_return,
    compute_annualized_volatility,
)

import torch


def evaluate_model(
    model: MAPPO,
    features: np.ndarray,
    returns: np.ndarray,
    sector_ids: np.ndarray,
    esg_scores: np.ndarray,
    train_ratio: float = 0.7,
) -> dict:
    """Evaluate a trained MAPPO model on the test split.

    Args:
        model: Trained MAPPO model.
        features: Full feature tensor [T, N, F].
        returns: Full returns [T, N].
        sector_ids: Shape [N].
        esg_scores: Shape [N].
        train_ratio: Where the test split starts.

    Returns:
        Dict with all evaluation metrics.
    """
    T = features.shape[0]
    test_start = int(T * train_ratio)
    test_features = features[test_start:]
    test_returns = returns[test_start:]

    portfolio_returns = []
    weights_history = []
    prev_weights = np.ones(model.n_assets) / model.n_assets

    for t in range(len(test_features) - 1):
        obs = torch.tensor(test_features[t], dtype=torch.float32)
        weights = model.get_weights(obs).numpy()
        weights = project_with_constraints(weights, sector_ids)

        port_ret = float(np.dot(weights, test_returns[t + 1]))
        portfolio_returns.append(port_ret)
        weights_history.append(weights)
        prev_weights = weights

    if not portfolio_returns:
        return {"error": "No test data available"}

    port_returns_series = pd.Series(portfolio_returns)
    equity_curve = (1 + port_returns_series).cumprod()

    # Equal-weight baseline
    ew_returns = test_returns[1:].mean(axis=1)
    ew_series = pd.Series(ew_returns[:len(portfolio_returns)])
    ew_equity = (1 + ew_series).cumprod()

    avg_weights = np.mean(weights_history, axis=0)
    avg_esg = float(np.dot(avg_weights, esg_scores))

    return {
        "test_total_return": float(equity_curve.iloc[-1] - 1),
        "ew_total_return": float(ew_equity.iloc[-1] - 1) if len(ew_equity) > 0 else 0,
        "annualized_return": compute_annualized_return(port_returns_series),
        "annualized_volatility": compute_annualized_volatility(port_returns_series),
        "sharpe_ratio": compute_sharpe_ratio(port_returns_series),
        "sortino_ratio": compute_sortino_ratio(port_returns_series),
        "max_drawdown": compute_max_drawdown(equity_curve),
        "avg_portfolio_esg": avg_esg,
        "num_holdings": int((avg_weights > 0.01).sum()),
        "max_position": float(avg_weights.max()),
        "test_days": len(portfolio_returns),
    }
