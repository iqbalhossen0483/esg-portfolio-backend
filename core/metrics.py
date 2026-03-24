import numpy as np
import pandas as pd


def compute_sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Sharpe = (Rp - Rf) / σp. Annualized with √252."""
    excess = returns - risk_free / 252
    if excess.std() == 0:
        return 0.0
    daily_sharpe = excess.mean() / excess.std()
    return float(daily_sharpe * np.sqrt(252))


def compute_sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Sortino = (Rp - Rf) / σd where σd = std of negative returns only."""
    excess = returns - risk_free / 252
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float((excess.mean() / downside.std()) * np.sqrt(252))


def compute_calmar_ratio(returns: pd.Series) -> float:
    """Calmar = Annualized Return / |Max Drawdown|."""
    ann_return = compute_annualized_return(returns)
    mdd = compute_max_drawdown_from_returns(returns)
    if mdd == 0:
        return 0.0
    return float(ann_return / abs(mdd))


def compute_max_drawdown(prices: pd.Series) -> float:
    """MDD = (P_min - P_max) / P_max. Returns negative value."""
    if len(prices) == 0:
        return 0.0
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return float(drawdown.min())


def compute_max_drawdown_from_returns(returns: pd.Series) -> float:
    """Compute MDD from returns series by reconstructing equity curve."""
    equity = (1 + returns).cumprod()
    return compute_max_drawdown(equity)


def compute_annualized_return(returns: pd.Series) -> float:
    """R_annual = (1 + R_total)^(252/T) - 1."""
    if len(returns) == 0:
        return 0.0
    total_return = (1 + returns).prod() - 1
    n_days = len(returns)
    if n_days == 0 or total_return <= -1:
        return 0.0
    return float((1 + total_return) ** (252 / n_days) - 1)


def compute_annualized_volatility(returns: pd.Series) -> float:
    """σ_annual = σ_daily × √252."""
    if len(returns) == 0:
        return 0.0
    return float(returns.std() * np.sqrt(252))


def compute_momentum(prices: pd.Series, k: int) -> float:
    """Momentum = (P_t - P_{t-k}) / P_{t-k}."""
    if len(prices) < k + 1:
        return 0.0
    return float((prices.iloc[-1] - prices.iloc[-k - 1]) / prices.iloc[-k - 1])


def compute_all_metrics(
    symbol: str,
    prices: pd.Series,
    esg_composite: float | None = None,
    e_score: float | None = None,
    s_score: float | None = None,
    g_score: float | None = None,
) -> dict:
    """Compute all metrics for one company. Returns dict matching computed_metrics table."""
    returns = prices.pct_change().dropna()

    if len(returns) < 20:
        return {"symbol": symbol, "as_of_date": pd.Timestamp.now().date()}

    return {
        "symbol": symbol,
        "as_of_date": pd.Timestamp.now().date(),
        "annual_return": compute_annualized_return(returns),
        "annual_volatility": compute_annualized_volatility(returns),
        "sharpe_252d": compute_sharpe_ratio(returns),
        "sortino_252d": compute_sortino_ratio(returns),
        "calmar_ratio": compute_calmar_ratio(returns),
        "max_drawdown": compute_max_drawdown(prices),
        "momentum_20d": compute_momentum(prices, 20),
        "momentum_60d": compute_momentum(prices, 60),
        "avg_esg_composite": esg_composite,
        "avg_e_score": e_score,
        "avg_s_score": s_score,
        "avg_g_score": g_score,
    }
