"""Feature engineering for DRL state space.
Computes technical indicators and builds the feature tensor [T, N, F].
"""

import numpy as np
import pandas as pd


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI = 100 - (100 / (1 + Avg_Gain / Avg_Loss))"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd(prices: pd.Series) -> pd.Series:
    """MACD histogram = MACD_line - Signal_line"""
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal
    return histogram


def compute_ema(prices: pd.Series, period: int) -> pd.Series:
    """EMA_t = α × P_t + (1-α) × EMA_{t-1}, α = 2/(period+1)"""
    return prices.ewm(span=period, adjust=False).mean()


def compute_bollinger_pct_b(prices: pd.Series, period: int = 20) -> pd.Series:
    """%B = (Price - Lower) / (Upper - Lower)"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    band_width = upper - lower
    pct_b = (prices - lower) / band_width.replace(0, np.nan)
    return pct_b.fillna(0.5)


def build_feature_tensor(
    prices_df: pd.DataFrame,
    esg_scores: dict[str, tuple[float, float]],
    symbols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build the DRL feature tensor from price data and ESG scores.

    Args:
        prices_df: DataFrame with columns=symbols, index=dates, values=close prices.
        esg_scores: Dict of {symbol: (esg_provider1, esg_provider2)} normalized 0-1.
        symbols: List of selected symbols (after screening).

    Returns:
        X: Feature tensor of shape [T, N, F] where F=11 features.
        R: Returns array of shape [T, N].
        dates: DatetimeIndex of length T.

    Feature dimensions (F=11):
        0: Daily returns
        1: 20-day momentum
        2: 60-day momentum
        3: 20-day rolling volatility
        4: 60-day drawdown
        5: ESG Score 1 (normalized)
        6: ESG Score 2 (normalized)
        7: MACD histogram
        8: RSI
        9: Bollinger %B
        10: Volume change (if available, else 0)
    """
    prices = prices_df[symbols].copy()
    returns = prices.pct_change()
    mom20 = prices.pct_change(20)
    mom60 = prices.pct_change(60)
    vol20 = returns.rolling(20).std()
    dd60 = prices / prices.rolling(60).max() - 1
    macd = pd.DataFrame({s: compute_macd(prices[s]) for s in symbols})
    rsi = pd.DataFrame({s: compute_rsi(prices[s]) / 100.0 for s in symbols})
    boll = pd.DataFrame({s: compute_bollinger_pct_b(prices[s]) for s in symbols})

    # ESG scores are static per company, tiled across time
    esg1 = np.array([esg_scores.get(s, (0.5, 0.5))[0] for s in symbols])
    esg2 = np.array([esg_scores.get(s, (0.5, 0.5))[1] for s in symbols])

    T = len(returns)
    N = len(symbols)

    feature_tensor = np.stack([
        returns.values,                                    # 0: returns
        mom20.values,                                      # 1: momentum 20d
        mom60.values,                                      # 2: momentum 60d
        vol20.values,                                      # 3: volatility 20d
        dd60.values,                                       # 4: drawdown 60d
        np.tile(esg1, (T, 1)),                            # 5: ESG score 1
        np.tile(esg2, (T, 1)),                            # 6: ESG score 2
        macd.values,                                       # 7: MACD
        rsi.values,                                        # 8: RSI
        boll.values,                                       # 9: Bollinger %B
        np.zeros((T, N)),                                  # 10: placeholder
    ], axis=-1)  # Shape: [T, N, 11]

    # Remove rows with NaN (warmup period)
    valid_rows = ~np.isnan(feature_tensor).any(axis=(1, 2))
    first_valid = np.argmax(valid_rows) if valid_rows.any() else 0

    X = np.nan_to_num(feature_tensor[first_valid:], nan=0.0)
    R = np.nan_to_num(returns.values[first_valid:], nan=0.0)
    dates = returns.index[first_valid:]

    return X, R, dates
