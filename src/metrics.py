from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def annualized_sharpe(returns: pd.Series) -> float:
    """Compute annualized Sharpe using daily returns and zero risk-free rate."""
    returns = returns.dropna()
    if returns.empty or returns.std(ddof=1) == 0:
        return np.nan
    return np.sqrt(TRADING_DAYS) * returns.mean() / returns.std(ddof=1)


def cagr(returns: pd.Series) -> float:
    """Compute compound annual growth rate from daily returns."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan

    cumulative = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    if years <= 0:
        return np.nan
    return cumulative ** (1 / years) - 1


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown from daily returns."""
    equity = (1 + returns.fillna(0)).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


def turnover(position: pd.Series) -> float:
    """Average daily absolute position change."""
    return position.diff().abs().fillna(0).mean()


def sharpe_standard_error(returns: pd.Series) -> float:
    """Approximate standard error of annualized Sharpe.

    This simple approximation is intentionally transparent for an MVP.
    A later version can use more robust formulas accounting for skew,
    kurtosis, and autocorrelation.
    """
    returns = returns.dropna()
    n = len(returns)
    if n <= 2:
        return np.nan
    return np.sqrt(TRADING_DAYS / n)
