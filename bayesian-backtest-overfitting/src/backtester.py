from __future__ import annotations

import pandas as pd

from .metrics import annualized_sharpe, cagr, max_drawdown, turnover, sharpe_standard_error
from .strategies import moving_average_signal


def backtest_signal(
    prices: pd.Series,
    position: pd.Series,
    transaction_cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Backtest a daily long/flat signal with turnover-based transaction costs."""
    asset_returns = prices.pct_change().fillna(0.0)
    gross_returns = position.reindex(prices.index).fillna(0.0) * asset_returns

    cost_per_unit_turnover = transaction_cost_bps / 10_000
    costs = position.diff().abs().fillna(position.abs()) * cost_per_unit_turnover

    net_returns = gross_returns - costs

    return pd.DataFrame(
        {
            "asset_return": asset_returns,
            "position": position,
            "gross_return": gross_returns,
            "transaction_cost": costs,
            "net_return": net_returns,
        }
    )


def summarize_returns(returns: pd.Series, position: pd.Series) -> dict[str, float]:
    """Summarize strategy performance."""
    return {
        "sharpe": annualized_sharpe(returns),
        "sharpe_se": sharpe_standard_error(returns),
        "cagr": cagr(returns),
        "max_drawdown": max_drawdown(returns),
        "turnover": turnover(position),
    }


def run_ma_grid(
    prices: pd.Series,
    short_windows: list[int],
    long_windows: list[int],
    split_date: str = "2019-01-01",
    transaction_cost_bps: float = 1.0,
) -> tuple[pd.DataFrame, dict[tuple[int, int], pd.DataFrame]]:
    """Run moving-average strategies across a parameter grid."""
    results: list[dict[str, float | int]] = []
    backtests: dict[tuple[int, int], pd.DataFrame] = {}

    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue

            position = moving_average_signal(prices, short_window, long_window)
            bt = backtest_signal(prices, position, transaction_cost_bps=transaction_cost_bps)
            backtests[(short_window, long_window)] = bt

            train_mask = bt.index < split_date
            test_mask = bt.index >= split_date

            train_summary = summarize_returns(
                bt.loc[train_mask, "net_return"],
                bt.loc[train_mask, "position"],
            )
            test_summary = summarize_returns(
                bt.loc[test_mask, "net_return"],
                bt.loc[test_mask, "position"],
            )

            results.append(
                {
                    "short_window": short_window,
                    "long_window": long_window,
                    "train_sharpe": train_summary["sharpe"],
                    "train_sharpe_se": train_summary["sharpe_se"],
                    "train_cagr": train_summary["cagr"],
                    "train_max_drawdown": train_summary["max_drawdown"],
                    "train_turnover": train_summary["turnover"],
                    "test_sharpe": test_summary["sharpe"],
                    "test_cagr": test_summary["cagr"],
                    "test_max_drawdown": test_summary["max_drawdown"],
                    "test_turnover": test_summary["turnover"],
                }
            )

    return pd.DataFrame(results), backtests
