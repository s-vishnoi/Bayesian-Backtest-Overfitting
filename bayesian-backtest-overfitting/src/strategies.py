from __future__ import annotations

import itertools
import pandas as pd


def moving_average_signal(
    prices: pd.Series,
    short_window: int,
    long_window: int,
) -> pd.Series:
    """Create a long/flat moving-average crossover signal."""
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window.")

    short_ma = prices.rolling(short_window).mean()
    long_ma = prices.rolling(long_window).mean()
    signal = (short_ma > long_ma).astype(float)

    # Trade on next day's close after observing today's signal.
    return signal.shift(1).fillna(0.0)


def parameter_grid(
    short_windows: list[int],
    long_windows: list[int],
) -> list[tuple[int, int]]:
    """Return valid short/long moving-average parameter pairs."""
    return [(s, l) for s, l in itertools.product(short_windows, long_windows) if s < l]
