from __future__ import annotations

from pathlib import Path
import pandas as pd
import yfinance as yf


def download_price_data(
    ticker: str = "SPY",
    start: str = "2010-01-01",
    end: str | None = None,
    data_dir: str | Path = "data",
) -> pd.DataFrame:
    """Download adjusted daily prices from Yahoo Finance and cache them locally."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / f"{ticker.lower()}_prices.csv"

    if cache_path.exists():
        prices = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        return prices

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No price data downloaded for {ticker}.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    prices = raw[["Close"]].rename(columns={"Close": "close"}).dropna()
    prices.to_csv(cache_path, index_label="Date")
    return prices
