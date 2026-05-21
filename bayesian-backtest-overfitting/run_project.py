from __future__ import annotations

from pathlib import Path

from src.data import download_price_data
from src.backtester import run_ma_grid
from src.bayesian import empirical_bayes_shrinkage
from src.plots import (
    save_train_sharpe_histogram,
    save_top_train_test_bar,
    save_raw_vs_bayes_scatter,
    save_equity_curve,
)


def main() -> None:
    data_dir = Path("data")
    figure_dir = Path("figures")
    report_dir = Path("reports")

    data_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    ticker = "SPY"
    split_date = "2019-01-01"

    short_windows = list(range(5, 65, 5))
    long_windows = list(range(50, 255, 10))

    prices_df = download_price_data(ticker=ticker, start="2010-01-01", data_dir=data_dir)
    prices = prices_df["close"]

    results, backtests = run_ma_grid(
        prices=prices,
        short_windows=short_windows,
        long_windows=long_windows,
        split_date=split_date,
        transaction_cost_bps=1.0,
    )

    results = empirical_bayes_shrinkage(results)
    results = results.sort_values("train_sharpe", ascending=False)

    results_path = report_dir / "strategy_results.csv"
    results.to_csv(results_path, index=False)

    best = results.iloc[0]
    best_key = (int(best["short_window"]), int(best["long_window"]))
    best_backtest = backtests[best_key]

    save_train_sharpe_histogram(results, figure_dir / "train_sharpe_distribution.png")
    save_top_train_test_bar(results, figure_dir / "top_train_vs_test_sharpe.png")
    save_raw_vs_bayes_scatter(results, figure_dir / "raw_vs_bayes_sharpe.png")
    save_equity_curve(best_backtest, split_date, figure_dir / "best_strategy_equity_curve.png")

    summary = f"""
# MVP Results Summary

## Best in-sample strategy

- Ticker: {ticker}
- Short window: {best_key[0]}
- Long window: {best_key[1]}
- Train Sharpe: {best['train_sharpe']:.3f}
- Bayesian-shrunken Sharpe: {best['bayes_sharpe']:.3f}
- Test Sharpe: {best['test_sharpe']:.3f}
- Train CAGR: {best['train_cagr']:.3%}
- Test CAGR: {best['test_cagr']:.3%}
- Train max drawdown: {best['train_max_drawdown']:.3%}
- Test max drawdown: {best['test_max_drawdown']:.3%}

## Interpretation

The best in-sample strategy should not be interpreted as a discovered trading edge by itself. The purpose of this MVP is to compare the raw train Sharpe against the Bayesian-shrunken estimate and the out-of-sample Sharpe.

A strong final project should emphasize whether the Bayesian-shrunken estimate is more realistic than the raw in-sample estimate.
"""

    (report_dir / "final_summary.md").write_text(summary.strip() + "\n", encoding="utf-8")

    print(f"Saved results to {results_path}")
    print(f"Best strategy: MA({best_key[0]}, {best_key[1]})")
    print(f"Train Sharpe: {best['train_sharpe']:.3f}")
    print(f"Bayesian-shrunken Sharpe: {best['bayes_sharpe']:.3f}")
    print(f"Test Sharpe: {best['test_sharpe']:.3f}")


if __name__ == "__main__":
    main()
