from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_train_sharpe_histogram(results: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(results["train_sharpe"].dropna(), bins=30)
    ax.set_title("Distribution of In-Sample Sharpes Across Strategy Trials")
    ax.set_xlabel("Train Sharpe")
    ax.set_ylabel("Number of strategies")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_top_train_test_bar(results: pd.DataFrame, output_path: str | Path, n: int = 10) -> None:
    output_path = Path(output_path)
    top = results.sort_values("train_sharpe", ascending=False).head(n).copy()
    labels = [f"{int(r.short_window)}/{int(r.long_window)}" for _, r in top.iterrows()]

    x = range(len(top))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], top["train_sharpe"], width=width, label="Train")
    ax.bar([i + width / 2 for i in x], top["test_sharpe"], width=width, label="Test")
    ax.set_title("Top In-Sample Strategies Often Weaken Out-of-Sample")
    ax.set_xlabel("MA short/long window")
    ax.set_ylabel("Sharpe")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_raw_vs_bayes_scatter(results: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(results["train_sharpe"], results["bayes_sharpe"], alpha=0.7)
    low = min(results["train_sharpe"].min(), results["bayes_sharpe"].min())
    high = max(results["train_sharpe"].max(), results["bayes_sharpe"].max())
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_title("Bayesian Shrinkage Pulls Extreme Sharpes Toward the Mean")
    ax.set_xlabel("Raw train Sharpe")
    ax.set_ylabel("Bayesian-shrunken Sharpe")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_equity_curve(backtest: pd.DataFrame, split_date: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    equity = (1 + backtest["net_return"].fillna(0)).cumprod()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.values)
    ax.axvline(pd.Timestamp(split_date), linestyle="--", label="Train/test split")
    ax.set_title("Equity Curve of Best In-Sample Strategy")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
