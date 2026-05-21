# Project Context: Bayesian Backtest Overfitting

## What this project is

A quant research portfolio project showing how parameter search can create misleading trading backtests, and how empirical Bayes shrinkage can give more realistic estimates of strategy quality.

The project is intentionally framed as an **anti-overfitting research tool**, not as a trading bot.

## Core hiring message

> Most student quant projects try to maximize backtest performance. This project does the more mature thing: it measures how much of apparent performance disappears after accounting for parameter search, noise, transaction costs, and out-of-sample validation.

## Current MVP scope

- Asset: SPY
- Strategy family: moving-average crossover
- Parameter search: short moving average × long moving average grid
- Train/test split: configurable, default split date is 2019-01-01
- Transaction costs: included as a simple turnover-based cost
- Bayesian method: empirical Bayes shrinkage of observed train Sharpe ratios
- Outputs:
  - CSV of all strategy results
  - histogram of train Sharpes
  - top strategy train/test comparison
  - raw vs Bayesian-shrunken Sharpe scatter
  - equity curve for the best in-sample strategy

## Bayesian idea in plain English

The observed Sharpe of a strategy is noisy. If we test many strategies, the best observed Sharpe is likely inflated by luck.

Bayesian shrinkage combines:

1. A prior belief that most strategies are mediocre
2. The observed Sharpe from the backtest
3. The uncertainty in the Sharpe estimate

This produces a posterior/shrunken estimate of the strategy's latent true Sharpe.

In simple language:

> The strategy looked great in the backtest, but after accounting for noise and the fact that many strategies were tested, its realistic Sharpe is probably lower.

## Current file map

```text
bayesian-backtest-overfitting/
├── README.md
├── PROJECT_CONTEXT.md
├── requirements.txt
├── run_project.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── strategies.py
│   ├── metrics.py
│   ├── backtester.py
│   ├── bayesian.py
│   └── plots.py
├── data/
├── figures/
└── reports/
```

## Next improvements

Priority 1:
- Run the MVP and inspect whether the results/plots are coherent.
- Improve README with actual generated plots.
- Add a concise `reports/final_summary.md`.

Priority 2:
- Add bootstrap uncertainty for out-of-sample Sharpe.
- Compare empirical Bayes estimate to test Sharpe.
- Add heatmaps of Sharpe by moving-average parameters.
- Add a "researcher degrees of freedom" experiment showing how more trials inflate the best raw Sharpe.

Priority 3:
- Add a second strategy family, such as mean reversion or volatility breakout.
- Add more ETFs.
- Add walk-forward validation.
- Add Deflated Sharpe Ratio comparison.

## How to continue in a new chat

Paste this prompt:

> I am working on a GitHub portfolio project called `bayesian-backtest-overfitting`. It tests many moving-average crossover strategies on SPY, shows that the best in-sample Sharpe is inflated, and applies empirical Bayes shrinkage to estimate more realistic latent Sharpe ratios. Please continue iteratively from the repo context in `PROJECT_CONTEXT.md`. Preserve the project structure and keep adding complete files or patches. The goal is to impress quant hiring teams by showing statistical discipline around backtest overfitting.

## Current push workflow

After unzipping the project:

```bash
cd bayesian-backtest-overfitting
git init
git add .
git commit -m "Initial Bayesian backtest overfitting MVP"
gh repo create bayesian-backtest-overfitting --public --source=. --remote=origin --push
```

For later iterations:

```bash
git add .
git commit -m "Describe the change"
git push
```
