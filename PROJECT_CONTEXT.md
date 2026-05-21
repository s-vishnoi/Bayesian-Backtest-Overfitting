# Project Context: Bayesian Backtest Overfitting

## What this project is

A quant research project studying how parameter search can create misleading trading backtests, and how empirical Bayes shrinkage can give more conservative estimates of strategy quality.

The public-facing framing should stay research-focused:

> This is a study of model selection risk and backtest overfitting, not a claim of profitable trading performance.

Avoid public README language like "hiring signal," "impress recruiters," or anything that makes the repo sound performative.

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

The observed Sharpe of a strategy is noisy. If many strategies are tested, the best observed Sharpe is likely inflated by luck.

Bayesian shrinkage combines:

1. A prior belief that most strategies are close to the strategy population average
2. The observed Sharpe from the backtest
3. The uncertainty in the Sharpe estimate

This produces a posterior/shrunken estimate of the strategy's latent true Sharpe.

In simple language:

> The strategy looked strong in the backtest, but after accounting for noise and the fact that many strategies were tested, its realistic Sharpe estimate is lower.

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

## Iteration workflow preference

When discussing a file change or strategy change, provide the updated replacement file quietly at the end with the same filename as the repo file.

Do not rename replacement artifacts unless explicitly requested.

Do not include extra push, copy, or terminal instructions after providing the replacement file unless explicitly requested.

For example:

- If `README.md` changes, provide only an updated `README.md`.
- If `PROJECT_CONTEXT.md` changes, provide only an updated `PROJECT_CONTEXT.md`.
- If `src/bayesian.py` changes, provide only an updated `bayesian.py`.

## README formatting note

README uses plain-text equations instead of LaTeX so formulas display reliably on GitHub.

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

> I am working on a GitHub project called `bayesian-backtest-overfitting`. It tests many moving-average crossover strategies on SPY, shows that the best in-sample Sharpe can be inflated, and applies empirical Bayes shrinkage to estimate more conservative latent Sharpe ratios. Please continue iteratively from the repo context in `PROJECT_CONTEXT.md`. Preserve the project structure and keep adding complete files or replacement files with original filenames. Public-facing language should stay research-focused and avoid sounding like the repo was created for hiring. After discussing a file or strategy change, quietly provide only the updated replacement file at the end, with no extra terminal or push instructions unless explicitly requested.
