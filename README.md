# Bayesian Backtest Overfitting

This project studies how parameter search can inflate the apparent performance of simple trading strategies, and how empirical Bayes shrinkage can produce more conservative estimates of strategy quality.

The goal is not to claim a profitable trading strategy. The goal is to build a transparent research workflow for measuring how much apparent performance may be due to noise, selection bias, and overfitting.

## Project overview

The first version tests a grid of moving-average crossover strategies on SPY. Each strategy uses a short moving average and a long moving average. The strategy is long when the short moving average is above the long moving average and flat otherwise.

The project compares:

1. Raw in-sample Sharpe ratios
2. Out-of-sample Sharpe ratios
3. Empirical Bayes-shrunken Sharpe estimates
4. Equity curves after transaction costs

## Why this matters

A standard backtest answers:

> Which strategy performed best historically?

That is not enough when many strategies or parameters have been tested. The best historical result may be unusually strong simply because many trials were attempted.

This project asks a more cautious question:

> After testing many strategy variants, how much of the best observed performance should be treated as reliable?

## Method

For each strategy `i`, the observed Sharpe ratio is treated as noisy evidence about an unknown true Sharpe ratio:

```text
observed_sharpe_i ~ Normal(true_sharpe_i, standard_error_i^2)
```

The true Sharpe ratios are assumed to come from a common population:

```text
true_sharpe_i ~ Normal(group_mean, between_strategy_variance)
```

This gives a shrinkage estimate:

```text
bayesian_sharpe_i =
    weight_i * observed_sharpe_i
    + (1 - weight_i) * group_mean
```

where:

```text
weight_i =
    between_strategy_variance
    / (between_strategy_variance + standard_error_i^2)
```

If a strategy's Sharpe estimate is noisy, the estimate is pulled more strongly toward the group average. If the estimate is more stable, it is trusted more.

## Current MVP

The current version includes:

- SPY daily data via `yfinance`
- Moving-average crossover parameter grid
- Train/test split
- Transaction costs
- Sharpe, CAGR, maximum drawdown, and turnover
- Empirical Bayes shrinkage of in-sample Sharpe ratios
- Diagnostic plots for overfitting and out-of-sample degradation

## Repository structure

```text
bayesian-backtest-overfitting/
├── README.md
├── PROJECT_CONTEXT.md
├── requirements.txt
├── run_project.py
├── src/
│   ├── data.py
│   ├── strategies.py
│   ├── backtester.py
│   ├── metrics.py
│   ├── bayesian.py
│   └── plots.py
├── data/
├── figures/
└── reports/
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
python run_project.py
```

The script downloads SPY data, runs the strategy grid, applies empirical Bayes shrinkage, and saves results and figures.

## Outputs

After running the project, the main outputs are:

```text
reports/strategy_results.csv
reports/final_summary.md
figures/train_sharpe_distribution.png
figures/top_train_vs_test_sharpe.png
figures/raw_vs_bayes_sharpe.png
figures/best_strategy_equity_curve.png
```

## Interpretation

This project should be interpreted as a study of model selection risk, not as investment advice. The central question is whether a strong in-sample result remains credible after accounting for multiple testing, transaction costs, and out-of-sample validation.
