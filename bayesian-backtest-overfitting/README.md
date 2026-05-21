# Bayesian Backtest Overfitting

This project demonstrates how easy it is to overfit a trading strategy through parameter search, and how Bayesian shrinkage can produce a more realistic estimate of strategy quality.

The goal is **not** to claim a profitable trading strategy. The goal is to show statistical discipline: when many strategies are tested, the best in-sample result is often inflated by noise.

## Project idea

I test a grid of moving-average crossover strategies on SPY. Each strategy uses a short moving average and a long moving average. The strategy is long when the short moving average is above the long moving average and flat otherwise.

The project compares:

1. Raw in-sample Sharpe
2. Out-of-sample Sharpe
3. Bayesian-shrunken Sharpe
4. Equity curves with transaction costs

The Bayesian component estimates a more realistic latent Sharpe ratio by shrinking noisy observed Sharpes toward the overall strategy population mean.

## Why this matters

A naive backtest asks:

> Which strategy had the best historical Sharpe?

A better research process asks:

> After testing many strategies, how much of the best Sharpe should I actually believe?

This project answers the second question.

## Method

For each strategy \(i\), the observed Sharpe is treated as noisy evidence about an unknown true Sharpe:

\[
\hat{S}_i \sim N(S_i, \sigma_i^2)
\]

The true Sharpes are assumed to come from a common population:

\[
S_i \sim N(\mu, \tau^2)
\]

This leads to a shrinkage estimate:

\[
E[S_i \mid \hat{S}_i] =
w_i \hat{S}_i + (1-w_i)\mu
\]

where:

\[
w_i = \frac{\tau^2}{\tau^2 + \sigma_i^2}
\]

If a strategy's Sharpe estimate is noisy, it is pulled strongly toward the group average. If the estimate is more reliable, it is trusted more.

## Current MVP

The first version includes:

- SPY daily data via `yfinance`
- Moving-average crossover strategy grid
- Train/test split
- Transaction costs
- Sharpe, CAGR, max drawdown, turnover
- Empirical Bayes shrinkage of Sharpe ratios
- Plots for overfitting diagnosis

## How to run

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
python run_project.py
```

The script downloads SPY data, runs the strategy grid, calculates Bayesian shrinkage estimates, and saves results/figures.

## Main hiring signal

This project shows that I understand a core problem in quant research: strategy discovery can easily become data mining. Instead of presenting a cherry-picked backtest, I quantify how much apparent alpha disappears after accounting for noise, parameter search, and out-of-sample validation.
