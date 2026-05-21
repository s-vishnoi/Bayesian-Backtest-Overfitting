from __future__ import annotations

import numpy as np
import pandas as pd


def empirical_bayes_shrinkage(
    df: pd.DataFrame,
    observed_col: str = "train_sharpe",
    se_col: str = "train_sharpe_se",
) -> pd.DataFrame:
    """Apply empirical Bayes shrinkage to observed Sharpe estimates.

    The model is:

        observed_sharpe_i ~ Normal(true_sharpe_i, se_i^2)
        true_sharpe_i ~ Normal(mu, tau^2)

    We estimate mu and tau from the cross-section of strategies, then compute
    the posterior mean under a normal-normal model.
    """
    out = df.copy()

    observed = out[observed_col].astype(float)
    se = out[se_col].astype(float)

    mu = observed.mean(skipna=True)
    observed_var = observed.var(skipna=True, ddof=1)
    noise_var = np.nanmean(se**2)

    # Between-strategy variance cannot be negative.
    tau2 = max(observed_var - noise_var, 1e-8)

    weights = tau2 / (tau2 + se**2)
    posterior_mean = weights * observed + (1 - weights) * mu

    out["eb_group_mean"] = mu
    out["eb_tau2"] = tau2
    out["eb_weight"] = weights
    out["bayes_sharpe"] = posterior_mean
    out["shrinkage_amount"] = out[observed_col] - out["bayes_sharpe"]

    return out
