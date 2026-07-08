"""Reliability of individual recovery differences.

- ``split_half``: odd/even event split within unit, correlate per-unit axis
  across halves, Spearman-Brown corrected (C1 step 3 D8 / c1_03b).
- ``fit_mlm``: multilevel random-effects model giving shrunk per-unit effects
  and random-effect variances (C1 c1_03b binomial MLM). Requires the optional
  Bayesian backend: ``pip install recoverr[bayes]``. Ported in M2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def spearman_brown(r: float) -> float:
    """Spearman-Brown prophecy for a two-half split."""
    if r is None or not np.isfinite(r) or r <= -1:
        return np.nan
    return 2 * r / (1 + r)


def split_half(
    ev_windows: pd.DataFrame,
    plc_windows: pd.DataFrame,
    axis: str = "depth",
    span: tuple[int, int] = (1, 5),
    min_events: int = 10,
    method: str = "spearman",
) -> dict:
    """Split-half reliability of a per-unit deviation axis.

    ``axis`` is descriptive; ``span`` selects the rel_pos range that defines it
    (e.g. (1, 5) for depth, (1, window) for AUC).
    """
    order = (ev_windows[["unit", "event_id"]].drop_duplicates()
             .assign(k=lambda d: d.groupby("unit").cumcount()))
    ev = ev_windows.merge(order, on=["unit", "event_id"])
    n_ev = order.groupby("unit")["k"].max() + 1
    ok = n_ev[n_ev >= min_events].index
    base = plc_windows[plc_windows["unit"].isin(ok) & plc_windows["rel_pos"].between(*span)]
    bmean = base.groupby("unit")["outcome"].mean()
    halves = {}
    for h in (0, 1):
        sub = ev[(ev["unit"].isin(ok)) & (ev["k"] % 2 == h) & ev["rel_pos"].between(*span)]
        halves[h] = sub.groupby("unit")["outcome"].mean() - bmean
    both = pd.concat(halves, axis=1).dropna()
    r = both[0].corr(both[1], method=method) if len(both) >= 3 else np.nan
    return {"axis": axis, "n_units": int(len(both)), "r_split": r, "r_sb": spearman_brown(r)}


def fit_mlm(
    ev_windows: pd.DataFrame,
    plc_windows: pd.DataFrame,
    seed: int = 0,
    mcmc: dict | None = None,
    target_accept: float = 0.95,
) -> dict:
    """Multilevel random-effects reliability of the event effect (binomial MLM).

    Aggregates each unit's event vs placebo observations to binomial counts and
    fits, in NumPyro:

        logit P(outcome=1) = b0 + u_user + is_event * (b_ev + u_user_ev)

    The random-effect SDs (``sigma_user_ev`` in particular) quantify reliable
    between-person differences in the post-failure effect; ``rhat`` checks
    convergence. Ported from c1_03b_fallback_mlm. Needs ``recoverr[bayes]``.
    Expects a binary ``outcome`` (0/1).
    """
    try:
        import jax
        import numpyro
        import numpyro.distributions as dist
        from numpyro.diagnostics import gelman_rubin
        from numpyro.infer import MCMC, NUTS
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "fit_mlm needs the Bayesian backend: pip install 'recoverr[bayes]'"
        ) from e

    mcmc = mcmc or {"num_warmup": 1000, "num_samples": 1000, "num_chains": 2}
    df = pd.concat([ev_windows.assign(is_ev=1), plc_windows.assign(is_ev=0)], ignore_index=True)
    agg = df.groupby(["unit", "is_ev"])["outcome"].agg(["sum", "size"]).reset_index()
    units, uidx = np.unique(agg["unit"], return_inverse=True)
    k = np.rint(agg["sum"].to_numpy()).astype(np.int32)
    n = agg["size"].to_numpy().astype(np.int32)
    is_ev = agg["is_ev"].to_numpy().astype(np.float32)

    def model(uidx, is_ev, n, k=None):
        b0 = numpyro.sample("b0", dist.Normal(0, 2))
        b_ev = numpyro.sample("b_ev", dist.Normal(0, 2))
        s_u = numpyro.sample("sigma_user", dist.HalfNormal(1.0))
        s_ue = numpyro.sample("sigma_user_ev", dist.HalfNormal(1.0))
        with numpyro.plate("units", len(units)):
            zu = numpyro.sample("z_user", dist.Normal(0, 1))
            zue = numpyro.sample("z_user_ev", dist.Normal(0, 1))
        eta = b0 + zu[uidx] * s_u + is_ev * (b_ev + zue[uidx] * s_ue)
        numpyro.sample("obs", dist.Binomial(total_count=n, logits=eta), obs=k)

    runner = MCMC(NUTS(model, target_accept_prob=target_accept),
                  **mcmc, chain_method="sequential", progress_bar=False)
    runner.run(jax.random.PRNGKey(seed), uidx, is_ev, n, k)
    by_chain = runner.get_samples(group_by_chain=True)
    flat = runner.get_samples()
    rows = []
    for key in ("b0", "b_ev", "sigma_user", "sigma_user_ev"):
        x = np.asarray(by_chain[key])
        rhat = float(gelman_rubin(x)) if x.shape[0] >= 2 else float("nan")
        rows.append((key, float(x.mean()), float(np.quantile(x, 0.025)),
                     float(np.quantile(x, 0.975)), rhat))
    summary = pd.DataFrame(rows, columns=["param", "mean", "ci2.5", "ci97.5", "rhat"])
    user_effects = pd.DataFrame({
        "unit": units,
        "u_ev_hat": np.asarray(flat["z_user_ev"]).mean(0) * np.asarray(flat["sigma_user_ev"]).mean(),
    })
    return {"summary": summary, "user_effects": user_effects}
