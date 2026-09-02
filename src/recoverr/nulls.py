"""Permutation null for the depth deviation (C1 step 3 D9).

Shuffles event/placebo labels *within unit* to build a null distribution for the
early-window deviation, ruling out regression-to-the-mean artifacts. The
shuffle is vectorized (one lexsort per permutation), so 10^3 permutations over
10^5 observations run in seconds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def permutation_null(
    ev_windows: pd.DataFrame,
    plc_windows: pd.DataFrame,
    span: tuple[int, int] = (1, 5),
    n_perm: int = 1000,
    seed: int = 0,
    statistic: str = "unit_mean",
) -> dict:
    """Within-unit label permutation test of the early-window deviation.

    ``statistic``:
      * ``"unit_mean"`` (default): mean over units of the per-unit deviation
        (event mean − placebo mean over ``span``) — the quantity whose
        individual differences the depth axis measures; its null is centred
        at zero by construction.
      * ``"pooled"``: pooled event rate − pooled placebo rate. Its null need
        not be centred at zero when the placebo/event ratio varies across
        units, so compare ``obs`` with ``null_q975`` rather than with zero.

    Returns the observed statistic, the null mean/SD/2.5th/97.5th percentiles,
    a one-sided Monte-Carlo p-value (``(#null >= obs + 1) / (n_perm + 1)``)
    and the null draws.
    """
    if statistic not in ("unit_mean", "pooled"):
        raise ValueError("statistic must be 'unit_mean' or 'pooled'")
    rng = np.random.default_rng(seed)
    ev = ev_windows[ev_windows["rel_pos"].between(*span)][["unit", "outcome"]].assign(is_ev=1)
    pl = plc_windows[plc_windows["rel_pos"].between(*span)][["unit", "outcome"]].assign(is_ev=0)
    both = pd.concat([ev, pl], ignore_index=True)

    unit_codes, units = pd.factorize(both["unit"])
    n_units = len(units)
    order = np.argsort(unit_codes, kind="stable")
    uc = unit_codes[order]
    y = both["outcome"].to_numpy(dtype=float)[order]
    lab = both["is_ev"].to_numpy()[order]
    n = len(y)
    n_ev_u = np.bincount(uc, weights=lab, minlength=n_units)
    n_pl_u = np.bincount(uc, minlength=n_units) - n_ev_u
    y_u = np.bincount(uc, weights=y, minlength=n_units)
    valid = (n_ev_u > 0) & (n_pl_u > 0)

    def stat(lab_):
        s_ev_u = np.bincount(uc, weights=y * lab_, minlength=n_units)
        if statistic == "pooled":
            s_ev = s_ev_u.sum()
            return s_ev / n_ev_u.sum() - (y.sum() - s_ev) / n_pl_u.sum()
        dev = s_ev_u[valid] / n_ev_u[valid] - (y_u[valid] - s_ev_u[valid]) / n_pl_u[valid]
        return float(dev.mean())

    obs = stat(lab)
    null = np.empty(n_perm)
    is_binary = bool(np.all((y == 0) | (y == 1)))
    if is_binary:
        # exact shortcut: a within-unit label permutation assigns each unit's
        # ``n_ev_u`` event slots a random subset of its outcomes, so the number
        # of failures landing in event slots is Hypergeometric(n_u, y_u, n_ev_u).
        good = y_u.astype(int); bad = (n_ev_u + n_pl_u).astype(int) - good
        draws = rng.hypergeometric(good, bad, n_ev_u.astype(int), size=(n_perm, n_units)).astype(float)
        if statistic == "pooled":
            s_ev = draws.sum(axis=1)
            null[:] = s_ev / n_ev_u.sum() - (y.sum() - s_ev) / n_pl_u.sum()
        else:
            dev = draws[:, valid] / n_ev_u[valid] - (y_u[valid] - draws[:, valid]) / n_pl_u[valid]
            null[:] = dev.mean(axis=1)
    else:
        for i in range(n_perm):
            perm = np.lexsort((rng.random(n), uc))   # random permutation *within* each unit block
            null[i] = stat(lab[perm])
    pval = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return {"obs": float(obs), "statistic": statistic, "n_perm": int(n_perm), "n_units": int(valid.sum()),
            "null_mean": float(np.nanmean(null)), "null_sd": float(np.nanstd(null)),
            "null_q025": float(np.nanquantile(null, 0.025)), "null_q975": float(np.nanquantile(null, 0.975)),
            "p_value": float(pval), "null": null}
