"""Permutation null for the depth deviation (C1 step 3 D9).

Shuffles event/placebo labels *within unit* to build a null distribution for the
early-window deviation, ruling out regression-to-the-mean artifacts.
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
) -> dict:
    rng = np.random.default_rng(seed)
    ev = ev_windows[ev_windows["rel_pos"].between(*span)][["unit", "outcome"]].assign(is_ev=1)
    pl = plc_windows[plc_windows["rel_pos"].between(*span)][["unit", "outcome"]].assign(is_ev=0)
    both = pd.concat([ev, pl], ignore_index=True)
    obs = ev["outcome"].mean() - pl["outcome"].mean()
    null = np.empty(n_perm)
    for i in range(n_perm):
        both["shuf"] = both.groupby("unit")["is_ev"].transform(
            lambda s: rng.permutation(s.values))
        m = both.groupby("shuf")["outcome"].mean()
        null[i] = m.get(1, np.nan) - m.get(0, np.nan)
    pval = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return {"obs": float(obs), "null_mean": float(np.nanmean(null)), "p_value": float(pval)}
