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


def fit_mlm(ev_windows: pd.DataFrame, plc_windows: pd.DataFrame, seed: int = 0):
    """Multilevel random-effects reliability (NumPyro). Ported in M2.

    Raises a helpful error until the [bayes] extra logic is wired in.
    """
    try:
        import numpyro  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "fit_mlm needs the Bayesian backend: pip install 'recoverr[bayes]'"
        ) from e
    raise NotImplementedError(
        "fit_mlm: port c1_03b_fallback_mlm binomial MLM here (M2)."
    )
