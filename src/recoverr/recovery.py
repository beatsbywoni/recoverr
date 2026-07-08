"""Three-axis recovery estimation.

Generalizes C1 step 3 D6:
  depth        = mean event-minus-placebo deviation over rel_pos in ``depth_span``
  speed        = exponential-decay time-constant tau (parametric) and/or the
                 mean deviation over the window (nonparametric AUC)
  completeness = mean deviation over rel_pos in ``complete_span`` (late window)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def exp_decay(t, d0, tau):
    """Exponential recovery: deviation(t) = d0 * exp(-t / tau)."""
    return d0 * np.exp(-t / tau)


def recovery_curve(ev_windows: pd.DataFrame, plc_windows: pd.DataFrame) -> pd.DataFrame:
    """Population deviation curve: event-rate minus placebo-rate at each rel_pos."""
    e = ev_windows.groupby("rel_pos")["outcome"].mean().rename("event")
    p = plc_windows.groupby("rel_pos")["outcome"].mean().rename("placebo")
    c = pd.concat([e, p], axis=1)
    c["deviation"] = c["event"] - c["placebo"]
    return c.reset_index()


def recovery_axes(
    ev_windows: pd.DataFrame,
    plc_windows: pd.DataFrame,
    window: int,
    depth_span: tuple[int, int] = (1, 5),
    complete_span: tuple[int, int] | None = None,
    fit_tau: bool = True,
) -> pd.DataFrame:
    """Per-unit recovery axes: depth, auc, tau, completeness, level."""
    if complete_span is None:
        complete_span = (max(1, window - 4), window)
    e = ev_windows.groupby(["unit", "rel_pos"])["outcome"].mean().rename("ev")
    p = plc_windows.groupby(["unit", "rel_pos"])["outcome"].mean().rename("pl")
    d = pd.concat([e, p], axis=1).dropna()
    d["dev"] = d["ev"] - d["pl"]
    rows = []
    for u, g in d.groupby(level="unit"):
        g = g.droplevel("unit")
        depth = g.loc[depth_span[0]:depth_span[1], "dev"].mean()
        auc = g.loc[1:window, "dev"].mean()
        comp = g.loc[complete_span[0]:complete_span[1], "dev"].mean()
        tau = np.nan
        if fit_tau and np.isfinite(depth) and depth > 0:
            gg = g["dev"].clip(lower=0)
            if gg.notna().sum() >= 6:
                try:
                    (_d0, tau), _ = curve_fit(
                        exp_decay, g.index.values.astype(float), gg.values,
                        p0=[max(depth, 0.01), 5.0], bounds=([0, 0.5], [1, 100]),
                        maxfev=2000,
                    )
                except Exception:
                    tau = np.nan
        rows.append((u, depth, auc, tau, comp, g["ev"].mean()))
    return pd.DataFrame(rows, columns=["unit", "depth", "auc", "tau", "completeness", "level"])
