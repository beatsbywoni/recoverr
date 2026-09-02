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


def fit_parametric(curve: pd.DataFrame, tau_bounds: tuple[float, float] = (0.5, 100.0)) -> dict:
    """Fit ``D(t) = D0 * exp(-t / tau)`` to a population deviation curve.

    Fits on the non-negative part of ``deviation`` by nonlinear least squares
    (SciPy ``curve_fit``), with D0 in [0, 1] and tau in ``tau_bounds``.
    Returns D0, tau, R² of the fit, the fitted values, and an ``identified``
    flag that is False when the fit fails, the curve is non-positive, or tau
    lands on a bound (the criterion under which the nonparametric area index is
    preferred as the speed axis).
    """
    t_all = curve["rel_pos"].to_numpy(dtype=float)
    y_all = curve["deviation"].to_numpy(dtype=float)
    out = {"d0": np.nan, "tau": np.nan, "r2": np.nan, "fitted": np.full_like(y_all, np.nan), "identified": False}
    ok = np.isfinite(y_all)
    t, yc = t_all[ok], np.clip(y_all[ok], 0, None)
    if len(t) < 4 or yc.max() <= 0:
        return out
    try:
        (d0, tau), _ = curve_fit(exp_decay, t, yc, p0=[max(yc.max(), 0.01), 5.0],
                                 bounds=([0, tau_bounds[0]], [1, tau_bounds[1]]), maxfev=5000)
    except Exception:
        return out
    fitted_ok = exp_decay(t, d0, tau)
    ss_res = float(np.sum((yc - fitted_ok) ** 2)); ss_tot = float(np.sum((yc - yc.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    at_bound = np.isclose(tau, tau_bounds[0]) or np.isclose(tau, tau_bounds[1])
    fitted = np.full_like(y_all, np.nan); fitted[ok] = fitted_ok
    out.update(d0=float(d0), tau=float(tau), r2=float(r2), fitted=fitted, identified=bool(not at_bound and d0 > 0))
    return out


def recovery_curve(ev_windows: pd.DataFrame, plc_windows: pd.DataFrame, weights: str = "unit") -> pd.DataFrame:
    """Population deviation curve by relative position.

    ``weights="unit"`` (default): at each rel_pos the event rate, baseline rate
    and their difference are first computed *within* each unit and then averaged
    across units with equal weight, so units that contribute many event windows
    but few baseline windows (or vice versa) cannot create a spurious pooled
    difference. ``weights="pooled"``: plain means over all windows (the v1.0.0
    behaviour; reported for comparison). Columns: rel_pos, event, placebo,
    deviation, n_units.
    """
    if weights not in ("unit", "pooled"):
        raise ValueError("weights must be 'unit' or 'pooled'")
    if weights == "pooled":
        e = ev_windows.groupby("rel_pos")["outcome"].mean().rename("event")
        p = plc_windows.groupby("rel_pos")["outcome"].mean().rename("placebo")
        c = pd.concat([e, p], axis=1)
        c["deviation"] = c["event"] - c["placebo"]
        c["n_units"] = ev_windows.groupby("rel_pos")["unit"].nunique()
        return c.reset_index()
    e = ev_windows.groupby(["unit", "rel_pos"], observed=True)["outcome"].mean().rename("event")
    p = plc_windows.groupby(["unit", "rel_pos"], observed=True)["outcome"].mean().rename("placebo")
    d = pd.concat([e, p], axis=1).dropna()
    d["deviation"] = d["event"] - d["placebo"]
    c = d.groupby(level="rel_pos").agg(event=("event", "mean"), placebo=("placebo", "mean"),
                                        deviation=("deviation", "mean"), n_units=("deviation", "size"))
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
    e = ev_windows.groupby(["unit", "rel_pos"], observed=True)["outcome"].mean().rename("ev")
    p = plc_windows.groupby(["unit", "rel_pos"], observed=True)["outcome"].mean().rename("pl")
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
