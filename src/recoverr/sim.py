"""Deterministic simulation and the ADEMP benchmark.

``simulate_recovery`` plants a known post-failure recovery signal (with a
choice of recovery *shape*) so estimators can be checked against ground truth.
``run_ademp`` sweeps a factorial design and compares four estimators of
early-window recovery depth, reporting bias, RMSE, and SD against the analytic
truth, plus recovery of the decay time-constant under the exponential shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Telemetry

SHAPES = ("exponential", "linear", "plateau")


def _shape(t: np.ndarray, tau: float, shape: str) -> np.ndarray:
    """Normalized recovery profile g(t) in [0,1]; g -> 0 as t grows."""
    t = np.asarray(t, dtype=float)
    if shape == "exponential":
        return np.exp(-t / tau)
    if shape == "linear":
        return np.clip(1.0 - t / (2.0 * tau), 0.0, 1.0)
    if shape == "plateau":  # delayed / non-monotonic-like: flat for tau steps, then decays
        return np.where(t <= tau, 1.0, np.exp(-(t - tau) / tau))
    raise ValueError(f"unknown shape {shape!r}; choose from {SHAPES}")


def simulate_recovery(
    n_units: int = 40,
    events_per_unit: int = 12,
    window: int = 20,
    baseline_rate: float = 0.2,
    depth: float = 0.3,
    tau: float = 5.0,
    gap: int | None = None,
    heterogeneity: float = 0.5,
    shape: str = "exponential",
    seed: int = 0,
    with_truth: bool = False,
):
    """Generate telemetry with a planted post-failure recovery.

    Each unit repeats: a failure anchor (phase='anchor'), a post-failure window of
    ``window`` observations whose failure probability is elevated by
    ``u_depth * g(t)`` and returns to ``baseline_rate``, then ``gap`` baseline
    observations (candidate placebo region; default ``2 * window + 5`` so that
    clean, full-length placebo windows exist between consecutive events under
    ``match_placebo(..., exclusion="both")``). Per-unit depth is
    ``depth * U(1-heterogeneity, 1+heterogeneity)`` to create true individual
    differences. With ``with_truth=True`` also returns the analytic expected
    early-window (positions 1..5) deviation and the per-unit true depths.
    """
    rng = np.random.default_rng(seed)
    if gap is None:
        gap = 2 * window + 5
    rows, u_depths = [], []
    span_t = np.arange(1, 6)
    for u in range(n_units):
        seq = 0
        u_depth = depth * rng.uniform(1 - heterogeneity, 1 + heterogeneity) if depth > 0 else 0.0
        u_depths.append(u_depth)
        for _ in range(events_per_unit):
            rows.append((u, seq, 1, "anchor")); seq += 1
            for rel in range(1, window + 1):
                p = np.clip(baseline_rate + u_depth * _shape(rel, tau, shape), 0.0, 1.0)
                rows.append((u, seq, int(rng.random() < p), "post")); seq += 1
            for _ in range(gap):
                rows.append((u, seq, int(rng.random() < baseline_rate), "gap")); seq += 1
    df = pd.DataFrame(rows, columns=["unit", "seq", "outcome", "phase"])
    tele = Telemetry.from_frame(df, covariates=["phase"])
    if not with_truth:
        return tele
    g_early = float(np.mean(_shape(span_t, tau, shape)))
    truth = {"early_dev": float(np.mean(u_depths)) * g_early,
             "unit_depth": np.asarray(u_depths), "tau": tau, "shape": shape}
    return tele, truth


def _anchor_rule(g):
    return (g["phase"] == "anchor").to_numpy()


def _pre_event_baseline(tele: Telemetry, anchors: pd.DataFrame, span: int) -> pd.Series:
    """Per-unit mean outcome over the ``span`` observations preceding each anchor."""
    out = {}
    f = tele.frame
    for u, g in f.groupby("unit", sort=False, observed=True):
        g = g.sort_values("seq").reset_index(drop=True)
        seqs = g["seq"].to_numpy(); y = g["outcome"].to_numpy()
        a = anchors[anchors["unit"] == u]["seq"].to_numpy()
        vals = []
        for s in a:
            idx = int(np.searchsorted(seqs, s, side="left"))
            lo = max(0, idx - span)
            if idx - lo > 0:
                vals.append(y[lo:idx].mean())
        if vals:
            out[u] = float(np.mean(vals))
    return pd.Series(out, name="pre")


def run_ademp(
    n_units: int = 40,
    events_grid=(4, 12),
    depth_grid=(0.0, 0.15, 0.3),
    shape_grid=("exponential", "linear", "plateau"),
    heterogeneity: float = 0.5,
    window: int = 20,
    tau: float = 5.0,
    baseline_rate: float = 0.2,
    span: tuple[int, int] = (1, 5),
    n_reps: int = 20,
    seed: int = 0,
) -> dict:
    """Factorial ADEMP benchmark of early-window recovery-depth estimators.

    Estimators compared (per unit, then averaged):
      * ``placebo_matched`` : event-window rate − matched placebo-window rate (recoverr)
      * ``naive_onesided``  : max(0, event-window rate − in-sample global mean)
      * ``naive_twosided``  : event-window rate − in-sample global mean
      * ``pre_event``       : event-window rate − mean of the ``span`` observations before each event
    Truth is the analytic expected early-window deviation from the data-generating
    process. Also fits an exponential decay to the population placebo-matched
    curve to recover τ (bias reported for the exponential shape) and records the
    per-unit unidentified-τ rate. Returns ``replications`` and ``summary``.
    """
    from .baseline import match_placebo
    from .events import detect_anchors, extract_windows
    from .recovery import fit_parametric, recovery_axes, recovery_curve

    rng = np.random.default_rng(seed)
    rows = []
    for shape in shape_grid:
        for depth in depth_grid:
            for epu in events_grid:
                density = "sparse" if epu <= 4 else "dense"
                for rep in range(n_reps):
                    s = int(rng.integers(1_000_000_000))
                    tele, truth = simulate_recovery(
                        n_units=n_units, events_per_unit=epu, window=window,
                        baseline_rate=baseline_rate, depth=depth, tau=tau,
                        heterogeneity=heterogeneity, shape=shape, seed=s, with_truth=True,
                    )
                    anchors = detect_anchors(tele, rule=_anchor_rule)
                    # anchors are *scheduled* (exogenous) in the DGP, so excluding
                    # placebo windows that span a later anchor does not condition on
                    # the outcome; 'both' gives the clean counterfactual baseline.
                    plc = match_placebo(tele, anchors, window=window, seed=s, exclusion="both")
                    ev_w = extract_windows(tele, anchors, window, "event")
                    plc_w = extract_windows(tele, plc[["unit", "event_id", "seq"]], window, "placebo")
                    early = ev_w[ev_w["rel_pos"].between(*span)].groupby("unit", observed=True)["outcome"].mean()
                    pl_early = plc_w[plc_w["rel_pos"].between(*span)].groupby("unit", observed=True)["outcome"].mean()
                    gmean = tele.frame.groupby("unit", observed=True)["outcome"].mean().reindex(early.index)
                    pre = _pre_event_baseline(tele, anchors, span[1] - span[0] + 1).reindex(early.index)
                    est = {
                        "placebo_matched": float((early - pl_early).dropna().mean()),
                        "naive_onesided": float((early - gmean).clip(lower=0).dropna().mean()),
                        "naive_twosided": float((early - gmean).dropna().mean()),
                        "pre_event": float((early - pre).dropna().mean()),
                    }
                    # tau recovery + unidentified rate (population curve fit; per-unit axes)
                    curve = recovery_curve(ev_w, plc_w)
                    fit = fit_parametric(curve)
                    axes = recovery_axes(ev_w, plc_w, window=window, depth_span=span)
                    unident = float(axes["tau"].isna().mean()) if len(axes) else np.nan
                    for name, val in est.items():
                        rows.append(dict(shape=shape, true_depth=depth, density=density, rep=rep,
                                         estimator=name, estimate=val, truth=truth["early_dev"],
                                         tau_hat=fit["tau"], tau_true=tau, tau_identified=fit["identified"],
                                         unidentified_rate=unident))
    reps = pd.DataFrame(rows)
    reps["error"] = reps["estimate"] - reps["truth"]
    g = reps.groupby(["shape", "true_depth", "density", "estimator"])
    summary = g.agg(mean=("estimate", "mean"), truth=("truth", "mean"),
                    bias=("error", "mean"), rmse=("error", lambda e: float(np.sqrt(np.mean(np.square(e))))),
                    sd=("estimate", "std"), n=("estimate", "size")).reset_index()
    # tau summary over *identified* population fits (median is robust to bound-hits)
    pm = reps[reps["estimator"] == "placebo_matched"].copy()
    pm["tau_id"] = pm["tau_hat"].where(pm["tau_identified"])
    tau_rows = pm.groupby(["shape", "true_depth", "density"]).agg(
        tau_hat_median=("tau_id", "median"), tau_true=("tau_true", "first"),
        tau_identified_rate=("tau_identified", "mean"), unidentified_unit_rate=("unidentified_rate", "mean"),
    ).reset_index()
    tau_rows["tau_bias"] = tau_rows["tau_hat_median"] - tau_rows["tau_true"]
    return {"replications": reps, "summary": summary, "tau": tau_rows}


def simulate_endogenous_null(
    n_units: int = 40,
    n_obs: int = 600,
    baseline_rate: float = 0.2,
    rho: float = 0.8,
    sigma: float = 1.0,
    seed: int = 0,
) -> Telemetry:
    """Pure regression-to-the-mean null: no post-failure effect at all.

    Each unit's failure probability follows a latent AR(1) process on the logit
    scale, ``s_t = rho * s_{t-1} + sigma * sqrt(1 - rho^2) * e_t``, so risk
    drifts slowly; outcomes are Bernoulli(p_t) with ``p_t = expit(logit(base) + s_t)``.
    Failures are *not* followed by any planted deviation, so the true
    post-failure deviation relative to the unit's counterfactual risk is zero.
    Anchors are defined *from the outcome* (a failure preceded by a failure,
    see :func:`endogenous_anchor_rule`), which is exactly the selection that
    makes in-sample and pre-event baselines regress to the mean.
    """
    rng = np.random.default_rng(seed)
    base = np.log(baseline_rate / (1 - baseline_rate))
    rows = []
    for u in range(n_units):
        s = 0.0
        for t in range(n_obs):
            s = rho * s + sigma * np.sqrt(1 - rho ** 2) * rng.standard_normal()
            p = 1.0 / (1.0 + np.exp(-(base + s)))
            rows.append((u, t, int(rng.random() < p)))
    df = pd.DataFrame(rows, columns=["unit", "seq", "outcome"])
    return Telemetry.from_frame(df)


def endogenous_anchor_rule(g):
    """Outcome-defined anchor: a failure immediately preceded by a failure."""
    y = g["outcome"].to_numpy()
    prev = np.concatenate([[0], y[:-1]])
    return (y == 1) & (prev == 1)


def recent_rate_bins(tele: Telemetry, m: int = 10, lag: int = 2, bins: int = 4, name: str = "recent_bin") -> Telemetry:
    """Add a matching covariate: quantile bin of the unit's failure rate over the
    ``m`` observations ending ``lag`` steps before each observation (the lag
    skips the anchor-defining span). Observations without ``m`` prior points get
    bin ``-1``. Useful as ``match_on`` when risk drifts slowly within a unit."""
    f = tele.frame.copy()
    f["_recent"] = f.groupby("unit", observed=True)["outcome"].transform(lambda y: y.shift(lag).rolling(m, min_periods=m).mean())
    f[name] = pd.qcut(f["_recent"].rank(method="first"), bins, labels=False)
    f[name] = f[name].where(f["_recent"].notna(), -1).astype(int)
    return Telemetry.from_frame(f.drop(columns="_recent"), covariates=[*tele.covariates, name])


def run_ademp_null(
    n_units: int = 40,
    n_obs: int = 600,
    rho_grid=(0.0, 0.5, 0.9),
    window: int = 20,
    span: tuple[int, int] = (1, 5),
    n_reps: int = 20,
    seed: int = 0,
) -> dict:
    """ADEMP null with *outcome-defined* anchors and autocorrelated risk.

    No post-failure effect is planted, so any non-zero estimate reflects
    selection on the outcome (regression to the mean) and/or persistence of
    the latent risk state that produced the failure cluster. Estimators:
    ``placebo_matched`` (symmetric ``exclusion="post"``), ``placebo_matched_recent``
    (additionally matched on :func:`recent_rate_bins`), ``naive_onesided``,
    ``naive_twosided``, ``pre_event`` (the ``span`` observations right before
    the anchor, which include the anchor-defining failure) and ``pre_event_gap``
    (same, skipping the defining span). Returns ``replications`` and ``summary``.
    """
    from .baseline import match_placebo
    from .events import detect_anchors, extract_windows

    rng = np.random.default_rng(seed)
    rows = []
    k = span[1] - span[0] + 1
    for rho in rho_grid:
        for rep in range(n_reps):
            s = int(rng.integers(1_000_000_000))
            tele = recent_rate_bins(simulate_endogenous_null(n_units=n_units, n_obs=n_obs, rho=rho, seed=s))
            anchors = detect_anchors(tele, rule=endogenous_anchor_rule)
            anchors = anchors[anchors["recent_bin"] >= 0]
            if anchors.empty:
                continue
            ev_w = extract_windows(tele, anchors, window, "event")
            early = ev_w[ev_w["rel_pos"].between(*span)].groupby("unit", observed=True)["outcome"].mean()
            est = {}
            for name, m_on in (("placebo_matched", []), ("placebo_matched_recent", ["recent_bin"])):
                plc = match_placebo(tele, anchors, window=window, seed=s, exclusion="post", match_on=m_on)
                plc_w = extract_windows(tele, plc[["unit", "event_id", "seq"]], window, "placebo", event_anchors=anchors)
                pl_early = plc_w[plc_w["rel_pos"].between(*span)].groupby("unit", observed=True)["outcome"].mean()
                est[name] = float((early - pl_early).dropna().mean())
            gmean = tele.frame.groupby("unit", observed=True)["outcome"].mean().reindex(early.index)
            pre = _pre_event_baseline(tele, anchors, k).reindex(early.index)
            shifted = anchors.copy(); shifted["seq"] = shifted["seq"] - 1   # skip the defining failure
            pre_gap = _pre_event_baseline(tele, shifted, k).reindex(early.index)
            est.update({
                "naive_onesided": float((early - gmean).clip(lower=0).dropna().mean()),
                "naive_twosided": float((early - gmean).dropna().mean()),
                "pre_event": float((early - pre).dropna().mean()),
                "pre_event_gap": float((early - pre_gap).dropna().mean()),
            })
            for name, val in est.items():
                rows.append(dict(rho=rho, rep=rep, estimator=name, estimate=val, truth=0.0,
                                 n_events=len(anchors)))
    reps = pd.DataFrame(rows)
    reps["error"] = reps["estimate"] - reps["truth"]
    g = reps.groupby(["rho", "estimator"])
    summary = g.agg(mean=("estimate", "mean"), bias=("error", "mean"),
                    rmse=("error", lambda e: float(np.sqrt(np.mean(np.square(e))))),
                    sd=("estimate", "std"), n=("estimate", "size"),
                    events_per_unit=("n_events", lambda x: float(np.mean(x) / n_units))).reset_index()
    return {"replications": reps, "summary": summary}
