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
    n_boot: int = 0,
    seed: int = 0,
    method_split: str = "odd_even",
) -> dict:
    """Split-half reliability of a per-unit deviation axis.

    ``axis`` is descriptive; ``span`` selects the rel_pos range that defines it
    (e.g. (1, 5) for depth, (1, window) for the area index). Events are split
    by ``method_split``: ``"odd_even"`` (default) alternates consecutive events,
    ``"first_second"`` contrasts the first and second half of each unit's
    events (more conservative when performance drifts over time). The baseline
    windows attached to each event (placebo or pre-event windows share its
    ``event_id``) go to the *same* half as the event, so the two halves have
    independent baselines and no shared sampling error. With ``n_boot > 0`` a
    percentile bootstrap over units gives a 95% CI for the Spearman–Brown
    reliability (``ci_low``, ``ci_high``).
    """
    if method_split not in ("odd_even", "first_second"):
        raise ValueError("method_split must be 'odd_even' or 'first_second'")
    order = (ev_windows[["unit", "event_id"]].drop_duplicates()
             .assign(k=lambda d: d.groupby("unit", observed=True).cumcount()))
    n_ev = order.groupby("unit", observed=True)["k"].max() + 1
    if method_split == "odd_even":
        order["half"] = order["k"] % 2
    else:
        order["half"] = (order["k"] >= order["unit"].map(n_ev) / 2).astype(int)
    ok = n_ev[n_ev >= min_events].index
    ev = ev_windows.merge(order[["event_id", "half"]], on="event_id")
    pl = plc_windows.merge(order[["event_id", "half"]], on="event_id")
    halves = {}
    for h in (0, 1):
        e = ev[ev["unit"].isin(ok) & (ev["half"] == h) & ev["rel_pos"].between(*span)]
        b = pl[pl["unit"].isin(ok) & (pl["half"] == h) & pl["rel_pos"].between(*span)]
        halves[h] = e.groupby("unit", observed=True)["outcome"].mean() - b.groupby("unit", observed=True)["outcome"].mean()
    both = pd.concat(halves, axis=1).dropna()
    r = both[0].corr(both[1], method=method) if len(both) >= 3 else np.nan
    out = {"axis": axis, "n_units": len(both), "r_split": r, "r_sb": spearman_brown(r), "method_split": method_split}
    if n_boot and len(both) >= 3:
        rng = np.random.default_rng(seed)
        vals = both.to_numpy()
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(vals), len(vals))
            b = pd.DataFrame(vals[idx])
            rb = b[0].corr(b[1], method=method)
            boots.append(spearman_brown(rb))
        boots = np.asarray(boots, dtype=float)
        out["ci_low"], out["ci_high"] = (float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5)))
        out["n_boot"] = int(n_boot)
    return out


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


def temporal_split_half(
    tele,
    anchor_rule,
    window: int,
    span: tuple[int, int] = (1, 5),
    axis: str = "depth",
    min_events: int = 10,
    n_per_event: int = 3,
    match_on=(),
    seed: int = 0,
    n_boot: int = 0,
    method: str = "spearman",
    baseline: str = "placebo",
    exclude_before: int = 0,
    max_events: int | None = None,
    **placebo_kw,
) -> dict:
    """Split-half reliability from two *disjoint time segments* per unit.

    Each unit's stream is cut at its median ``seq``; anchors, windows and
    placebo baselines are then built independently inside each segment, so the
    two half-estimates share no observations (event windows that overlap, or
    placebo draws reused across events, cannot inflate the correlation as they
    can with an odd/even split of a single analysis). ``placebo_kw`` is passed
    to :func:`recoverr.baseline.match_placebo` (e.g. ``exclude_episode``,
    ``stream``). Returns the same keys as :func:`split_half`.
    """
    from .baseline import match_placebo
    from .events import detect_anchors, extract_windows
    from .schema import Telemetry

    f = tele.frame
    med = f.groupby("unit", observed=True)["seq"].transform("median")
    halves = {}
    for h, mask in ((0, f["seq"] < med), (1, f["seq"] >= med)):
        sub = Telemetry(f[mask].reset_index(drop=True), tele.covariates)
        anchors = detect_anchors(sub, rule=anchor_rule)
        if anchors.empty:
            halves[h] = pd.Series(dtype=float)
            continue
        n_anchor = anchors.groupby("unit", observed=True).size()
        anchors = anchors[anchors["unit"].isin(n_anchor[n_anchor >= min_events].index)]
        if max_events is not None:   # keep the first ``max_events`` anchors per unit (reliability-vs-episodes curves)
            anchors = anchors[anchors.groupby("unit", observed=True).cumcount() < max_events]
        if anchors.empty:
            halves[h] = pd.Series(dtype=float)
            continue
        stream = placebo_kw.get("stream")
        ev_w = extract_windows(sub, anchors, window, "event", stream=stream)
        if baseline == "placebo":
            plc = match_placebo(sub, anchors, window=window, n_per_event=n_per_event,
                                match_on=list(match_on), seed=seed + h, exclude_before=exclude_before, **placebo_kw)
            plc_w = extract_windows(sub, plc[["unit", "event_id", "seq"]], window, "placebo",
                                    event_anchors=anchors, stream=stream)
        else:
            plc_w = extract_windows(sub, anchors, window, "pre_event", event_anchors=anchors,
                                    stream=stream, direction="pre", gap=exclude_before)
        e = ev_w[ev_w["rel_pos"].between(*span)].groupby("unit", observed=True)["outcome"].mean()
        b = plc_w[plc_w["rel_pos"].between(*span)].groupby("unit", observed=True)["outcome"].mean()
        halves[h] = e - b
    both = pd.concat(halves, axis=1).dropna()
    r = both[0].corr(both[1], method=method) if len(both) >= 3 else np.nan
    out = {"axis": axis, "n_units": len(both), "r_split": r, "r_sb": spearman_brown(r), "method_split": "temporal"}
    if n_boot and len(both) >= 3:
        rng = np.random.default_rng(seed)
        vals = both.to_numpy(); boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(vals), len(vals)); bdf = pd.DataFrame(vals[idx])
            boots.append(spearman_brown(bdf[0].corr(bdf[1], method=method)))
        boots = np.asarray(boots, dtype=float)
        out["ci_low"], out["ci_high"] = float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))
        out["n_boot"] = int(n_boot)
    return out
