"""Deterministic simulation for validation (seed of the ADEMP benchmark, M3).

``simulate_recovery`` plants a known post-failure recovery signal so tests and
vignettes can check the pipeline recovers it. ``run_ademp`` (M3) will sweep the
regression-to-the-mean-contamination and sparse-episode scenarios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Telemetry


def simulate_recovery(
    n_units: int = 40,
    events_per_unit: int = 12,
    window: int = 20,
    baseline_rate: float = 0.2,
    depth: float = 0.3,
    tau: float = 5.0,
    gap: int = 8,
    seed: int = 0,
) -> Telemetry:
    """Generate telemetry with a planted exponential post-failure recovery.

    Each unit repeats: one failure anchor (phase='anchor'), a post-failure window
    of ``window`` observations whose failure probability decays back to baseline,
    then a ``gap`` of baseline observations (candidate placebo region). Per-unit
    depth is jittered to create true individual differences.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        seq = 0
        u_depth = depth * rng.uniform(0.5, 1.5)
        for _ in range(events_per_unit):
            rows.append((u, seq, 1, "anchor")); seq += 1
            for rel in range(1, window + 1):
                p = np.clip(baseline_rate + u_depth * np.exp(-rel / tau), 0.0, 1.0)
                rows.append((u, seq, int(rng.random() < p), "post")); seq += 1
            for _ in range(gap):
                rows.append((u, seq, int(rng.random() < baseline_rate), "gap")); seq += 1
    df = pd.DataFrame(rows, columns=["unit", "seq", "outcome", "phase"])
    return Telemetry.from_frame(df, covariates=["phase"])


def run_ademp(
    n_units: int = 40,
    events_per_unit: int = 12,
    window: int = 20,
    depth_grid=(0.0, 0.3),
    sparse_grid=(False, True),
    n_reps: int = 20,
    baseline_rate: float = 0.2,
    tau: float = 5.0,
    span: tuple[int, int] = (1, 5),
    seed: int = 0,
) -> dict:
    """Deterministic ADEMP benchmark of two recovery-depth estimators.

    For each cell of (true depth) x (episode density), over ``n_reps`` seeded
    replications, estimates the early-window recovery depth two ways:

      * ``placebo_matched`` : event-window rate minus *matched placebo* rate
        (the pipeline's estimator).
      * ``naive_onesided``  : event-window rate minus the unit's in-sample global
        mean, one-sided (clipped at 0) -- the biased index the method warns about.

    Key result reproduced under the null (true depth = 0): the placebo-matched
    estimate is ~unbiased while the naive one-sided index is inflated above zero
    (regression-to-the-mean / circular-baseline contamination). Returns per-rep
    ``replications`` and a ``summary`` table.
    """
    from .baseline import match_placebo
    from .events import detect_anchors, extract_windows

    def _rule(g):
        return (g["phase"] == "anchor").to_numpy()

    rng = np.random.default_rng(seed)
    rows = []
    for depth in depth_grid:
        for sparse in sparse_grid:
            epu = 4 if sparse else events_per_unit
            density = "sparse" if sparse else "dense"
            for rep in range(n_reps):
                s = int(rng.integers(1_000_000_000))
                tele = simulate_recovery(
                    n_units=n_units, events_per_unit=epu, window=window,
                    baseline_rate=baseline_rate, depth=depth, tau=tau, seed=s,
                )
                anchors = detect_anchors(tele, rule=_rule)
                plc = match_placebo(tele, anchors, window=window, seed=s)
                ev_w = extract_windows(tele, anchors, window, "event")
                plc_w = extract_windows(tele, plc[["unit", "event_id", "seq"]], window, "placebo")

                ev_early = ev_w[ev_w["rel_pos"].between(*span)].groupby("unit")["outcome"].mean()
                pl_early = plc_w[plc_w["rel_pos"].between(*span)].groupby("unit")["outcome"].mean()
                gmean = tele.frame.groupby("unit")["outcome"].mean()

                dev_placebo = (ev_early - pl_early).dropna()
                dev_naive = (ev_early - gmean.reindex(ev_early.index)).clip(lower=0).dropna()
                rows.append((depth, density, "placebo_matched", rep, float(dev_placebo.mean())))
                rows.append((depth, density, "naive_onesided", rep, float(dev_naive.mean())))

    reps = pd.DataFrame(rows, columns=["true_depth", "density", "method", "rep", "estimate"])
    summary = (reps.groupby(["true_depth", "density", "method"])["estimate"]
               .agg(mean="mean", sd="std", n="size").reset_index())
    return {"replications": reps, "summary": summary}
