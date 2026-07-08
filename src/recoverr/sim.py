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


def run_ademp(*args, **kwargs):  # pragma: no cover
    """ADEMP benchmark over contamination/sparsity scenarios (M3)."""
    raise NotImplementedError("run_ademp: implement ADEMP sweep in M3 (see #15 design).")
