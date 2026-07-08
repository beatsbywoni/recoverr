"""Event (failure) detection and post-anchor window extraction.

Generalizes C1 step 2 (`c1_02_events_baseline.py`): detect event anchors from
the stream, then extract the window of ``window`` observations that follow each
anchor, labeled with relative position 1..window.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .schema import Telemetry

# A rule maps a per-unit frame (sorted by seq) to a boolean anchor mask.
Rule = Callable[[pd.DataFrame], "np.ndarray"]


def outcome_threshold(min_value: float = 1.0) -> Rule:
    """Built-in rule: flag observations whose ``outcome >= min_value`` as anchors."""

    def rule(g: pd.DataFrame) -> np.ndarray:
        return g["outcome"].to_numpy() >= min_value

    return rule


def detect_anchors(tele: Telemetry, rule: Rule | None = None, min_value: float = 1.0) -> pd.DataFrame:
    """Return event anchors as a frame of (unit, seq, event_id[, covariates])."""
    rule = rule or outcome_threshold(min_value)
    out = []
    cov = list(tele.covariates)
    for u, g in tele.frame.groupby("unit", sort=False):
        g = g.sort_values("seq")
        mask = np.asarray(rule(g), dtype=bool)
        if not mask.any():
            continue
        a = g.loc[mask, ["unit", "seq", *cov]].copy()
        a["event_id"] = [f"{u}:{int(s)}" if float(s).is_integer() else f"{u}:{s}"
                         for s in a["seq"]]
        out.append(a)
    if not out:
        return pd.DataFrame(columns=["unit", "seq", "event_id", *cov])
    return pd.concat(out, ignore_index=True)


def extract_windows(tele: Telemetry, anchors: pd.DataFrame, window: int, kind: str) -> pd.DataFrame:
    """Extract the ``window`` observations after each anchor (rel_pos 1..window).

    Returns long frame: unit, event_id, rel_pos, outcome, kind.
    """
    rows = []
    anchors_by_unit = {u: g for u, g in anchors.groupby("unit", sort=False)}
    for u, g in tele.frame.groupby("unit", sort=False):
        if u not in anchors_by_unit:
            continue
        g = g.sort_values("seq").reset_index(drop=True)
        seqs = g["seq"].to_numpy()
        for _, r in anchors_by_unit[u].iterrows():
            idx = int(np.searchsorted(seqs, r["seq"], side="right"))
            w = g.iloc[idx: idx + window]
            if len(w) == 0:
                continue
            rows.append(pd.DataFrame({
                "unit": u,
                "event_id": r["event_id"],
                "rel_pos": np.arange(1, len(w) + 1),
                "outcome": w["outcome"].to_numpy(),
                "kind": kind,
            }))
    if not rows:
        return pd.DataFrame(columns=["unit", "event_id", "rel_pos", "outcome", "kind"])
    return pd.concat(rows, ignore_index=True)
