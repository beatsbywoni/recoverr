"""Context-matched placebo baselines.

Generalizes C1 step 2 D5: for each event anchor, sample non-event anchors from
the same unit (optionally matched on covariate bins) that are not within
``exclude_within`` steps of any event anchor, to serve as a within-person
placebo baseline immune to regression-to-the-mean contamination.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .schema import Telemetry


def match_placebo(
    tele: Telemetry,
    event_anchors: pd.DataFrame,
    window: int,
    n_per_event: int = 3,
    match_on: Sequence[str] = (),
    seed: int = 0,
) -> pd.DataFrame:
    """Return placebo anchors as (unit, event_id, seq[, match_on...])."""
    rng = np.random.default_rng(seed)
    match_on = list(match_on)
    ev_seq_by_unit = event_anchors.groupby("unit")["seq"].apply(lambda s: np.asarray(s)).to_dict()
    plc = []
    ev_by_unit = {u: g for u, g in event_anchors.groupby("unit", sort=False)}
    for u, g in tele.frame.groupby("unit", sort=False):
        if u not in ev_by_unit:
            continue
        g = g.sort_values("seq")
        ev_seqs = ev_seq_by_unit.get(u, np.array([]))
        is_anchor = g["seq"].isin(set(ev_seqs.tolist()))
        # candidate = not an event anchor and not within `window` of any event anchor
        near = g["seq"].apply(lambda s: bool(np.any(np.abs(ev_seqs - s) <= window))
                              if ev_seqs.size else False)
        cand = g[~is_anchor & ~near]
        if cand.empty:
            continue
        for _, r in ev_by_unit[u].iterrows():
            pool = cand
            for m in match_on:
                pool = pool[pool[m] == r[m]]
            if pool.empty:
                continue
            take = pool.sample(min(n_per_event, len(pool)),
                               random_state=int(rng.integers(1_000_000_000)))
            for _, p in take.iterrows():
                rec = {"unit": u, "event_id": r["event_id"], "seq": p["seq"]}
                rec.update({m: p[m] for m in match_on})
                plc.append(rec)
    cols = ["unit", "event_id", "seq", *match_on]
    return pd.DataFrame(plc, columns=cols)
