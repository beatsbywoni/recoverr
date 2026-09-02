"""Event (failure) detection and post-anchor window extraction.

Generalizes C1 step 2 (`c1_02_events_baseline.py`): detect event anchors from
the stream, then extract the window of ``window`` observations that follow each
anchor, labeled with relative position 1..window.
"""
from __future__ import annotations

from collections.abc import Callable

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
    for u, g in tele.frame.groupby("unit", sort=False, observed=True):
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


def extract_windows(
    tele: Telemetry,
    anchors: pd.DataFrame,
    window: int,
    kind: str,
    event_anchors: pd.DataFrame | None = None,
    stream: str | None = None,
    direction: str = "post",
    gap: int = 0,
) -> pd.DataFrame:
    """Extract the ``window`` observations after each anchor (rel_pos 1..window).

    Returns long frame: unit, event_id, rel_pos, outcome, kind, overlap. ``overlap``
    is True when the window contains another event anchor (a later failure falls
    inside this window). ``event_anchors`` supplies the anchor set used to flag
    overlap (defaults to ``anchors`` when kind='event').

    ``stream`` (optional) names a covariate that identifies sub-streams within a
    unit (e.g. the lexeme in spaced-repetition logs). The window then consists
    of the next ``window`` observations of the *same stream* as the anchor —
    e.g. the subsequent reviews of the item that was just forgotten — and
    overlap is flagged only for later anchors of that stream.

    With ``direction="pre"`` the ``window`` observations *before* the anchor are
    returned instead, mirrored so that rel_pos 1 is the observation closest to
    the anchor; ``gap`` observations immediately before the anchor are skipped
    (use it to drop the span that defines the anchor, e.g. the preceding failure
    of a failure-cluster rule). Pre-windows serve as a within-person pre-post
    baseline (``RecoveryPipeline(baseline="pre_event")``).
    """
    if direction not in ("post", "pre"):
        raise ValueError("direction must be 'post' or 'pre'")
    ev_src = event_anchors if event_anchors is not None else (anchors if kind == "event" else None)
    anchors_by_unit = {u: g for u, g in anchors.groupby("unit", sort=False, observed=True)}
    ev_by_unit = ({u: np.sort(g["seq"].to_numpy()) for u, g in ev_src.groupby("unit", sort=False, observed=True)}
                  if ev_src is not None else {})
    out_unit, out_eid, out_rel, out_y, out_ov = [], [], [], [], []
    for u, g in tele.frame.groupby("unit", sort=False, observed=True):
        if u not in anchors_by_unit:
            continue
        g = g.sort_values("seq").reset_index(drop=True)
        seqs = g["seq"].to_numpy()
        y = g["outcome"].to_numpy()
        n = len(g)
        ev_seqs_all = ev_by_unit.get(u, np.array([]))
        # position (row index) of every anchor of this unit
        a_rows = anchors_by_unit[u]
        a_pos = np.searchsorted(seqs, a_rows["seq"].to_numpy(), side="right") - 1
        ok = (a_pos >= 0) & (seqs[np.clip(a_pos, 0, n - 1)] == a_rows["seq"].to_numpy())
        if stream is not None:
            st = g[stream].to_numpy()
            codes, uniq = pd.factorize(st)
            # positions of each stream, sorted, as a ragged structure
            order = np.argsort(codes, kind="stable")
            bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
            rank_in_stream = np.empty(n, dtype=int)           # index of each row within its stream
            rank_in_stream[order] = np.arange(n) - np.repeat(bounds[:-1], np.diff(bounds))
            ev_pos = np.searchsorted(seqs, ev_seqs_all) if ev_seqs_all.size else np.array([], dtype=int)
            ev_pos = ev_pos[(ev_pos < n) & (seqs[np.clip(ev_pos, 0, n - 1)] == ev_seqs_all)] if ev_seqs_all.size else ev_pos
            ev_code = codes[ev_pos] if ev_pos.size else np.array([], dtype=int)
            ev_rank = rank_in_stream[ev_pos] if ev_pos.size else np.array([], dtype=int)
            ev_sorted = np.lexsort((ev_rank, ev_code))
            ev_code, ev_rank = ev_code[ev_sorted], ev_rank[ev_sorted]
        for eid, ap, good in zip(a_rows["event_id"].to_numpy(), a_pos, ok):
            if not good:
                continue
            if stream is None:
                if direction == "post":
                    sel = np.arange(ap + 1, min(ap + 1 + window, n))
                else:
                    hi = ap - gap
                    sel = np.arange(hi - 1, max(hi - 1 - window, -1), -1)
                if len(sel) == 0:
                    continue
                last = seqs[sel[-1]]
                if direction == "post":
                    overlap = bool(ev_seqs_all.size and np.any((ev_seqs_all > seqs[ap]) & (ev_seqs_all <= last)))
                else:
                    overlap = bool(ev_seqs_all.size and np.any((ev_seqs_all < seqs[ap]) & (ev_seqs_all >= last)))
            else:
                c = codes[ap]; r = rank_in_stream[ap]
                members = order[bounds[c]:bounds[c + 1]]          # rows of this stream, in seq order
                if direction == "post":
                    sel = members[r + 1: r + 1 + window]
                    lo_r, hi_r = r + 1, r + len(sel)
                else:
                    sel = members[max(r - gap - window, 0): r - gap][::-1]
                    lo_r, hi_r = r - gap - len(sel), r - gap - 1
                if len(sel) == 0:
                    continue
                if ev_code.size:
                    i0, i1 = np.searchsorted(ev_code, c, side="left"), np.searchsorted(ev_code, c, side="right")
                    rr = ev_rank[i0:i1]
                    overlap = bool(np.any((rr >= lo_r) & (rr <= hi_r)))
                else:
                    overlap = False
            k = len(sel)
            out_unit.append(np.full(k, u, dtype=object)); out_eid.append(np.full(k, eid, dtype=object))
            out_rel.append(np.arange(1, k + 1)); out_y.append(y[sel]); out_ov.append(np.full(k, overlap))
    if not out_rel:
        return pd.DataFrame(columns=["unit", "event_id", "rel_pos", "outcome", "kind", "overlap"])
    return pd.DataFrame({
        "unit": np.concatenate(out_unit), "event_id": np.concatenate(out_eid),
        "rel_pos": np.concatenate(out_rel), "outcome": np.concatenate(out_y),
        "kind": kind, "overlap": np.concatenate(out_ov),
    })


def thin_anchors(anchors: pd.DataFrame, window: int) -> pd.DataFrame:
    """Greedy thinning so that kept anchors are more than ``window`` apart.

    Within each unit the first anchor is kept and every later anchor that falls
    inside the kept anchor's post-window (``seq - kept <= window``) is dropped.
    Kept windows therefore never overlap (no observation is counted twice), while
    the windows themselves still contain whatever later failures occur — unlike
    ``exclude_overlapping``, thinning does not condition on the outcome. Used
    for split-half reliability, where overlapping windows inflate the
    correlation between halves.
    """
    keep = np.zeros(len(anchors), dtype=bool)
    a = anchors.sort_values(["unit", "seq"], kind="stable")
    pos = {idx: i for i, idx in enumerate(a.index)}
    for _, g in a.groupby("unit", sort=False, observed=True):
        last = -np.inf
        for idx, s in zip(g.index, g["seq"].to_numpy()):
            if s - last > window:
                keep[pos[idx]] = True
                last = s
    return a[keep].sort_index()
