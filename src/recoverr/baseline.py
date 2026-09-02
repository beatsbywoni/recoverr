"""Context-matched placebo baselines.

Generalizes C1 step 2 D5: for each event anchor, sample non-event anchors from
the same unit (optionally matched on covariate bins) that are not within
``exclude_within`` steps of any event anchor, to serve as a within-person
placebo baseline immune to regression-to-the-mean contamination.
"""
from __future__ import annotations

from collections.abc import Sequence

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
    exclusion: str = "post",
    exclude_before: int = 0,
    exclude_episode: str | None = None,
    stream: str | None = None,
) -> pd.DataFrame:
    """Return placebo anchors as (unit, event_id, seq[, match_on...]).

    A placebo anchor is a non-failure observation of the same unit (optionally
    matched on ``match_on`` covariate bins). The exclusion zone around event
    anchors is built from three pieces:

    * always: the event anchors themselves and every observation lying *inside*
      the post-failure window of an earlier event (``0 < s - t_event <= window``).
      Placebo windows may still contain a *later* failure, exactly as event
      windows may, so the two window types are symmetric (``exclusion="post"``,
      default). Overlap is flagged for both and can be handled with
      ``RecoveryPipeline(exclude_overlapping=True)``.
    * ``exclude_before``: also exclude the ``exclude_before`` observations
      preceding each anchor. Use this to cover the span that *defines* the
      anchor (e.g. the tokens of an exercise whose last token is the anchor), so
      that placebo windows cannot start inside an outcome-defined episode.
    * ``exclude_episode``: name of a column identifying the episode each
      observation belongs to (e.g. ``"ex_id"``); every observation sharing an
      episode with an anchor is excluded. This is the exact form of the
      previous rule when episodes are explicit.
    * ``stream``: name of a covariate identifying sub-streams within a unit
      (see :func:`recoverr.events.extract_windows`). Distances are then
      measured in positions *within the stream*, i.e. the post-failure window
      of an anchor is the next ``window`` observations of the same stream.
    * ``exclusion="both"``: legacy rule that also excludes ``window``
      observations before every anchor, so placebo windows never contain a
      failure. This conditions the baseline on future success and is kept only
      for comparison and for simulations with *scheduled* (exogenous) anchors.
    """
    if exclusion not in ("post", "both"):
        raise ValueError("exclusion must be 'post' or 'both'")
    if exclusion == "both":
        exclude_before = max(int(exclude_before), int(window))
    rng = np.random.default_rng(seed)
    match_on = list(match_on)
    ev_seq_by_unit = event_anchors.groupby("unit", observed=True)["seq"].apply(lambda s: np.asarray(s)).to_dict()
    ev_by_unit = {u: g for u, g in event_anchors.groupby("unit", sort=False, observed=True)}
    plc = []
    for u, g in tele.frame.groupby("unit", sort=False, observed=True):
        if u not in ev_by_unit:
            continue
        g = g.sort_values("seq")
        ev_seqs = np.sort(ev_seq_by_unit.get(u, np.array([])))
        seqs = g["seq"].to_numpy()
        is_anchor = np.isin(seqs, ev_seqs)
        near = np.zeros(len(g), dtype=bool)
        if ev_seqs.size and stream is None:
            # inside the post window of the nearest *preceding* anchor
            prev_idx = np.searchsorted(ev_seqs, seqs, side="right") - 1
            d_prev = np.where(prev_idx >= 0, seqs - ev_seqs[np.clip(prev_idx, 0, None)], np.inf)
            near |= (d_prev > 0) & (d_prev <= window)
            if exclude_before > 0:
                nxt_idx = np.searchsorted(ev_seqs, seqs, side="left")
                d_next = np.where(nxt_idx < ev_seqs.size,
                                  ev_seqs[np.clip(nxt_idx, None, ev_seqs.size - 1)] - seqs, np.inf)
                near |= (d_next > 0) & (d_next <= exclude_before)
        elif ev_seqs.size:
            # same rules, but distances counted in positions within each stream
            st = g[stream].to_numpy()
            pos = pd.Series(np.arange(len(g))).groupby(st).cumcount().to_numpy()
            for s_val in np.unique(st[is_anchor]):
                m = st == s_val
                a_pos = pos[m & is_anchor]
                p_all = pos[m]
                prev_idx = np.searchsorted(a_pos, p_all, side="right") - 1
                d_prev = np.where(prev_idx >= 0, p_all - a_pos[np.clip(prev_idx, 0, None)], np.inf)
                sub = (d_prev > 0) & (d_prev <= window)
                if exclude_before > 0:
                    nxt_idx = np.searchsorted(a_pos, p_all, side="left")
                    d_next = np.where(nxt_idx < a_pos.size, a_pos[np.clip(nxt_idx, None, a_pos.size - 1)] - p_all, np.inf)
                    sub |= (d_next > 0) & (d_next <= exclude_before)
                near[np.flatnonzero(m)] |= sub
        if ev_seqs.size and exclude_episode is not None:
            ep = g[exclude_episode].to_numpy()
            near |= np.isin(ep, ep[is_anchor])
        cand = g[~is_anchor & ~near]
        if cand.empty:
            continue
        # candidate pools keyed by the matching covariates (vectorized sampling)
        cand_seq = cand["seq"].to_numpy()
        if match_on:
            pools = {k if isinstance(k, tuple) else (k,): np.asarray(v)
                     for k, v in cand.groupby(match_on, sort=False, observed=True).indices.items()}
        else:
            pools = {(): np.arange(len(cand))}
        ev_u = ev_by_unit[u]
        keys = list(zip(*[ev_u[m].to_numpy() for m in match_on])) if match_on else [()] * len(ev_u)
        ev_ids = ev_u["event_id"].to_numpy()
        for eid, key in zip(ev_ids, keys):
            pool = pools.get(key)
            if pool is None or len(pool) == 0:
                continue
            take = pool if len(pool) <= n_per_event else rng.choice(pool, size=n_per_event, replace=False)
            for j in take:
                rec = {"unit": u, "event_id": eid, "seq": cand_seq[j]}
                if match_on:
                    rec.update(dict(zip(match_on, key)))
                plc.append(rec)
    cols = ["unit", "event_id", "seq", *match_on]
    return pd.DataFrame(plc, columns=cols)


def balance_diagnostics(
    tele: Telemetry,
    event_anchors: pd.DataFrame,
    placebo_anchors: pd.DataFrame,
    covariates: Sequence[str] = (),
) -> dict:
    """Covariate balance between event and placebo anchors, plus match coverage.

    For each covariate returns the standardized mean difference
    ``SMD = (mean_event - mean_placebo) / pooled_sd`` (numeric covariates; categorical
    covariates are one-hot expanded). Also reports how many events received at
    least one placebo (``events_matched``), the mean placebos per event, and the
    number of unmatched events (which are dropped from estimation).
    """
    f = tele.frame.set_index(["unit", "seq"])
    def _pull(anchors):
        keys = list(zip(anchors["unit"], anchors["seq"]))
        return f.loc[[k for k in keys if k in f.index], list(covariates)]
    ev = _pull(event_anchors); pl = _pull(placebo_anchors)
    rows = []
    for c in covariates:
        e, p = ev[c], pl[c]
        if pd.api.types.is_numeric_dtype(e) and e.nunique() > 2:
            sp = np.sqrt((e.var(ddof=1) + p.var(ddof=1)) / 2.0)
            smd = float((e.mean() - p.mean()) / sp) if sp > 0 else 0.0
            rows.append({"covariate": c, "level": "(numeric)", "event_mean": float(e.mean()),
                         "placebo_mean": float(p.mean()), "smd": smd})
        else:
            levels = sorted(set(e.dropna().unique()) | set(p.dropna().unique()), key=str)
            for lv in levels:
                pe, pp = float((e == lv).mean()), float((p == lv).mean())
                sp = np.sqrt((pe * (1 - pe) + pp * (1 - pp)) / 2.0)
                smd = float((pe - pp) / sp) if sp > 0 else 0.0
                rows.append({"covariate": c, "level": str(lv), "event_mean": pe, "placebo_mean": pp, "smd": smd})
    per_event = placebo_anchors.groupby("event_id").size() if len(placebo_anchors) else pd.Series(dtype=int)
    n_ev = int(event_anchors["event_id"].nunique())
    n_matched = int(per_event.index.nunique())
    return {
        "smd": pd.DataFrame(rows),
        "events_total": n_ev,
        "events_matched": n_matched,
        "events_unmatched": n_ev - n_matched,
        "mean_placebos_per_event": float(per_event.mean()) if len(per_event) else 0.0,
        "max_abs_smd": float(np.max(np.abs([r["smd"] for r in rows]))) if rows else np.nan,
    }
