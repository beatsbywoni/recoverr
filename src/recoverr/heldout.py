"""Held-out validation of personalization gain (C2 step 3 D10 / c2_03).

Holds out the tail of each unit's event sequence and compares a personalized
(shrunk per-unit) predictor of the early-window failure rate against a global
predictor by log-loss / Brier. Because the personalized estimate is formed on
the training events only, this corrects the in-sample optimism that inflates
one-sided within-person indices. Generalized from c2_03_reliability_heldout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def heldout_gain(
    ev_windows: pd.DataFrame,
    span: tuple[int, int] = (1, 5),
    tail_frac: float = 0.3,
    shrink_k: float = 8.0,
) -> dict:
    """Personalized-vs-global held-out prediction of early-window outcomes.

    Returns log-loss / Brier for the global and personalized predictors and the
    gain (global minus personalized; positive = personalization helps). Expects
    a binary ``outcome`` (0/1) in ``ev_windows``.
    """
    order = (ev_windows[["unit", "event_id"]].drop_duplicates()
             .assign(k=lambda d: d.groupby("unit").cumcount()))
    ev = ev_windows.merge(order, on=["unit", "event_id"])
    ev = ev[ev["rel_pos"].between(*span)].copy()
    n_ev = (order.groupby("unit")["k"].max() + 1).rename("n_ev")
    ev = ev.merge(n_ev, on="unit")
    is_test = ev["k"] >= ev["n_ev"] * (1 - tail_frac)
    tr, te = ev[~is_test], ev[is_test]
    if len(tr) == 0 or len(te) == 0:
        return {"n_test": 0}

    g = float(tr["outcome"].mean())  # global early-window failure rate
    agg = tr.groupby("unit")["outcome"].agg(["mean", "size"])
    w = agg["size"] / (agg["size"] + shrink_k)  # shrink toward global
    p_user = (w * agg["mean"] + (1 - w) * g).rename("p_user")
    te = te.join(p_user, on="unit").dropna(subset=["p_user"])
    if len(te) == 0:
        return {"n_test": 0}

    y = te["outcome"].to_numpy().astype(float)

    def score(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(ll.mean()), float(((p - y) ** 2).mean())

    ll_g, br_g = score(np.full(len(te), g))
    ll_u, br_u = score(te["p_user"].to_numpy())
    return {
        "n_test": int(len(te)),
        "logloss_global": ll_g, "logloss_user": ll_u, "logloss_gain": ll_g - ll_u,
        "brier_global": br_g, "brier_user": br_u, "brier_gain": br_g - br_u,
    }
