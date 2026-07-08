"""Held-out validation of personalization gain (C2 step 3 D10 / c2_03).

Holds out the tail of each unit's stream and compares a personalized (shrunk
per-unit) predictor against a global predictor by log-loss / Brier, correcting
the in-sample optimism that inflates one-sided within-person indices. Ported in M2.
"""
from __future__ import annotations


def heldout_gain(*args, **kwargs):  # pragma: no cover
    """Personalized-vs-global held-out gain. Ported from c2_03 in M2."""
    raise NotImplementedError(
        "heldout_gain: port c2_03_reliability_heldout (tail hold-out, "
        "shrink-to-global, log-loss/Brier gain) in M2."
    )
