"""M2 tests: held-out gain (always) and multilevel model (if [bayes] installed)."""
import numpy as np
import pytest

import recoverr as rc
from recoverr import baseline, events, heldout, reliability


def _anchor_rule(g):
    return (g["phase"] == "anchor").to_numpy()


def _windows(seed=11):
    tele = rc.sim.simulate_recovery(n_units=50, events_per_unit=14, seed=seed)
    anchors = events.detect_anchors(tele, rule=_anchor_rule)
    plc = baseline.match_placebo(tele, anchors, window=20, seed=seed)
    ev_w = events.extract_windows(tele, anchors, 20, "event")
    plc_w = events.extract_windows(tele, plc[["unit", "event_id", "seq"]], 20, "placebo")
    return ev_w, plc_w


def test_heldout_gain_runs():
    ev_w, _ = _windows()
    out = heldout.heldout_gain(ev_w, span=(1, 5), tail_frac=0.3)
    assert out["n_test"] > 0
    for key in ("logloss_global", "logloss_user", "logloss_gain",
                "brier_global", "brier_user", "brier_gain"):
        assert np.isfinite(out[key])


def test_fit_mlm_if_bayes():
    pytest.importorskip("numpyro")
    ev_w, plc_w = _windows()
    res = reliability.fit_mlm(
        ev_w, plc_w, seed=0,
        mcmc={"num_warmup": 150, "num_samples": 150, "num_chains": 2},
    )
    summ = res["summary"]
    assert set(summ["param"]) == {"b0", "b_ev", "sigma_user", "sigma_user_ev"}
    # event effect on failure is positive (errors elevate post-failure failure rate)
    b_ev = summ.loc[summ["param"] == "b_ev", "mean"].item()
    assert b_ev > 0
    assert len(res["user_effects"]) == ev_w["unit"].nunique()
