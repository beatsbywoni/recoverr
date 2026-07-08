"""Core pipeline test: recover a planted post-failure recovery signal."""
import numpy as np

import recoverr as rc
from recoverr import events, nulls


def _anchor_rule(g):
    # In the simulator, true anchors carry phase == 'anchor'.
    return (g["phase"] == "anchor").to_numpy()


def test_pipeline_recovers_planted_signal():
    tele = rc.sim.simulate_recovery(n_units=60, events_per_unit=14, window=20,
                                    depth=0.3, tau=5.0, seed=20260708)
    pipe = rc.RecoveryPipeline(window=20, depth_span=(1, 5))
    res = pipe.run(tele, anchor_rule=_anchor_rule, min_events=10, seed=1)

    # curve has one row per relative position
    assert len(res["curve"]) == 20
    # early-window deviation is positive (failures elevate error, then recover)
    early = res["curve"].query("rel_pos <= 5")["deviation"].mean()
    assert early > 0.03, early
    # per-unit depth axis is mostly positive
    depth = res["axes"]["depth"]
    assert np.nanmean(depth) > 0.02
    # split-half reliability is computable and finite
    r_sb = res["reliability"]["depth"]["r_sb"]
    assert np.isfinite(r_sb)


def test_permutation_null_detects_effect():
    tele = rc.sim.simulate_recovery(n_units=50, events_per_unit=12, seed=7)
    anchors = events.detect_anchors(tele, rule=_anchor_rule)
    from recoverr import baseline
    plc_anchors = baseline.match_placebo(tele, anchors, window=20, seed=2)
    ev_w = events.extract_windows(tele, anchors, 20, "event")
    plc_w = events.extract_windows(tele, plc_anchors[["unit", "event_id", "seq"]], 20, "placebo")
    out = nulls.permutation_null(ev_w, plc_w, span=(1, 5), n_perm=200, seed=3)
    assert out["obs"] > out["null_mean"]
    assert out["p_value"] < 0.05
