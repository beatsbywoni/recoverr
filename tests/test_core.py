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
    # anchors are scheduled (exogenous) in the simulator, so the clean
    # counterfactual baseline excludes placebo windows spanning a later anchor
    pipe = rc.RecoveryPipeline(window=20, depth_span=(1, 5), placebo_exclusion="both")
    res = pipe.run(tele, anchor_rule=_anchor_rule, min_events=5, seed=1)  # min_events applies per temporal half

    # curve has one row per relative position
    assert len(res["curve"]) == 20
    # early-window deviation is positive (failures elevate error, then recover)
    early = res["curve"].query("rel_pos <= 5")["deviation"].mean()
    assert early > 0.03, early
    # per-unit depth axis is mostly positive
    depth = res["axes"]["depth"]
    assert np.nanmean(depth) > 0.02
    # split-half reliability (thinned odd/even, default) is computable, finite and
    # positive for planted per-unit heterogeneity; the temporal variant also runs
    r_sb = res["reliability"]["depth"]["r_sb"]
    assert np.isfinite(r_sb) and r_sb > 0, r_sb
    assert res["reliability"]["depth"]["thinned"] is True
    res_t = rc.RecoveryPipeline(window=20, placebo_exclusion="both", split="temporal").run(
        tele, anchor_rule=_anchor_rule, min_events=5, seed=1)
    assert res_t["reliability"]["depth"]["method_split"] == "temporal"
    assert np.isfinite(res_t["reliability"]["depth"]["r_sb"])


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


def test_placebo_exclusion_modes_are_symmetric():
    """'post' lets placebo windows contain later failures (like event windows);
    'both' never does. Overlap is flagged for both window types."""
    tele = rc.sim.simulate_recovery(n_units=20, events_per_unit=8, window=20, gap=8, seed=3)
    anchors = events.detect_anchors(tele, rule=_anchor_rule)
    from recoverr import baseline
    p_post = baseline.match_placebo(tele, anchors, window=20, seed=1, exclusion="post")
    p_both = baseline.match_placebo(tele, anchors, window=20, seed=1, exclusion="both")
    w_post = events.extract_windows(tele, p_post[["unit", "event_id", "seq"]], 20, "placebo", event_anchors=anchors)
    w_both = events.extract_windows(tele, p_both[["unit", "event_id", "seq"]], 20, "placebo", event_anchors=anchors)
    assert w_post["overlap"].mean() > 0.5          # gap (8) < window (20): most placebo windows span the next anchor
    assert w_both["overlap"].mean() == 0.0
    # no placebo anchor lies inside a post-failure window in either mode
    ev = anchors.groupby("unit")["seq"].apply(np.sort).to_dict()
    for _, r in p_post.iterrows():
        d = r["seq"] - ev[r["unit"]]
        assert not np.any((d > 0) & (d <= 20))


def test_split_half_variants_under_pure_null():
    """Odd/even splits of one analysis are inflated when event windows overlap
    and placebo draws are shared; the temporal split is not."""
    from recoverr import baseline, reliability
    from recoverr.sim import endogenous_anchor_rule, simulate_endogenous_null
    oe, thinned, tmp = [], [], []
    for seed in range(3):
        tele = simulate_endogenous_null(n_units=60, n_obs=600, rho=0.0, seed=seed)
        a = events.detect_anchors(tele, rule=endogenous_anchor_rule)
        p = baseline.match_placebo(tele, a, window=20, seed=1)
        e = events.extract_windows(tele, a, 20, "event")
        w = events.extract_windows(tele, p[["unit", "event_id", "seq"]], 20, "placebo", event_anchors=a)
        oe.append(reliability.split_half(e, w, span=(1, 5))["r_split"])
        keep = events.thin_anchors(a, 20)["event_id"]
        thinned.append(reliability.split_half(e[e["event_id"].isin(keep)], w[w["event_id"].isin(keep)], span=(1, 5))["r_split"])
        tmp.append(reliability.temporal_split_half(tele, endogenous_anchor_rule, window=20, min_events=5, seed=1)["r_split"])
    assert np.mean(oe) > 0.2                # inflated by overlapping windows
    assert abs(np.mean(thinned)) < 0.15     # ~0 once windows do not overlap
    assert abs(np.mean(tmp)) < 0.2          # ~0 for disjoint time segments
