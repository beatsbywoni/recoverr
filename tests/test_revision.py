"""Tests for the v1.1.0 additions: stream mode, pre-event baseline, thinning,
placebo exclusion rules, balance diagnostics, permutation variants, temporal
split-half, endogenous-null ADEMP, and the HLR / Lichess adapters."""
import gzip

import numpy as np
import pandas as pd
import pytest

import recoverr as rc
from recoverr import baseline, events, nulls, recovery, reliability, sim
from recoverr import io as rio


def _anchor_rule(g):
    return (g["phase"] == "anchor").to_numpy()


# ---------------------------------------------------------------- events
def test_extract_windows_pre_direction_and_gap():
    df = pd.DataFrame({"unit": ["a"] * 12, "seq": range(12), "outcome": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]})
    tele = rc.Telemetry.from_frame(df)
    anchors = pd.DataFrame({"unit": ["a"], "seq": [3], "event_id": ["a:3"]})
    post = events.extract_windows(tele, anchors, 3, "event")
    assert post["rel_pos"].tolist() == [1, 2, 3] and post["outcome"].tolist() == [0, 0, 0]
    pre = events.extract_windows(tele, anchors, 3, "pre_event", direction="pre", gap=0)
    assert pre["outcome"].tolist() == [1, 0, 0]          # seq 2, 1, 0 mirrored
    pre_gap = events.extract_windows(tele, anchors, 3, "pre_event", direction="pre", gap=1)
    assert pre_gap["outcome"].tolist() == [0, 0]         # seq 1, 0 (seq 2 skipped)
    with pytest.raises(ValueError):
        events.extract_windows(tele, anchors, 3, "event", direction="sideways")


def test_extract_windows_stream_mode_follows_same_stream():
    df = pd.DataFrame({"unit": ["a"] * 10, "seq": range(10),
                       "outcome": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                       "item": ["x", "y", "x", "y", "x", "y", "x", "x", "y", "x"]})
    tele = rc.Telemetry.from_frame(df, covariates=["item"])
    anchors = pd.DataFrame({"unit": ["a", "a"], "seq": [0, 3], "event_id": ["a:0", "a:3"]})
    w = events.extract_windows(tele, anchors, 3, "event", stream="item")
    # anchor seq 0 (item x): next x reviews are seq 2, 4, 6 -> outcomes 0, 0, 0
    assert w[w["event_id"] == "a:0"]["outcome"].tolist() == [0, 0, 0]
    # anchor seq 3 (item y): next y reviews are seq 5, 8 -> outcomes 0, 0 (truncated)
    assert w[w["event_id"] == "a:3"]["outcome"].tolist() == [0, 0]
    # overlap: anchor at seq 7 (item x) is not in ``anchors`` -> no overlap flags
    assert not w["overlap"].any()
    anchors2 = pd.concat([anchors, pd.DataFrame({"unit": ["a"], "seq": [7], "event_id": ["a:7"]})])
    w2 = events.extract_windows(tele, anchors2, 4, "event", stream="item")
    assert bool(w2[w2["event_id"] == "a:0"]["overlap"].iloc[0])   # seq 7 (x) inside x-window of seq 0
    pre = events.extract_windows(tele, anchors2, 2, "pre_event", stream="item", direction="pre")
    assert pre[pre["event_id"] == "a:7"]["outcome"].tolist() == [0, 0]  # x reviews at seq 6, 4


def test_thin_anchors_non_overlapping():
    a = pd.DataFrame({"unit": ["a"] * 5 + ["b"], "seq": [0, 5, 12, 30, 31, 0],
                      "event_id": ["a:0", "a:5", "a:12", "a:30", "a:31", "b:0"]})
    t = events.thin_anchors(a, window=10)
    assert t["event_id"].tolist() == ["a:0", "a:12", "a:30", "b:0"]


# -------------------------------------------------------------- baseline
def test_match_placebo_exclusion_options():
    tele = sim.simulate_recovery(n_units=5, events_per_unit=4, window=10, gap=30, seed=1)
    anchors = events.detect_anchors(tele, rule=_anchor_rule)
    with pytest.raises(ValueError):
        baseline.match_placebo(tele, anchors, window=10, exclusion="neither")
    p_post = baseline.match_placebo(tele, anchors, window=10, seed=0)
    p_before = baseline.match_placebo(tele, anchors, window=10, seed=0, exclude_before=5)
    p_ep = baseline.match_placebo(tele, anchors, window=10, seed=0, exclude_episode="phase")
    ev = anchors.groupby("unit")["seq"].apply(np.sort).to_dict()
    for _, r in p_before.iterrows():
        d = ev[r["unit"]] - r["seq"]
        assert not np.any((d > 0) & (d <= 5))                 # nothing within 5 before an anchor
    assert not tele.frame.set_index(["unit", "seq"]).loc[list(zip(p_ep["unit"], p_ep["seq"])), "phase"].eq("anchor").any()
    assert len(p_post) >= len(p_before)


def test_match_placebo_stream_mode_and_matching_keys():
    df = pd.DataFrame({"unit": ["a"] * 12, "seq": range(12),
                       "outcome": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       "item": ["x", "x", "x", "y", "y", "y", "x", "y", "x", "y", "x", "y"],
                       "k": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]})
    tele = rc.Telemetry.from_frame(df, covariates=["item", "k"])
    anchors = events.detect_anchors(tele, rule=lambda g: (g["outcome"] == 1).to_numpy())
    p = baseline.match_placebo(tele, anchors, window=2, seed=0, stream="item", match_on=["k"], n_per_event=5)
    # x reviews at seq 1 and 2 are inside the post-window (positions within stream) -> excluded
    assert not set(p["seq"]).intersection({0, 1, 2})
    assert (p["k"] == 0).all()                                # matched on k of the anchor


def test_balance_diagnostics_reports_smd_and_coverage():
    tele = sim.simulate_recovery(n_units=8, events_per_unit=5, seed=2)
    anchors = events.detect_anchors(tele, rule=_anchor_rule)
    plc = baseline.match_placebo(tele, anchors, window=20, seed=0, exclusion="both")
    b = baseline.balance_diagnostics(tele, anchors, plc, ["phase"])
    assert b["events_total"] == len(anchors) and b["events_matched"] + b["events_unmatched"] == b["events_total"]
    assert set(b["smd"]["covariate"]) == {"phase"} and np.isfinite(b["max_abs_smd"])
    assert 0 < b["mean_placebos_per_event"] <= 3


# -------------------------------------------------------------- recovery
def test_recovery_curve_unit_vs_pooled_weights():
    # unit A: 2 event windows, 1 placebo; unit B: 1 event, 2 placebos -> pooled != unit-weighted
    ev = pd.DataFrame({"unit": ["A", "A", "B"], "event_id": ["A:1", "A:2", "B:1"], "rel_pos": 1, "outcome": [1, 1, 0]})
    pl = pd.DataFrame({"unit": ["A", "B", "B"], "event_id": ["A:1", "B:1", "B:1"], "rel_pos": 1, "outcome": [0, 0, 0]})
    cu = recovery.recovery_curve(ev, pl, weights="unit"); cp = recovery.recovery_curve(ev, pl, weights="pooled")
    assert cu["deviation"].item() == pytest.approx(0.5) and cp["deviation"].item() == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        recovery.recovery_curve(ev, pl, weights="median")


def test_fit_parametric_flags_unidentified():
    flat = pd.DataFrame({"rel_pos": np.arange(1, 21), "deviation": np.full(20, 0.1)})
    assert recovery.fit_parametric(flat)["identified"] is False      # tau at the upper bound
    neg = pd.DataFrame({"rel_pos": np.arange(1, 21), "deviation": -np.ones(20)})
    assert recovery.fit_parametric(neg)["identified"] is False       # non-positive curve
    good = pd.DataFrame({"rel_pos": np.arange(1, 21), "deviation": 0.3 * np.exp(-np.arange(1, 21) / 4)})
    f = recovery.fit_parametric(good)
    assert f["identified"] and f["tau"] == pytest.approx(4.0, abs=0.05) and f["r2"] > 0.99


# ----------------------------------------------------------------- nulls
def test_permutation_null_pooled_and_continuous_paths():
    tele = sim.simulate_recovery(n_units=20, events_per_unit=6, seed=4)
    a = events.detect_anchors(tele, rule=_anchor_rule); p = baseline.match_placebo(tele, a, window=20, seed=1, exclusion="both")
    e = events.extract_windows(tele, a, 20, "event"); w = events.extract_windows(tele, p[["unit", "event_id", "seq"]], 20, "placebo")
    r_unit = nulls.permutation_null(e, w, n_perm=200, seed=0)
    r_pool = nulls.permutation_null(e, w, n_perm=200, seed=0, statistic="pooled")
    assert r_unit["p_value"] < 0.05 and r_pool["p_value"] < 0.05 and abs(r_unit["null_mean"]) < 0.01
    e_cont = e.assign(outcome=e["outcome"].astype(float) + 1e-6)     # forces the explicit-shuffle path
    r_cont = nulls.permutation_null(e_cont, w, n_perm=50, seed=0)
    assert r_cont["null_sd"] == pytest.approx(r_unit["null_sd"], rel=0.5)
    with pytest.raises(ValueError):
        nulls.permutation_null(e, w, statistic="median")


# ----------------------------------------------------------- reliability
def test_split_half_first_second_and_bootstrap_ci():
    tele = sim.simulate_recovery(n_units=40, events_per_unit=12, depth=0.3, seed=5)
    a = events.detect_anchors(tele, rule=_anchor_rule); p = baseline.match_placebo(tele, a, window=20, seed=1, exclusion="both")
    e = events.extract_windows(tele, a, 20, "event"); w = events.extract_windows(tele, p[["unit", "event_id", "seq"]], 20, "placebo")
    r = reliability.split_half(e, w, min_events=6, method_split="first_second", n_boot=100, seed=0)
    assert r["method_split"] == "first_second" and r["ci_low"] <= r["r_sb"] <= r["ci_high"]
    with pytest.raises(ValueError):
        reliability.split_half(e, w, method_split="random")
    rt = reliability.temporal_split_half(tele, _anchor_rule, window=20, min_events=3, seed=0, n_boot=50, exclusion="both")
    assert rt["method_split"] == "temporal" and "ci_low" in rt and rt["n_units"] == 40
    rt_pre = reliability.temporal_split_half(tele, _anchor_rule, window=20, min_events=3, seed=0, baseline="pre_event", exclude_before=1)
    assert np.isfinite(rt_pre["r_sb"])
    rt_max = reliability.temporal_split_half(tele, _anchor_rule, window=20, min_events=3, max_events=3, seed=0, exclusion="both")
    assert rt_max["n_units"] == 40


# --------------------------------------------------------------- pipeline
def test_pipeline_pre_event_baseline_and_temporal_split():
    tele = sim.simulate_recovery(n_units=30, events_per_unit=10, depth=0.3, seed=6)
    res = rc.RecoveryPipeline(window=20, baseline="pre_event", exclude_before=1, split="temporal", n_perm=100).run(
        tele, anchor_rule=_anchor_rule, min_events=3, seed=1)
    assert res["curve"].query("rel_pos <= 5")["deviation"].mean() > 0.05
    assert res["permutation"]["p_value"] < 0.05 and res["reliability"]["depth"]["method_split"] == "temporal"
    assert "deviation_pooled" in res["curve"].columns and res["n_events_thinned"] <= res["n_events"]
    with pytest.raises(ValueError):
        rc.RecoveryPipeline(baseline="global").run(tele, anchor_rule=_anchor_rule)


# -------------------------------------------------------------------- sim
def test_endogenous_null_and_recent_bins():
    tele = sim.simulate_endogenous_null(n_units=5, n_obs=200, rho=0.5, seed=1)
    assert tele.frame["outcome"].isin([0, 1]).all() and len(tele) == 1000
    tele_b = sim.recent_rate_bins(tele, m=10, lag=2, bins=4)
    assert "recent_bin" in tele_b.covariates and tele_b.frame["recent_bin"].min() == -1
    a = events.detect_anchors(tele, rule=sim.endogenous_anchor_rule)
    f = tele.frame.set_index(["unit", "seq"])
    for _, r in a.head(20).iterrows():                           # every anchor is a failure preceded by a failure
        assert f.loc[(r["unit"], r["seq"]), "outcome"] == 1 and f.loc[(r["unit"], r["seq"] - 1), "outcome"] == 1
    out = sim.run_ademp_null(n_units=15, n_obs=300, rho_grid=(0.0,), n_reps=2, seed=3)
    s = out["summary"]
    assert set(s["estimator"]) == {"placebo_matched", "placebo_matched_recent", "naive_onesided", "naive_twosided", "pre_event", "pre_event_gap"}
    assert s.set_index("estimator").loc["pre_event", "bias"] < -0.05     # includes the defining failure
    assert abs(s.set_index("estimator").loc["pre_event_gap", "bias"]) < 0.05
    with pytest.raises(ValueError):
        sim._shape(np.arange(3), 5.0, "sigmoid")


# --------------------------------------------------------------------- io
def test_from_hlr_raw_csv(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for u in range(3):
        for t in range(40):
            for lex in ("a", "b", "c"):
                seen = 2; correct = int(rng.integers(0, 3))
                rows.append(dict(p_recall=correct / seen, timestamp=1000 + 100 * t, delta=100, user_id=f"u{u}",
                                 learning_language="es", ui_language="en", lexeme_id=f"lex_{lex}",
                                 lexeme_string=f"{lex}/{lex}<n><sg>", history_seen=1 + t, history_correct=t,
                                 session_seen=seen, session_correct=correct))
    path = tmp_path / "traces.csv.gz"
    with gzip.open(path, "wt") as fh:
        pd.DataFrame(rows).to_csv(fh, index=False)
    tele = rio.from_hlr(path, min_rows=10, max_users=2)
    assert tele.n_units == 2 and tele.frame["outcome"].isin([0, 1]).all()
    for c in ["lexeme", "session", "hist_seen_bin", "lex_diff_bin", "pos", "lang"]:
        assert c in tele.frame.columns
    assert (tele.frame["pos"] == "n").all()
    tele_rate = rio.from_hlr(path, binarize=False, min_rows=10)
    assert tele_rate.frame["outcome"].between(0, 1).all()


_PGN = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abc12345"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[UTCDate "2014.07.01"]
[UTCTime "10:00:00"]
[WhiteElo "1500"]
[BlackElo "1450"]
[TimeControl "300+0"]

1. e4 { [%eval 0.3] } 1... e5 { [%eval 0.2] } 2. Qh5?? { [%eval -2.0] } 2... Nc6 { [%eval -1.8] } 3. Bc4 { [%eval -1.9] } 3... g6?! { [%eval -1.2] } 4. Qf3 { [%eval -1.3] } 4... Nf6?? { [%eval #3] } 5. Qxf7# { [%eval #0] } 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/noeval01"]
[White "carol"]
[Black "dave"]
[Result "0-1"]

1. d4 d5 2. c4 e6 0-1

"""


def test_parse_lichess_pgn_and_from_lichess_moves(tmp_path):
    path = tmp_path / "games.pgn"
    path.write_text(_PGN)
    mv = rio.parse_lichess_pgn(path)
    assert mv["game_id"].nunique() == 1 and len(mv) == 9            # only the annotated game
    assert mv["player"].nunique() == 2 and "alice" not in set(mv["player"])  # hashed
    white = mv[mv["color"] == "w"]
    assert white["nag"].tolist() == ["", "??", "", "", ""]
    assert white["wp_loss"].iloc[1] > 0.15                             # the Qh5?? blunder (0.2 -> -2.0 = 19 wp points)
    assert (mv["wp_before"].between(0, 1) & mv["wp_after"].between(0, 1)).all()
    mv_raw = rio.parse_lichess_pgn(path, hash_players=False)
    assert set(mv_raw["player"]) == {"alice", "bob"}
    tele = rio.from_lichess_moves(mv, min_moves=1)
    assert tele.n_units == 2 and tele.frame["blunder"].sum() == 2 and tele.frame["outcome"].sum() == 2
    for c in ["game_id", "phase", "wp_before_bin", "wp_after_bin", "elo_bin", "time_control", "color"]:
        assert c in tele.frame.columns
    assert rio.parse_lichess_pgn(path, skip_games=1).empty
