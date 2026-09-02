"""Illustrative example: post-error recovery in Duolingo SLAM (learning domain).

Loads the public SLAM corpus, defines an error-cluster failure anchor
(an exercise with >= 2 token errors), runs the recoverr pipeline, and reports the
recovery curve, per-learner axes, and split-half reliability. Optionally saves a
figure.

Data (public, CC BY-NC 4.0): Duolingo SLAM 2018,
Harvard Dataverse DOI 10.7910/DVN/8SWHNO -> data_en_es.tar.gz

Usage:
    python examples/slam_quickstart.py /path/to/data_en_es.tar.gz --track en_es --max-users 300
"""
from __future__ import annotations

import argparse

import recoverr as rc
from recoverr import io as rio


def error_cluster_rule(min_errors: int = 2):
    """SLAM anchor: last token of any exercise with >= min_errors token errors."""
    def rule(g):
        ex_err = g.groupby("ex_id")["outcome"].transform("sum")
        is_last = g["seq"] == g.groupby("ex_id")["seq"].transform("max")
        return ((ex_err >= min_errors) & is_last).to_numpy()
    return rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="path to data_{track}.tar.gz")
    ap.add_argument("--track", default="en_es")
    ap.add_argument("--max-users", type=int, default=300)
    ap.add_argument("--fig", default=None, help="optional output PNG path")
    args = ap.parse_args()

    tele = rio.from_slam(args.data, track=args.track, max_users=args.max_users)
    print(tele)

    # placebo anchors may not start inside a failed exercise (exclude_episode) and
    # are matched on the learner's days-since-start bin and the exercise format
    pipe = rc.RecoveryPipeline(window=20, depth_span=(1, 5), exclude_episode="ex_id",
                               n_perm=1000, n_boot=500)
    res = pipe.run(tele, anchor_rule=error_cluster_rule(2),
                   match_on=["days_bin", "format"], min_events=10, seed=20260708)

    print(f"\nevents detected: {res['n_events']:,} (non-overlapping: {res['n_events_thinned']:,};"
          f" windows containing a later failure: {res['overlap_rate']:.1%} event / {res['overlap_rate_placebo']:.1%} placebo)")
    print(f"balance: max |SMD| = {res['balance']['max_abs_smd']:.3f}, unmatched events = {res['balance']['events_unmatched']}")
    print("\nrecovery curve (unit-weighted event - placebo error rate):")
    print(res["curve"][["rel_pos", "event", "placebo", "deviation", "n_units"]].head(8).round(4).to_string(index=False))
    f = res["fit"]
    print(f"\nexponential fit: tau = {f['tau']:.1f} tokens, R2 = {f['r2']:.2f}, identified = {f['identified']}"
          f" (per-learner tau unidentified: {res['tau_unidentified_rate']:.0%}; area index recommended when unidentified)")
    p = res["permutation"]
    print(f"permutation null: observed {p['obs']:.4f} vs null 97.5th pct {p['null_q975']:.4f} (p = {p['p_value']:.3g})")
    print("\nper-learner axes (describe):")
    print(res["axes"][["depth", "auc", "tau", "completeness", "level"]].describe().round(4))
    print("\nsplit-half reliability (thinned odd/even, Spearman-Brown):")
    for axis, r in res["reliability"].items():
        print(f"  {axis:10s} n={r['n_units']:4d}  r_sb={r['r_sb']:.3f}  95% CI [{r.get('ci_low', float('nan')):.2f}, {r.get('ci_high', float('nan')):.2f}]")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        c = res["curve"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axhline(0, color="0.7", lw=0.8)
        ax.plot(c["rel_pos"], c["deviation"], marker="o", color="#c0392b")
        ax.set_xlabel("relative position after failure (tokens)")
        ax.set_ylabel("error-rate deviation (event - placebo)")
        ax.set_title(f"SLAM {args.track}: post-error recovery")
        fig.tight_layout()
        fig.savefig(args.fig, dpi=150)
        print(f"\nsaved figure -> {args.fig}")


if __name__ == "__main__":
    main()
