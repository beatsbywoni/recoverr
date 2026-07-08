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

    pipe = rc.RecoveryPipeline(window=20, depth_span=(1, 5))
    res = pipe.run(tele, anchor_rule=error_cluster_rule(2),
                   match_on=["days_bin", "format"], min_events=10, seed=20260708)

    print(f"\nevents detected: {res['n_events']:,}")
    print("\nrecovery curve (event - placebo error rate):")
    print(res["curve"].head(8).round(4).to_string(index=False))
    print("\nper-learner axes (describe):")
    print(res["axes"][["depth", "auc", "tau", "completeness", "level"]].describe().round(4))
    print("\nsplit-half reliability (Spearman-Brown):")
    for axis, r in res["reliability"].items():
        print(f"  {axis:10s} n={r['n_units']:4d}  r_sb={r['r_sb']}")

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
