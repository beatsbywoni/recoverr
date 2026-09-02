"""Illustrative example: post-lapse relearning in Duolingo spaced-repetition logs (memory domain).

Loads the public HLR learning traces (Settles & Meeder, 2016), keeps the most
active learners, and runs the recoverr pipeline in *item-stream* mode: a failure
is a lapse (an item answered wrongly in a session) and the recovery window is
the item's next reviews; the placebo baseline is a non-lapsed review of the same
learner matched on prior exposures and item difficulty, followed by that item's
next reviews.

Data (public, CC BY-NC 4.0): https://doi.org/10.7910/DVN/N8XJME
    -> settles.acl16.learning_traces.13m.csv.gz

Usage:
    python examples/hlr_quickstart.py /path/to/settles.acl16.learning_traces.13m.csv.gz --max-users 300
"""
from __future__ import annotations

import argparse

import recoverr as rc
from recoverr import io as rio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="path to settles.acl16.learning_traces.13m.csv.gz")
    ap.add_argument("--max-users", type=int, default=300)
    ap.add_argument("--nrows", type=int, default=None, help="rows to read (smoke tests)")
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()

    tele = rio.from_hlr(args.data, max_users=args.max_users, nrows=args.nrows)
    print(tele)

    pipe = rc.RecoveryPipeline(window=5, depth_span=(1, 2), stream="lexeme",
                               exclude_episode="session", n_perm=1000, n_boot=500)
    res = pipe.run(tele, anchor_rule=lambda g: (g["outcome"] >= 1).to_numpy(),
                   match_on=["hist_seen_bin", "lex_diff_bin"], min_events=10, seed=20260902)

    print(f"\nlapses: {res['n_events']:,} (with a later review of the item: {res['n_events_used']:,};"
          f" non-overlapping: {res['n_events_thinned']:,})")
    print(f"balance: max |SMD| = {res['balance']['max_abs_smd']:.3f}, unmatched events = {res['balance']['events_unmatched']}")
    print("\nmiss-rate deviation on the item's next reviews (lapsed - matched non-lapsed):")
    print(res["curve"].round(4).to_string(index=False))
    p = res["permutation"]
    print(f"\npermutation null: observed {p['obs']:.4f} vs null 97.5th pct {p['null_q975']:.4f} (p = {p['p_value']:.3g})")
    print("\nsplit-half reliability (thinned odd/even, Spearman-Brown):")
    for axis, r in res["reliability"].items():
        print(f"  {axis:10s} n={r['n_units']:4d}  r_sb={r['r_sb']:.3f}  95% CI [{r.get('ci_low', float('nan')):.2f}, {r.get('ci_high', float('nan')):.2f}]")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        c = res["curve"]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.axhline(0, color="0.7", lw=0.8)
        ax.plot(c["rel_pos"], c["deviation"], marker="o", color="#c0392b")
        ax.set_xlabel("subsequent review of the lapsed item")
        ax.set_ylabel("miss-rate deviation (lapsed - matched)")
        fig.tight_layout(); fig.savefig(args.fig, dpi=150)
        print(f"\nsaved figure -> {args.fig}")


if __name__ == "__main__":
    main()
