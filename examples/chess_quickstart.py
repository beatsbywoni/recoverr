"""Illustrative example: post-blunder recovery in online chess (performance domain).

Streams one month of the public Lichess database (games with Stockfish
[%eval] annotations), builds per-player move telemetry, and runs recoverr with
the game as the stream: a failure is a blunder ('??'), the recovery window is
the player's next moves in the same game, and the placebo baseline is a
non-blunder move of the same player matched on game phase and on the win
probability *after* the move (a position as good or bad as the one the blunder
produced, reached without a blunder).

Data (public, CC0): https://database.lichess.org/standard/lichess_db_standard_rated_2014-07.pgn.zst
(any month works; ~6-16 % of games carry evaluations)

Usage:
    python examples/chess_quickstart.py /path/to/lichess_db_standard_rated_2014-07.pgn.zst --max-games 40000 --max-players 100
"""
from __future__ import annotations

import argparse

import recoverr as rc
from recoverr import io as rio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn", help="Lichess .pgn or .pgn.zst")
    ap.add_argument("--max-games", type=int, default=40000, help="annotated games to parse")
    ap.add_argument("--max-players", type=int, default=100)
    ap.add_argument("--min-moves", type=int, default=500)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()

    moves = rio.parse_lichess_pgn(args.pgn, max_games=args.max_games)
    print(f"parsed {moves['game_id'].nunique():,} annotated games, {len(moves):,} moves")
    tele = rio.from_lichess_moves(moves, min_moves=args.min_moves, max_players=args.max_players)
    print(tele)

    pipe = rc.RecoveryPipeline(window=10, depth_span=(1, 3), stream="game_id", n_perm=1000, n_boot=500)
    res = pipe.run(tele, anchor_rule=lambda g: (g["blunder"] == 1).to_numpy(),
                   match_on=["phase", "wp_after_bin"], min_events=10, seed=20260903)

    print(f"\nblunders: {res['n_events']:,} (with a next move in the game: {res['n_events_used']:,};"
          f" non-overlapping: {res['n_events_thinned']:,})")
    print(f"balance: max |SMD| = {res['balance']['max_abs_smd']:.3f}")
    print("\nerror-rate deviation on the next own moves (after blunder - matched non-blunder move):")
    print(res["curve"][["rel_pos", "event", "placebo", "deviation", "n_units"]].round(4).to_string(index=False))
    f = res["fit"]
    print(f"\nexponential fit: tau = {f['tau']:.2f} moves, R2 = {f['r2']:.3f}, identified = {f['identified']}")
    p = res["permutation"]
    print(f"permutation null: observed {p['obs']:.4f} vs null 97.5th pct {p['null_q975']:.4f} (p = {p['p_value']:.3g})")
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
        ax.plot(c["rel_pos"], c["deviation"], marker="o", color="#c0392b", label="observed")
        ax.plot(c["rel_pos"], f["fitted"], "--", color="#2471a3", label=f"exp. fit, tau = {f['tau']:.1f}")
        ax.set_xlabel("own moves after the blunder (same game)")
        ax.set_ylabel("error-rate deviation")
        ax.legend(frameon=False); fig.tight_layout(); fig.savefig(args.fig, dpi=150)
        print(f"\nsaved figure -> {args.fig}")


if __name__ == "__main__":
    main()
