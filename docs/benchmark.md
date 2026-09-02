# Performance benchmark

Environment: Python 3.10.12, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3; Linux aarch64, 4 vCPU, 4.1 GB RAM (sandbox);
single-threaded. Simulated telemetry (12 events/unit, 20-obs windows, 45-obs gaps); times in seconds, wall clock.

| rows | units | events | detect | match_placebo | windows | curve+axes+τ | reliability | permutation (1,000) | balance | **pipeline total** | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 × 10⁴ | 13 | 156 | .01 | .00 | .01 | .02 | .07 | .00 | .00 | **.05** | 160 MB |
| 1.0 × 10⁵ | 126 | 1,512 | .04 | .03 | .08 | .17 | .09 | .01 | .02 | **.39** | 207 MB |
| 1.0 × 10⁶ | 1,263 | 15,156 | .43 | .33 | .90 | 1.57 | .26 | .09 | .25 | **4.1** | 657 MB |
| 3.0 × 10⁶ | 3,788 | 45,456 | 1.36 | 1.00 | 2.83 | 4.83 | .65 | .29 | .80 | **12.5** | 1.6 GB |

Scaling is linear in rows (≈ 4 s and ≈ 0.5 GB per 10⁶ rows, the latter dominated by the pandas long table and the
windows table, which holds `window` rows per event and per placebo). 10⁷ rows exceeded the 4 GB sandbox; on a 16 GB
workstation the extrapolated cost is ≈ 40 s / 5 GB. The per-unit τ fits (SciPy `curve_fit` per unit) are the largest
single stage; the exact hypergeometric permutation null costs < 0.3 s for 1,000 draws at 3 × 10⁶ rows.

Real-data runs (same machine, full pipeline incl. 1,000 permutations, 1,000 bootstrap draws and held-out scoring):
| dataset | rows | units | events | mode | seconds |
|---|---|---|---|---|---|
| SLAM en_es, 1,000 learners | 989,818 | 1,000 | 25,445 | token stream | 17 (odd/even) / 42 (temporal split) |
| HLR, 300 learners | 901,390 | 300 | 152,926 | item stream (`stream="lexeme"`) | 66 |
| Lichess 2014-07, 200 players | 1,738,243 | 200 | 86,742 | game stream (`stream="game_id"`) | 11 |
| Lichess PGN parsing, 175 k annotated games (12.1 M moves) | — | — | — | streamed .zst | ≈ 70 (5 chunks) |

Script: `benchmarks/benchmark.py` in the repository (run one size per invocation).
