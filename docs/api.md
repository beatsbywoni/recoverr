# API reference (v1.1.0)

Import as `import recoverr as rc`. Every stochastic step takes `seed`.

| Object | Purpose | Key arguments | Returns |
|---|---|---|---|
| `rc.Telemetry.from_frame(df, unit, seq, outcome, covariates)` | normalize a long table | column names | `Telemetry(frame, covariates)` sorted by (unit, seq) |
| `rc.RecoveryPipeline(...)` | end-to-end analysis | `window`, `depth_span`, `complete_span`, `n_per_event`, `fit_tau`, `stream`, `baseline` (`placebo`/`pre_event`), `placebo_exclusion`, `exclude_before`, `exclude_episode`, `exclude_overlapping`, `split` (`odd_even`/`first_second`/`temporal`), `thin_reliability`, `n_boot`, `n_perm`, `heldout_tail` | — |
| `RecoveryPipeline.run(tele, anchor_rule, match_on, min_events, seed)` | run it | rule = callable(frame) → bool mask | dict: `n_events`, `n_events_used`, `n_events_thinned`, `overlap_rate`, `overlap_rate_placebo`, `balance`, `curve`, `fit`, `axes`, `tau_unidentified_rate`, `reliability`, [`permutation`], [`heldout`] |
| `rc.events.outcome_threshold(min_value)` | built-in anchor rule | | rule |
| `rc.events.detect_anchors(tele, rule)` | anchors | | frame (unit, seq, event_id, covariates) |
| `rc.events.extract_windows(tele, anchors, window, kind, event_anchors, stream, direction, gap)` | post (or mirrored pre) windows | `stream` follows a sub-stream; `direction="pre"` | long frame (unit, event_id, rel_pos, outcome, kind, overlap) |
| `rc.events.thin_anchors(anchors, window)` | non-overlapping anchors | | frame |
| `rc.baseline.match_placebo(tele, anchors, window, n_per_event, match_on, seed, exclusion, exclude_before, exclude_episode, stream)` | placebo anchors | | frame (unit, event_id, seq, match_on…) |
| `rc.baseline.balance_diagnostics(tele, anchors, placebo_anchors, covariates)` | SMD table + coverage | | dict (`smd`, `events_matched`, `events_unmatched`, `mean_placebos_per_event`, `max_abs_smd`) |
| `rc.recovery.recovery_curve(ev_w, plc_w, weights)` | population curve | `unit` (default) / `pooled` | frame (rel_pos, event, placebo, deviation, n_units) |
| `rc.recovery.fit_parametric(curve, tau_bounds)` | exponential fit | | dict (`d0`, `tau`, `r2`, `fitted`, `identified`) |
| `rc.recovery.recovery_axes(ev_w, plc_w, window, depth_span, complete_span, fit_tau)` | per-unit axes | | frame (unit, depth, auc, tau, completeness, level) |
| `rc.reliability.split_half(ev_w, plc_w, axis, span, min_events, method, n_boot, seed, method_split)` | split-half r, Spearman–Brown, bootstrap CI | | dict |
| `rc.reliability.temporal_split_half(tele, anchor_rule, window, span, …, baseline, exclude_before, max_events, **placebo_kw)` | disjoint-segment reliability | | dict |
| `rc.reliability.fit_mlm(ev_w, plc_w, …)` | NumPyro binomial multilevel model (`[bayes]`) | | dict (posterior summaries, diagnostics, shrunken effects) |
| `rc.nulls.permutation_null(ev_w, plc_w, span, n_perm, seed, statistic)` | within-unit permutation test | `unit_mean` (default) / `pooled` | dict (`obs`, `null_mean`, `null_sd`, `null_q025`, `null_q975`, `p_value`, `null`) |
| `rc.heldout.heldout_gain(ev_w, span, tail_frac, shrink_k)` | personalized vs global held-out prediction | | dict (log-loss / Brier, gains) |
| `rc.sim.simulate_recovery(n_units, events_per_unit, window, baseline_rate, depth, tau, gap, heterogeneity, shape, seed, with_truth)` | planted-signal generator | shapes `exponential` / `linear` / `plateau` | `Telemetry` [, truth] |
| `rc.sim.run_ademp(...)` | factorial estimator benchmark | | dict (`replications`, `summary`, `tau`) |
| `rc.sim.simulate_endogenous_null(...)`, `rc.sim.run_ademp_null(...)`, `rc.sim.recent_rate_bins(...)` | pure-RTM null with outcome-defined anchors | | — |
| `rc.io.from_slam(path, track, max_users)` | Duolingo SLAM tarball → token telemetry | | `Telemetry` |
| `rc.io.from_hlr(path, binarize, max_users, min_rows, nrows)` | Duolingo HLR CSV → review telemetry | | `Telemetry` |
| `rc.io.parse_lichess_pgn(path, max_games, hash_players, skip_games)` | Lichess PGN(.zst) → move table | | frame |
| `rc.io.from_lichess_moves(moves, error_threshold, min_moves, max_players)` | move table → player telemetry | | `Telemetry` |
| `rc.io.from_chess(path, unit, seq, outcome, covariates)` | any move-level table | | `Telemetry` |

## Library-mode example (stages used independently)

```python
import recoverr as rc
from recoverr import events, baseline, recovery, reliability, nulls

anchors = events.detect_anchors(tele, rule=my_rule)
plc     = baseline.match_placebo(tele, anchors, window=20, match_on=["format"], seed=1,
                                 exclude_episode="ex_id")
ev_w    = events.extract_windows(tele, anchors, 20, "event")
plc_w   = events.extract_windows(tele, plc[["unit", "event_id", "seq"]], 20, "placebo",
                                 event_anchors=anchors)
curve   = recovery.recovery_curve(ev_w, plc_w)               # unit-weighted
fit     = recovery.fit_parametric(curve)                      # tau, R2, identified
axes    = recovery.recovery_axes(ev_w, plc_w, window=20)
keep    = events.thin_anchors(anchors, 20)["event_id"]
rel     = reliability.split_half(ev_w[ev_w.event_id.isin(keep)], plc_w[plc_w.event_id.isin(keep)],
                                 n_boot=1000, seed=1)
null    = nulls.permutation_null(ev_w, plc_w, n_perm=1000, seed=1)
```
