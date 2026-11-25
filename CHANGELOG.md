# Changelog

All notable changes to `recoverr` are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Documentation only; the code is unchanged from v1.1.0.

### Documentation
- Cite the published SoftwareX article (doi:10.1016/j.softx.2026.103056):
  `preferred-citation` and author ORCID in `CITATION.cff`, citation section in
  the README and `docs/index.md`.
- `docs/methods.md` section 7 lists defaults and specifications that the
  article does not state (window length, placebos per anchor, covariate bins,
  multilevel model, held-out shrinkage, the usage example's anchor rule, the
  Lichess failure definition, dependency versions).
- Erratum for the article: Section 3.1 says "seven modules"; the package has
  nine.

## [1.1.0] — 2026-09-02

Revision release prepared in response to peer review. Three defaults changed
because they biased v1.0.0 estimates; results computed with v1.0.0 should be
re-run.

### Changed (behaviour-affecting)
- **Placebo exclusion is now symmetric** (`match_placebo(exclusion="post")`,
  default). v1.0.0 barred every observation within ±`window` of any failure
  from placebo use, so placebo windows could never contain a later failure
  while event windows could; this conditioned the baseline on future success
  and inflated the deviation. The old rule is available as `exclusion="both"`
  (appropriate only for simulations with scheduled anchors). New
  `exclude_before` and `exclude_episode` options bar the anchor-defining span
  (e.g. the anchor's own exercise) from placebo use.
- **Population curve is unit-weighted** (`recovery_curve(weights="unit")`,
  default): within-unit event/baseline differences are averaged with equal
  weight per unit. The pooled curve (v1.0.0) mixes between-unit composition
  into the deviation and is kept as `deviation_pooled`.
- **Split-half reliability is computed on non-overlapping windows**
  (`RecoveryPipeline(thin_reliability=True)`, default) and each half uses its
  own baseline windows. Overlapping event windows and shared placebo draws
  inflated odd/even split-half correlations (≈ .45 under a pure null in
  simulation; ≈ 0 after the fix). `split="temporal"` (disjoint time segments
  per unit) and `split="first_second"` are available as conservative variants.
- `permutation_null` uses the unit-mean statistic by default (null centred at
  zero) with an exact hypergeometric shortcut for binary outcomes.
- `simulate_recovery` default gap between events is `2 * window + 5` so that
  clean, full-length placebo windows exist (τ recovery in the benchmark is now
  unbiased: 5.1 vs truth 5.0).

### Added
- `events.extract_windows(stream=..., direction="pre", gap=...)`: windows that
  follow a sub-stream (e.g. the same lexeme / the same game) and mirrored
  pre-event windows; `events.thin_anchors`.
- `RecoveryPipeline(baseline="pre_event", stream=..., exclude_before=...,
  exclude_episode=..., split=..., thin_reliability=..., n_boot=..., n_perm=...,
  heldout_tail=...)`; output now includes overlap rates for event and placebo
  windows, `n_events_thinned`, balance diagnostics (SMD), parametric fit with
  identifiability flag, per-unit unidentified-τ rate, bootstrap CIs for
  reliability, permutation null and held-out gain.
- `baseline.balance_diagnostics`, `recovery.fit_parametric`,
  `reliability.temporal_split_half`, `sim.simulate_endogenous_null`,
  `sim.run_ademp_null` (outcome-defined anchors, autocorrelated risk),
  `sim.recent_rate_bins`, shapes `linear` / `plateau` in the simulator.
- IO: `io.from_hlr` reads the raw public HLR CSV; `io.parse_lichess_pgn` and
  `io.from_lichess_moves` build move-level telemetry from the Lichess database
  (usernames hashed).
- Examples for all three domains (`examples/slam_quickstart.py`,
  `hlr_quickstart.py`, `chess_quickstart.py`), `docs/` (methods, API,
  reproduction), `CONTRIBUTING.md`, coverage in the test suite (90 %).

### Performance
- Vectorised window extraction, placebo sampling and permutation null:
  ≈ 4 s per 10⁶ observations for the full pipeline (see `docs/benchmark.md`).

## [1.0.0] — 2026-07

Initial release: schema, events, context-matched placebo baseline, recovery
axes (depth / τ / area index / completeness), split-half and NumPyro
multilevel reliability, permutation null, held-out gain, ADEMP simulator, SLAM
/ HLR / chess adapters. Zenodo DOI 10.5281/zenodo.21264167.
