# recoverr

[![tests](https://github.com/beatsbywoni/recoverr/actions/workflows/ci.yml/badge.svg)](https://github.com/beatsbywoni/recoverr/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21264167.svg)](https://doi.org/10.5281/zenodo.21264167)
[![PyPI](https://img.shields.io/pypi/v/recoverr-telemetry.svg)](https://pypi.org/project/recoverr-telemetry/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Open pipeline for estimating **within-person recovery dynamics after failure**
from behavioral telemetry — learning, memory and performance logs — in one
reusable API. The name reads *recover + err*: recovery from errors.

After a failure, how much worse does a person do, for how long, and does it
matter who they are? `recoverr` turns a long table of `(unit, seq, outcome)`
into a **failure-locked recovery curve** with a context-matched, within-person
baseline, three recovery axes per person (depth, speed, completeness), and the
diagnostics needed to trust them: covariate balance, overlap rates, an exact
permutation null, split-half reliability that is *not* inflated by overlapping
windows, bootstrap CIs and held-out predictive gain.

```
Telemetry → events → baseline → recovery → reliability / nulls / heldout
```

## Install

```bash
pip install recoverr-telemetry            # numpy / pandas / scipy / pyarrow
pip install "recoverr-telemetry[bayes]"   # + NumPyro / JAX multilevel model
pip install "recoverr-telemetry[chess]"   # + zstandard for Lichess .pgn.zst
```

Distribution name `recoverr-telemetry`, import name `recoverr`. Python ≥ 3.10.

## 60-second example

```python
import recoverr as rc

tele = rc.Telemetry.from_frame(df, unit="user", seq="pos_idx",
                               outcome="error", covariates=["ex_id", "format"])

pipe = rc.RecoveryPipeline(window=20, depth_span=(1, 5),
                           exclude_episode="ex_id",   # placebo may not start inside a failed exercise
                           n_perm=1000, n_boot=1000)
res = pipe.run(tele, anchor_rule=my_failure_rule, match_on=["format"],
               min_events=10, seed=20260708)

res["curve"]                 # unit-weighted event − placebo deviation by position
res["fit"]                   # exponential τ, R², identified flag
res["axes"]                  # per-person depth / auc / tau / completeness / level
res["balance"]["smd"]        # covariate balance of event vs placebo anchors
res["overlap_rate"]          # share of windows containing a later failure
res["permutation"]           # exact within-person permutation null
res["reliability"]           # split-half r_SB (+ bootstrap CI) on non-overlapping windows
```

Every stage is also a plain function (`rc.events`, `rc.baseline`, `rc.recovery`,
`rc.reliability`, `rc.nulls`, `rc.heldout`), see `docs/api.md`.

## Design choices that matter

- **Baseline** — for each failure, placebo anchors are non-failure moments of the
  *same* person, matched on context, outside the post-failure windows of earlier
  failures and outside the span that defines the anchor (`exclude_before`,
  `exclude_episode`). Placebo windows may contain later failures exactly as event
  windows may; both overlap rates are reported. A mirrored **pre-event** baseline
  (`baseline="pre_event"`) is available for pre/post designs.
- **Streams** — with `stream="lexeme"` (or `"game_id"`) windows follow the
  anchor's own sub-sequence: the next reviews of the word that was just
  forgotten, the next moves of the game in which the blunder happened.
- **Curve** — averaged within person first, then across people, so that people
  with many events and few placebos cannot manufacture a pooled difference.
- **Speed** — reported two ways: a parametric decay constant τ (flagged when
  unidentified) and a nonparametric area index.
- **Reliability** — computed on thinned, non-overlapping windows with separate
  baselines per half (odd/even or first/second), or on disjoint time segments
  (`split="temporal"`). Overlapping windows inflate a naive odd/even split to
  ≈ .45 under a pure null; the fix brings it to ≈ 0 (tested).
- **Inference** — exact hypergeometric permutation null for binary outcomes,
  percentile-bootstrap CIs, held-out personalized-vs-global gain.

## Worked examples (public data)

| domain | data | design | script |
|---|---|---|---|
| learning | Duolingo SLAM (2.6 M tokens) | error cluster → next 20 tokens | `examples/slam_quickstart.py` |
| memory | Duolingo HLR (12.9 M reviews) | lapse → next reviews of the same word | `examples/hlr_quickstart.py` |
| performance | Lichess games with Stockfish evals | blunder → next moves of the same game | `examples/chess_quickstart.py` |

See `docs/reproduce.md` for download links and the exact commands.

## Simulation benchmarks

`rc.sim.run_ademp` plants known recovery signals (exponential / linear /
plateau shapes, per-person heterogeneity) and compares placebo-matched,
pre-event and naive in-sample estimators against the analytic truth (bias, RMSE,
SD, τ recovery). `rc.sim.run_ademp_null` generates a *pure regression-to-the-mean
null* with outcome-defined anchors and autocorrelated risk, where every
baseline's bias can be seen directly.

## Performance

≈ 4 s and ≈ 0.5 GB per 10⁶ observations for the full pipeline on one CPU core
(`docs/benchmark.md`); Lichess PGN parsing streams ≈ 2.5 k annotated games/s.

## Documentation, tests, contributing

`docs/methods.md` gives the formal definitions; `docs/api.md` the reference.
`pytest -q --cov=recoverr` runs 26 self-contained tests (≈ 90 % coverage) on
Python 3.10–3.12 in CI. Contributions and issues: `CONTRIBUTING.md`.

## Citation

Jeong H. recoverr: an open pipeline for within-person recovery dynamics from
behavioral telemetry. Version 1.1.0. Zenodo; 2026. https://doi.org/10.5281/zenodo.22243328
(concept DOI for all versions: https://doi.org/10.5281/zenodo.21264167; see `CITATION.cff`).

Preregistration and reanalysis materials: https://osf.io/qnfth/. MIT licensed.
