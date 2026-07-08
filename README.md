# recoverr

[![tests](https://github.com/beatsbywoni/recoverr/actions/workflows/ci.yml/badge.svg)](https://github.com/beatsbywoni/recoverr/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21264167.svg)](https://doi.org/10.5281/zenodo.21264167)
[![PyPI](https://img.shields.io/pypi/v/recoverr-telemetry.svg)](https://pypi.org/project/recoverr-telemetry/)

Open pipeline for estimating **within-person recovery dynamics after failure** from
behavioral telemetry — learning, memory, and performance logs — in one reusable API.

Failure-and-recovery is measured ad hoc across learning analytics, sport, and
cognition, with no shared tooling. `recoverr` packages a validated pipeline that
estimates *how deeply, how fast, and how completely* a person bounces back after
errors, with the reliability and out-of-sample checks that make those estimates
trustworthy.

## Pipeline

```
Telemetry → events → baseline → recovery → reliability → heldout
```

1. **events** — detect failure anchors from the stream (pluggable rules).
2. **baseline** — context-matched *placebo* windows (regression-to-the-mean-safe).
3. **recovery** — three axes: depth, speed (parametric τ / nonparametric AUC), completeness.
4. **reliability** — split-half + multilevel shrinkage of per-person differences.
5. **heldout** — personalized-vs-global validation to correct in-sample optimism.

## Install

```bash
pip install recoverr-telemetry           # light: numpy / pandas / scipy / pyarrow
pip install "recoverr-telemetry[bayes]"  # adds NumPyro + JAX for the multilevel model
```

The distribution name is `recoverr-telemetry`; the import name is `recoverr` (e.g. `import recoverr as rc`).

## Quickstart

```python
import recoverr as rc

tele = rc.Telemetry.from_frame(df, unit="user", seq="pos_idx",
                               outcome="error", covariates=["days_bin"])
res = rc.RecoveryPipeline(window=20, depth_span=(1, 5)).run(
    tele, anchor_rule=rc.events.outcome_threshold(1),
    match_on=["days_bin"], min_events=10, seed=20260708)

res["curve"]        # event-minus-placebo deviation by relative position
res["axes"]         # per-unit depth / auc / tau / completeness / level
res["reliability"]  # split-half r_sb for depth and auc
```

## Status

Preregistration and reanalysis materials: https://osf.io/qnfth/
Target journal: *SoftwareX*. Single-author. MIT licensed.

Implemented and tested (M1–M3): the recovery core (events, baseline, recovery,
split-half reliability, permutation null, simulator), the SLAM/HLR IO adapters
(`io.from_slam`, `io.from_hlr`), the binomial multilevel model (`reliability.fit_mlm`,
needs `[bayes]`), held-out personalization gain (`heldout.heldout_gain`), and the
ADEMP benchmark (`sim.run_ademp`) that reproduces the regression-to-the-mean
inflation of naive one-sided indices and its correction by placebo matching.
The three IO adapters (`from_slam`, `from_hlr`, `from_chess`), CI across Python
3.10–3.12, and a citation file are in place. See `examples/slam_quickstart.py`
for a worked example on real SLAM data.
Remaining (M5+): HLR/chess worked examples, PyPI release, and a Zenodo DOI.
