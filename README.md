# recoverr

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
pip install recoverr           # light: numpy / pandas / scipy
pip install "recoverr[bayes]"  # adds NumPyro + JAX for the multilevel model
```

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

This is an M1 scaffold: the recovery core (events, baseline, recovery, split-half
reliability, permutation null, simulator) is implemented and tested; the Bayesian
multilevel model, domain IO adapters, held-out gain, and the full ADEMP benchmark
are ported next (see `pyproject` extras and module docstrings).
