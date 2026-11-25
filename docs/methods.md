# Methods: what recoverr estimates

## 1. Data model

A telemetry table has one row per observation with columns `unit` (person),
`seq` (order within unit) and `outcome` $y_{it}$ (binary failure indicator, or a
continuous outcome). Optional covariates describe the context of each
observation (e.g. exercise format, days since start, game phase, position
quality) and, when relevant, a *stream* id that groups observations into
sub-sequences within a unit (the lexeme in spaced-repetition logs, the game in
chess).

## 2. Failure anchors and windows

An **anchor rule** $A$ maps a unit's ordered frame to a boolean mask; anchors
$a_{ik}$ are the observations flagged for unit $i$ ($k = 1..K_i$). The
**post-failure window** is the next $W$ observations after the anchor (or, in
stream mode, the next $W$ observations of the anchor's stream):

$$\mathcal{W}^{+}_{ik} = \{\,t : \mathrm{pos}(t) - \mathrm{pos}(a_{ik}) \in [1, W]\,\}.$$

The window may contain later anchors; the fraction of such windows is reported
as the **overlap rate** and is *not* used to drop windows (dropping them would
condition on the outcome). For reliability, anchors are **thinned** so that kept
windows do not share observations.

## 3. Baselines

### 3.1 Context-matched placebo baseline (default)

For each anchor, $m$ **placebo anchors** are drawn without replacement from the
same unit's observations that (i) are not anchors, (ii) do not lie inside the
post-failure window of an earlier anchor, (iii) optionally are not within
`exclude_before` observations before an anchor and do not share an *episode*
with an anchor (`exclude_episode`, e.g. the anchor's own exercise), and (iv)
match the anchor on the chosen covariate bins. Placebo windows $\mathcal{W}^{0}$
are then extracted exactly like event windows (same $W$, same stream rule), so
both window types may contain later failures at their natural rates.

Balance is reported as the standardized mean difference between event and
placebo anchors for every matching covariate, together with the number of
anchors that received no placebo.

### 3.2 Pre-event baseline (option)

`baseline="pre_event"` uses the $W$ observations *before* each anchor, mirrored,
after skipping `exclude_before` observations (the span that defines the anchor).
In simulation with stationary autocorrelated risk and no post-failure effect
this baseline is unbiased, whereas any baseline drawn from outside the failure
neighbourhood (placebo, global mean) absorbs the persistence of the risk state
that produced the failure. The two baselines therefore answer different
questions (deviation from matched non-failure moments vs. change relative to the
immediately preceding state) and both are reported in the paper's benchmarks.

## 4. Recovery curve and axes

Let $\bar y^{+}_{i}(r)$ and $\bar y^{0}_{i}(r)$ be unit $i$'s mean outcome at
relative position $r$ over event and baseline windows. The **population
deviation curve** is the equal-weight average of within-unit differences,

$$D(r) = \frac{1}{N}\sum_i \big[\bar y^{+}_{i}(r) - \bar y^{0}_{i}(r)\big],$$

(`weights="unit"`); the pooled curve is kept as `deviation_pooled`. Per-unit axes:

- **depth** $= \frac{1}{|S|}\sum_{r\in S} [\bar y^{+}_{i}(r) - \bar y^{0}_{i}(r)]$ over the early span $S$ (default $r = 1..5$);
- **area index** (nonparametric speed) $=$ the same mean over $r = 1..W$;
- **τ** (parametric speed): least-squares fit of $D_i(r) = D_0 e^{-r/\tau}$ with
  $D_0\in[0,1]$, $\tau\in[0.5, 100]$; a fit is *unidentified* when $\tau$ lands
  on a bound, $D_0 = 0$ or fewer than 6 finite points exist, and the area index
  is then the recommended speed axis;
- **completeness** $=$ mean deviation over the late span (default last 5 positions);
- **level** $=$ mean event-window outcome (the reference the deviation is taken from).

## 5. Inference and reliability

- **Permutation null**: event/baseline labels are permuted *within unit*; the
  test statistic is the unit-mean early deviation (null centred at 0). For
  binary outcomes the per-unit count of failures landing in event slots is
  hypergeometric, which gives the exact permutation distribution without
  shuffling. A one-sided Monte-Carlo p-value $(\#\{D^{*}\ge D\}+1)/(B+1)$ is reported.
- **Split-half reliability**: per-unit depth (or area index) is computed on two
  halves and correlated (Spearman), then Spearman–Brown corrected. Halves are
  odd/even (default) or first/second events after **thinning** to non-overlapping
  windows, each half with its own baseline windows; `split="temporal"` builds the
  two estimates on disjoint time segments per unit (most conservative). A
  percentile bootstrap over units gives 95 % CIs.
- **Multilevel model** (optional, NumPyro): binomial random-effects model of
  event vs baseline counts with unit-level random slopes, giving shrunken
  per-unit effects and the random-slope variance.
- **Held-out gain**: the last fraction of each unit's events is held out; the
  early-window failure probability is predicted by the global rate versus a
  shrunken per-unit rate, and the log-loss / Brier improvement is reported.

## 6. Interpretation and limits

The deviation curve is a *descriptive within-person* quantity: the excess
failure probability after a failure relative to matched moments of the same
person. It is not a causal effect of the failure; when the latent risk state is
persistent, failure clusters mark high-risk periods and the placebo-matched
deviation includes that persistence (quantified in `sim.run_ademp_null`).
Reliability of individual differences rises with the number of non-overlapping
episodes per unit (simulated curves in the paper) and should be reported with
its CI and the split used.

## 7. Details not stated in the paper

The SoftwareX article (Jeong, 2026, https://doi.org/10.1016/j.softx.2026.103056)
describes v1.1.0 but does not give the items below. Values are the v1.1.0 code
defaults unless stated otherwise.

| Topic | v1.1.0 behaviour | Code |
|---|---|---|
| Window length $W$ | `window=20` | `RecoveryPipeline` |
| Early and late spans | depth over `depth_span=(1, 5)`; completeness over the last five positions of the window (`complete_span=None`) | `RecoveryPipeline`, `recovery.recovery_axes` |
| Placebos per anchor $m$ | `n_per_event=3`; the SLAM, HLR and Lichess example scripts all use this default | `RecoveryPipeline`, `baseline.match_placebo` |
| Anchor-defining span | `exclude_before=0` and `exclude_episode=None` by default; the examples bar the anchor's own episode with `exclude_episode` (SLAM `"ex_id"`, HLR `"session"`) and the Lichess example follows the game with `stream="game_id"` | `baseline.match_placebo` |
| Covariate matching | exact on the values of the `match_on` columns; observations with a missing value are not used as placebos; continuous covariates must be binned beforehand. Bins built by the adapters: SLAM `days_bin` (edges 0, 7, 14, 30 days); HLR `hist_seen_bin` (1–3, 4–9, 10–29, 30+ prior exposures) and `lex_diff_bin` (quartiles of the lexeme's corpus-wide miss rate); Lichess `phase` (opening up to move 10, middlegame, endgame after move 30) and `wp_before_bin` / `wp_after_bin` (win probability 0–.25, .25–.5, .5–.75, .75–1) | `io.from_slam`, `io.from_hlr`, `io.from_lichess_moves` |
| Matching on recent failure rate | used only in the simulation in the paper, but available to users: add a binned covariate with `sim.recent_rate_bins` and pass it in `match_on` | `sim.recent_rate_bins` |
| Multilevel model | logit P(y = 1) = b0 + u_i + event × (b_ev + v_i); b0, b_ev ~ Normal(0, 2); random-effect SDs ~ HalfNormal(1); non-centred random effects; NUTS (target acceptance .95), 2 chains of 1000 warm-up and 1000 draws; R-hat reported. Not evaluated in the paper's benchmarks | `reliability.fit_mlm` (`[bayes]` extra) |
| Held-out shrinkage | per-unit early-window rate shrunk toward the global rate with weight n_i / (n_i + 8), n_i = the unit's early-window observations in the training part (`shrink_k=8`) | `heldout.heldout_gain` |
| Anchor rule of the usage example | `error_cluster_rule(k)`: last token of an exercise with at least k token errors (k = 2 in the SLAM example, 1 to 3 in the sensitivity analysis); defined in `examples/slam_quickstart.py`, not in the package | `examples/slam_quickstart.py` |
| Lichess failure and outcome | anchor = blunder (Lichess annotation `??`, or a win-probability loss of at least .30 when annotations are absent); outcome = mistake or blunder (`?` or `??`, or a loss of at least `error_threshold=0.10`); win probability = 1 / (1 + exp(−0.00368208 × cp)) with centipawns clipped to ±1000 | `io.from_lichess_moves`, `io._win_prob` |
| Dependencies | `numpy>=1.24`, `pandas>=2.0`, `scipy>=1.10`, `pyarrow>=14` (Parquet input in the adapters); optional extras `bayes` (`numpyro>=0.13`, `jax>=0.4`, multilevel model) and `chess` (`zstandard>=0.21`, streaming `.pgn.zst` files) | `pyproject.toml` |
| Module count | the article says "seven modules"; the package has nine: `schema`, `events`, `baseline`, `recovery`, `reliability`, `nulls`, `heldout`, `sim`, `io`, plus the `RecoveryPipeline` orchestrator | `src/recoverr/` |
