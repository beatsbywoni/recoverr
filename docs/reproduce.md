# Reproducing the paper's examples

All three worked examples use public data that you download yourself (nothing
is bundled); every script prints the exact figures reported in the paper for
the stated seed.

| domain | data (licence) | command |
|---|---|---|
| learning — Duolingo SLAM | https://doi.org/10.7910/DVN/8SWHNO (CC BY-NC 4.0), file `data_en_es.tar.gz` | `python examples/slam_quickstart.py data_en_es.tar.gz --max-users 1000` |
| memory — Duolingo HLR | https://doi.org/10.7910/DVN/N8XJME (CC BY-NC 4.0), file `settles.acl16.learning_traces.13m.csv.gz` | `python examples/hlr_quickstart.py settles.acl16.learning_traces.13m.csv.gz --max-users 300` |
| performance — Lichess | https://database.lichess.org/standard/lichess_db_standard_rated_2014-07.pgn.zst (CC0) | `python examples/chess_quickstart.py lichess_db_standard_rated_2014-07.pgn.zst --max-games 175033 --max-players 200` |

Simulation benchmarks (`sim.run_ademp`, `sim.run_ademp_null`, reliability-vs-
episodes) need no data:

```python
import recoverr as rc
out = rc.sim.run_ademp(n_units=40, n_reps=50, seed=20260901)     # ~90 s per shape
out["summary"]; out["tau"]
null = rc.sim.run_ademp_null(n_units=40, n_obs=600, n_reps=50, seed=1)
null["summary"]
```

The full revision workflow (chunked scripts, seeds, outputs) is archived with
the paper's supplementary material.
