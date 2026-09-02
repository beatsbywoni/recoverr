"""Performance benchmark (reviewer R1/R3: runtime, memory, scaling).

Simulated telemetry with N units x (events x 46 obs) rows; full RecoveryPipeline
(placebo matching, windows, curve/axes with per-unit tau fits, thinned split-half
reliability, 1,000-draw permutation null, balance diagnostics). Peak RSS via
resource.getrusage. One size per call (chunked):

  python 05_benchmark.py <n_rows_target>     -> appends to out/benchmark.csv
  python 05_benchmark.py report              -> prints the table + environment
"""
import json
import os
import platform
import resource
import sys
import time

import numpy as np
import pandas as pd

import recoverr as rc

OUT = os.environ.get("RECOVERR_BENCH_OUT", ".")


def run(n_target):
    epu = 12; per_event = 1 + 20 + 45
    n_units = max(10, int(round(n_target / (epu * per_event))))
    t0 = time.time()
    tele = rc.sim.simulate_recovery(n_units=n_units, events_per_unit=epu, window=20, depth=0.3, seed=1)
    t_sim = time.time() - t0
    rule = lambda g: (g["phase"] == "anchor").to_numpy()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    stages = {}
    t = time.time(); anchors = rc.events.detect_anchors(tele, rule=rule); stages["detect_anchors"] = time.time() - t
    t = time.time(); plc = rc.baseline.match_placebo(tele, anchors, window=20, seed=1, exclusion="both"); stages["match_placebo"] = time.time() - t
    t = time.time(); ev_w = rc.events.extract_windows(tele, anchors, 20, "event"); plc_w = rc.events.extract_windows(tele, plc[["unit", "event_id", "seq"]], 20, "placebo", event_anchors=anchors); stages["extract_windows"] = time.time() - t
    t = time.time(); curve = rc.recovery.recovery_curve(ev_w, plc_w); fit = rc.recovery.fit_parametric(curve); axes = rc.recovery.recovery_axes(ev_w, plc_w, window=20); stages["curve_axes_tau"] = time.time() - t
    t = time.time(); keep = rc.events.thin_anchors(anchors, 20)["event_id"]; rel = rc.reliability.split_half(ev_w[ev_w["event_id"].isin(keep)], plc_w[plc_w["event_id"].isin(keep)], min_events=5); stages["reliability"] = time.time() - t
    t = time.time(); perm = rc.nulls.permutation_null(ev_w, plc_w, n_perm=1000); stages["permutation_1000"] = time.time() - t
    t = time.time(); bal = rc.baseline.balance_diagnostics(tele, anchors, plc, ["phase"]); stages["balance"] = time.time() - t
    t = time.time(); full = rc.RecoveryPipeline(window=20, placebo_exclusion="both", n_perm=1000).run(tele, anchor_rule=rule, min_events=5, seed=1); stages["pipeline_total"] = time.time() - t
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    row = dict(n_rows=len(tele), n_units=n_units, n_events=len(anchors), sim_seconds=round(t_sim, 2),
               peak_rss_mb=round(rss1, 0), **{k: round(v, 2) for k, v in stages.items()})
    print(json.dumps(row))
    path = f"{OUT}/benchmark.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def report():
    df = pd.read_csv(f"{OUT}/benchmark.csv").drop_duplicates("n_rows", keep="last").sort_values("n_rows")
    print(df.to_string(index=False))
    env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
               scipy=__import__("scipy").__version__, platform=platform.platform(), cpu=platform.processor() or os.uname().machine,
               n_cpu=os.cpu_count(), ram_gb=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1))
    print(json.dumps(env, indent=1)); json.dump(env, open(f"{OUT}/benchmark_env.json", "w"), indent=1)


if __name__ == "__main__":
    report() if sys.argv[1] == "report" else run(int(float(sys.argv[1])))
