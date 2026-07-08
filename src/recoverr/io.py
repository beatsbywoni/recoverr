"""Domain adapters: public datasets -> normalized Telemetry.

Each adapter reads a public source and returns a :class:`~recoverr.schema.Telemetry`
with unit/seq/outcome mapped as documented, so the same pipeline runs on all
validation domains. Ported from C1 (`c1_01_extract`) and C2 (`c2_01_build`).
"""
from __future__ import annotations

import io as _io
import os
import tarfile

import numpy as np
import pandas as pd

from .schema import Telemetry

SLAM_STAMP = "20190204"  # SLAM v4.0 train file stamp
SLAM_DAYS_BINS = (0.0, 7.0, 14.0, 30.0)


def from_slam(
    path,
    track: str = "en_es",
    stamp: str = SLAM_STAMP,
    max_users: int | None = None,
    days_bins=SLAM_DAYS_BINS,
) -> Telemetry:
    """Duolingo SLAM ``data_{track}.tar.gz`` -> token-level Telemetry.

    Mapping: unit=user, seq=pos_idx (per-user stream order), outcome=error
    (token marked incorrect). Covariates: ex_id, days_bin, format, token_freq.
    ``max_users`` streams only the first N learners (fast tests / smoke).
    """
    tok_rows, ex_meta = [], []
    ex_id = -1
    user = days = fmt = None
    seen: set[str] = set()
    skip = False
    with tarfile.open(str(path), "r:gz") as tf:
        # exact basename match avoids macOS AppleDouble "._" sidecar members
        member = next(m for m in tf.getmembers()
                      if os.path.basename(m.name) == f"{track}.slam.{stamp}.train")
        f = _io.TextIOWrapper(tf.extractfile(member), encoding="utf-8", errors="replace")
        for line in f:
            if line.startswith("# user:"):
                ex_id += 1
                meta = dict(p.split(":", 1) for p in line[2:].split() if ":" in p)
                user = meta["user"]
                if max_users is not None and user not in seen and len(seen) >= max_users:
                    skip = True
                    continue
                skip = False
                seen.add(user)
                days = float(meta.get("days", "nan"))
                fmt = meta.get("format", "")
                ex_meta.append((ex_id, user, days, fmt))
            elif line.startswith("#") or skip:
                continue
            else:
                cols = line.split()
                if len(cols) >= 3 and cols[-1] in ("0", "1"):
                    tok_rows.append((ex_id, user, cols[1].lower(), int(cols[-1])))

    tok = pd.DataFrame(tok_rows, columns=["ex_id", "user", "token", "error"])
    ex = pd.DataFrame(ex_meta, columns=["ex_id", "user", "days", "format"])
    tok["pos_idx"] = tok.groupby("user").cumcount()
    tok["token_freq"] = tok["token"].map(tok["token"].value_counts())
    tok = tok.merge(ex[["ex_id", "days", "format"]], on="ex_id", how="left")
    tok["days_bin"] = pd.cut(tok["days"], list(days_bins), labels=False, include_lowest=True)
    return Telemetry.from_frame(
        tok, unit="user", seq="pos_idx", outcome="error",
        covariates=["ex_id", "days_bin", "format", "token_freq"],
    )


def from_hlr(path, binarize: bool = False) -> Telemetry:
    """Duolingo HLR sessions parquet -> Telemetry.

    Mapping: unit=user, seq=ts, outcome=session miss rate ``1 - correct/seen``
    (or a 0/1 miss indicator if ``binarize=True``). Covariates: lexeme, dt_days,
    pos, seq_k. The failure-anchor / recovery-window definition is an
    analysis-time choice; this adapter only normalizes the stream.
    """
    df = pd.read_parquet(path).copy()
    acc = (df["correct"] / df["seen"]).clip(0, 1)
    df["miss"] = 1.0 - acc
    outcome = "miss"
    if binarize:
        df["miss_bin"] = (df["miss"] > 0).astype(int)
        outcome = "miss_bin"
    cov = [c for c in ["lexeme", "dt_days", "pos", "seq_k"] if c in df.columns]
    return Telemetry.from_frame(df, unit="user", seq="ts", outcome=outcome, covariates=cov)


def from_chess(path) -> Telemetry:  # pragma: no cover
    """Online chess (performance). Map: unit=player, seq=move index, outcome=blunder.

    Use a time window disjoint from tilt studies #13/#16 (data de-dup).
    """
    raise NotImplementedError("from_chess: implement chess adapter (data de-dup vs #13/#16).")
