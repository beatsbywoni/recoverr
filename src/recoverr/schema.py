"""Normalized input schema for within-person behavioral telemetry.

Every supported domain (language-learning tokens, spaced-repetition reviews,
chess moves, ...) reduces to the *same* tidy long table once normalized, which
is what lets one pipeline serve all of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

REQUIRED = ("unit", "seq", "outcome")


@dataclass
class Telemetry:
    """A normalized long table of within-person observations.

    Standardized columns:
      unit    : within-person grouping id (learner / player / ...)
      seq     : monotonic order within unit (integer index or timestamp)
      outcome : numeric outcome per observation (1 = failure/error, 0 = success;
                or continuous, e.g. log response time)
    plus any optional covariate columns used for baseline matching / phase binning.

    Rows are sorted by (unit, seq).
    """

    frame: pd.DataFrame
    covariates: tuple[str, ...] = ()

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        unit: str = "unit",
        seq: str = "seq",
        outcome: str = "outcome",
        covariates: Sequence[str] = (),
    ) -> "Telemetry":
        covariates = tuple(covariates)
        missing = [c for c in (unit, seq, outcome, *covariates) if c not in df.columns]
        if missing:
            raise ValueError(f"Telemetry.from_frame: missing columns {missing}")
        renamed = df.rename(columns={unit: "unit", seq: "seq", outcome: "outcome"})
        keep = ["unit", "seq", "outcome", *covariates]
        out = renamed[keep].copy()
        out["outcome"] = pd.to_numeric(out["outcome"], errors="coerce")
        out = out.sort_values(["unit", "seq"]).reset_index(drop=True)
        return cls(out, covariates)

    @property
    def n_units(self) -> int:
        return int(self.frame["unit"].nunique())

    def __len__(self) -> int:
        return len(self.frame)

    def __repr__(self) -> str:
        return (
            f"Telemetry(rows={len(self)}, units={self.n_units}, "
            f"covariates={list(self.covariates)})"
        )
