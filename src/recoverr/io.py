"""Domain adapters: public datasets -> normalized Telemetry (ported in M2).

Each adapter reads a public source and returns a :class:`~recoverr.schema.Telemetry`
with unit/seq/outcome mapped as documented, so the same pipeline runs on all three
validation domains.
"""
from __future__ import annotations

from .schema import Telemetry


def from_slam(path, track: str = "en_es") -> Telemetry:  # pragma: no cover
    """Duolingo SLAM (learning). Map: unit=user, seq=pos_idx, outcome=error.

    Port from c1_01_extract.py in M2.
    """
    raise NotImplementedError("from_slam: port c1_01_extract mapping in M2.")


def from_hlr(path) -> Telemetry:  # pragma: no cover
    """Duolingo HLR (memory). Map: unit=user, seq=timestamp, outcome=recall.

    Port from c2_01_build.py in M2.
    """
    raise NotImplementedError("from_hlr: port c2_01_build mapping in M2.")


def from_chess(path) -> Telemetry:  # pragma: no cover
    """Online chess (performance). Map: unit=player, seq=move/game index, outcome=blunder.

    Use a time window disjoint from tilt studies #13/#16 (data de-dup). M2.
    """
    raise NotImplementedError("from_chess: implement chess adapter in M2.")
