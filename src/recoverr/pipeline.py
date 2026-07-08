"""End-to-end orchestrator tying the five stages together.

    detect anchors -> match placebo -> extract windows -> curve/axes -> reliability

One call runs the whole within-person recovery pipeline on any normalized
Telemetry, which is the cross-domain reuse claim the package makes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import events as _events
from . import baseline as _baseline
from . import recovery as _recovery
from . import reliability as _reliability
from .schema import Telemetry


@dataclass
class RecoveryPipeline:
    window: int = 20
    depth_span: tuple[int, int] = (1, 5)
    complete_span: tuple[int, int] | None = None
    n_per_event: int = 3
    fit_tau: bool = True

    def run(
        self,
        tele: Telemetry,
        anchor_rule=None,
        match_on: Sequence[str] = (),
        min_events: int = 10,
        seed: int = 0,
    ) -> dict:
        anchors = _events.detect_anchors(tele, rule=anchor_rule)
        placebo_anchors = _baseline.match_placebo(
            tele, anchors, window=self.window,
            n_per_event=self.n_per_event, match_on=match_on, seed=seed,
        )
        ev_w = _events.extract_windows(tele, anchors, self.window, kind="event")
        plc_w = _events.extract_windows(
            tele, placebo_anchors[["unit", "event_id", "seq"]], self.window, kind="placebo",
        )
        curve = _recovery.recovery_curve(ev_w, plc_w)
        axes = _recovery.recovery_axes(
            ev_w, plc_w, window=self.window, depth_span=self.depth_span,
            complete_span=self.complete_span, fit_tau=self.fit_tau,
        )
        rel_depth = _reliability.split_half(
            ev_w, plc_w, axis="depth", span=self.depth_span, min_events=min_events,
        )
        rel_auc = _reliability.split_half(
            ev_w, plc_w, axis="auc", span=(1, self.window), min_events=min_events,
        )
        return {
            "n_events": int(anchors.shape[0]),
            "curve": curve,
            "axes": axes,
            "reliability": {"depth": rel_depth, "auc": rel_auc},
        }
