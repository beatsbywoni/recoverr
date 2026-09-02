"""End-to-end orchestrator tying the stages together.

    detect anchors -> match placebo -> extract windows -> curve/axes -> reliability
    (+ optional permutation null, held-out gain, balance diagnostics, parametric fit)

``RecoveryPipeline.run`` is a convenience wrapper; every stage is also exposed
as an independent function (see :mod:`recoverr.events`, :mod:`recoverr.baseline`,
:mod:`recoverr.recovery`, :mod:`recoverr.reliability`, :mod:`recoverr.nulls`,
:mod:`recoverr.heldout`) so users can feed individual steps into their own
workflows.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import baseline as _baseline
from . import events as _events
from . import heldout as _heldout
from . import nulls as _nulls
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
    exclude_overlapping: bool = False   # drop event windows that contain a later failure anchor
    n_boot: int = 0                     # bootstrap replicates for reliability CIs (0 = off)
    n_perm: int = 0                     # permutation replicates for the null test (0 = off)
    heldout_tail: float = 0.0           # >0 enables personalized-vs-global held-out gain
    placebo_exclusion: str = "post"     # 'post' (symmetric, default) or 'both' (legacy, see baseline.match_placebo)
    exclude_before: int = 0             # observations before each anchor barred from placebo use (anchor-defining span)
    exclude_episode: str | None = None  # column naming the episode of each observation (e.g. 'ex_id'); anchors' episodes barred
    stream: str | None = None           # windows follow the anchor's sub-stream (e.g. the same lexeme); see events.extract_windows
    baseline: str = "placebo"           # 'placebo' (context-matched placebo windows) or 'pre_event' (mirrored pre-anchor windows, skipping exclude_before)
    split: str = "odd_even"             # reliability split: 'odd_even' / 'first_second' (thinned, non-overlapping events) or 'temporal' (disjoint time segments)
    thin_reliability: bool = True       # thin anchors to non-overlapping windows before the odd/even or first/second split

    def run(
        self,
        tele: Telemetry,
        anchor_rule=None,
        match_on: Sequence[str] = (),
        min_events: int = 10,
        seed: int = 0,
    ) -> dict:
        if self.baseline not in ("placebo", "pre_event"):
            raise ValueError("baseline must be 'placebo' or 'pre_event'")
        anchors = _events.detect_anchors(tele, rule=anchor_rule)
        ev_w = _events.extract_windows(tele, anchors, self.window, kind="event", stream=self.stream)
        if self.baseline == "placebo":
            placebo_anchors = _baseline.match_placebo(
                tele, anchors, window=self.window,
                n_per_event=self.n_per_event, match_on=match_on, seed=seed,
                exclusion=self.placebo_exclusion, exclude_before=self.exclude_before,
                exclude_episode=self.exclude_episode, stream=self.stream,
            )
            plc_w = _events.extract_windows(
                tele, placebo_anchors[["unit", "event_id", "seq"]], self.window, kind="placebo",
                event_anchors=anchors, stream=self.stream,
            )
        else:
            placebo_anchors = anchors[["unit", "event_id", "seq"]]
            plc_w = _events.extract_windows(
                tele, anchors, self.window, kind="pre_event", event_anchors=anchors,
                stream=self.stream, direction="pre", gap=self.exclude_before,
            )
        overlap_rate = float(ev_w.drop_duplicates("event_id")["overlap"].mean()) if len(ev_w) else 0.0
        # several placebo windows share an event_id; count each window once via rel_pos == 1
        overlap_rate_placebo = float(plc_w[plc_w["rel_pos"] == 1]["overlap"].mean()) if len(plc_w) else 0.0
        if self.exclude_overlapping:
            # symmetric: drop event windows AND placebo windows that contain a later failure
            keep = ev_w[~ev_w["overlap"]]["event_id"].unique()
            ev_w = ev_w[ev_w["event_id"].isin(keep)]
            plc_w = plc_w[plc_w["event_id"].isin(keep) & ~plc_w["overlap"]]

        curve = _recovery.recovery_curve(ev_w, plc_w, weights="unit")
        curve["deviation_pooled"] = _recovery.recovery_curve(ev_w, plc_w, weights="pooled").set_index("rel_pos")["deviation"].reindex(curve["rel_pos"]).to_numpy()
        fit = _recovery.fit_parametric(curve) if self.fit_tau else None
        axes = _recovery.recovery_axes(
            ev_w, plc_w, window=self.window, depth_span=self.depth_span,
            complete_span=self.complete_span, fit_tau=self.fit_tau,
        )
        if self.split == "temporal":
            kw = dict(exclusion=self.placebo_exclusion, exclude_episode=self.exclude_episode, stream=self.stream)
            rel_depth = _reliability.temporal_split_half(
                tele, anchor_rule, window=self.window, span=self.depth_span, axis="depth", min_events=min_events,
                n_per_event=self.n_per_event, match_on=match_on, seed=seed, n_boot=self.n_boot,
                baseline=self.baseline, exclude_before=self.exclude_before, **kw)
            rel_auc = _reliability.temporal_split_half(
                tele, anchor_rule, window=self.window, span=(1, self.window), axis="area_index", min_events=min_events,
                n_per_event=self.n_per_event, match_on=match_on, seed=seed, n_boot=self.n_boot,
                baseline=self.baseline, exclude_before=self.exclude_before, **kw)
        else:
            ev_r, plc_r = ev_w, plc_w
            if self.thin_reliability:
                keep_ids = _events.thin_anchors(anchors, self.window)["event_id"]
                ev_r = ev_w[ev_w["event_id"].isin(keep_ids)]
                plc_r = plc_w[plc_w["event_id"].isin(keep_ids)]
            rel_depth = _reliability.split_half(ev_r, plc_r, axis="depth", span=self.depth_span, min_events=min_events,
                                                n_boot=self.n_boot, seed=seed, method_split=self.split)
            rel_auc = _reliability.split_half(ev_r, plc_r, axis="area_index", span=(1, self.window), min_events=min_events,
                                              n_boot=self.n_boot, seed=seed, method_split=self.split)
            for r in (rel_depth, rel_auc):
                r["thinned"] = bool(self.thin_reliability)
        out = {
            "n_events": int(anchors.shape[0]),
            "n_events_used": int(ev_w["event_id"].nunique()) if len(ev_w) else 0,
            "overlap_rate": overlap_rate,
            "overlap_rate_placebo": overlap_rate_placebo,
            "balance": _baseline.balance_diagnostics(tele, anchors, placebo_anchors, list(match_on)),
            "curve": curve,
            "fit": fit,
            "axes": axes,
            "tau_unidentified_rate": float(axes["tau"].isna().mean()) if len(axes) else float("nan"),
            "n_events_thinned": int(_events.thin_anchors(anchors, self.window).shape[0]) if len(anchors) else 0,
            "reliability": {"depth": rel_depth, "area_index": rel_auc},
        }
        if self.n_perm:
            out["permutation"] = _nulls.permutation_null(ev_w, plc_w, span=self.depth_span,
                                                         n_perm=self.n_perm, seed=seed)
        if self.heldout_tail:
            out["heldout"] = _heldout.heldout_gain(ev_w, span=self.depth_span, tail_frac=self.heldout_tail)
        return out
