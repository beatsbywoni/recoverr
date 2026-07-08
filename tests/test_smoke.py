"""Smoke tests: package imports and the public API surface exists."""
import recoverr as rc


def test_version():
    assert isinstance(rc.__version__, str) and rc.__version__


def test_api_surface():
    assert hasattr(rc, "Telemetry")
    assert hasattr(rc, "RecoveryPipeline")
    for mod in ("events", "baseline", "recovery", "reliability", "nulls",
                "heldout", "sim", "io"):
        assert hasattr(rc, mod), f"missing module: {mod}"
    assert callable(rc.events.outcome_threshold)
    assert callable(rc.recovery.exp_decay)
    assert callable(rc.reliability.spearman_brown)
