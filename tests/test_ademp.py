"""ADEMP benchmark: placebo-matched is ~unbiased under the null, the naive
one-sided index inflates (worse when sparse), and the truth is recovered under
a real effect. Deterministic (seed=1)."""
import recoverr as rc


def _cell(s, shape, depth, density, estimator, col="mean"):
    row = s[(s["shape"] == shape) & (s["true_depth"] == depth)
            & (s["density"] == density) & (s["estimator"] == estimator)]
    return row[col].item()


def test_ademp_null_inflation_and_placebo_unbiased():
    out = rc.sim.run_ademp(n_units=40, events_grid=(4, 12), depth_grid=(0.0, 0.3),
                           shape_grid=("exponential",), n_reps=8, seed=1)
    s = out["summary"]
    # placebo-matched ~ unbiased under the null (both densities)
    assert abs(_cell(s, "exponential", 0.0, "dense", "placebo_matched", "bias")) < 0.02
    assert abs(_cell(s, "exponential", 0.0, "sparse", "placebo_matched", "bias")) < 0.02
    # naive one-sided inflated under the null; worse when sparse
    nd = _cell(s, "exponential", 0.0, "dense", "naive_onesided", "bias")
    ns = _cell(s, "exponential", 0.0, "sparse", "naive_onesided", "bias")
    assert nd > 0 and ns > nd
    # under a real effect, placebo-matched tracks the analytic truth (small bias)
    assert abs(_cell(s, "exponential", 0.3, "dense", "placebo_matched", "bias")) < 0.03
    assert _cell(s, "exponential", 0.3, "dense", "placebo_matched", "mean") > 0.05


def test_ademp_reports_tau_and_shapes():
    out = rc.sim.run_ademp(n_units=30, events_grid=(12,), depth_grid=(0.3,),
                           shape_grid=("exponential", "plateau"), n_reps=3, seed=2)
    tau = out["tau"]
    assert set(tau["shape"]) == {"exponential", "plateau"}
    # exponential shape: identified population fits recover tau near truth (5.0)
    t_exp = tau[tau["shape"] == "exponential"]["tau_hat_median"].item()
    assert 2.5 < t_exp < 10.0
