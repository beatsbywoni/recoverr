"""M3: ADEMP benchmark reproduces the one-sided-index inflation under the null.

Deterministic (seed=1): placebo-matched is ~unbiased under the null while the
naive one-sided index is inflated, and the inflation worsens with sparser
episodes. Under a real effect the naive index also under-estimates.
"""
import recoverr as rc


def test_ademp_null_inflation_and_placebo_unbiased():
    out = rc.sim.run_ademp(n_units=40, events_per_unit=12, n_reps=10, seed=1)
    s = out["summary"]

    def cell(depth, density, method):
        row = s[(s["true_depth"] == depth) & (s["density"] == density)
                & (s["method"] == method)]
        return row["mean"].item()

    # Placebo-matched ~ unbiased under the null, both densities.
    assert abs(cell(0.0, "dense", "placebo_matched")) < 0.02
    assert abs(cell(0.0, "sparse", "placebo_matched")) < 0.02

    # Naive one-sided index inflated under the null, worse when episodes are sparse.
    naive_dense = cell(0.0, "dense", "naive_onesided")
    naive_sparse = cell(0.0, "sparse", "naive_onesided")
    assert naive_dense > cell(0.0, "dense", "placebo_matched")
    assert naive_sparse > naive_dense
    assert naive_sparse - cell(0.0, "sparse", "placebo_matched") > 0.01

    # Under a real effect, placebo-matched detects it.
    assert cell(0.3, "dense", "placebo_matched") > 0.05
