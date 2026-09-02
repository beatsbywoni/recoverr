# Contributing to recoverr

Thanks for your interest. Bug reports, questions and pull requests are welcome
at https://github.com/beatsbywoni/recoverr/issues.

## Development setup

```bash
git clone https://github.com/beatsbywoni/recoverr.git
cd recoverr
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + pytest / pytest-cov / ruff
pip install -e ".[bayes]"        # optional: NumPyro multilevel model
pytest -q --cov=recoverr         # ~10 s, no external data needed
ruff check src tests
```

The test suite is self-contained (synthetic SLAM tarball, HLR CSV and Lichess
PGN fixtures are generated in `tests/`). Real-data examples in `examples/`
download nothing: pass the path of the public file you have downloaded.

## Reporting a bug

Please include the `recoverr.__version__`, Python / NumPy / pandas versions, a
minimal telemetry frame (or `recoverr.sim.simulate_recovery(...)` call) that
reproduces the problem, and the full traceback.

## Pull requests

- One topic per PR; add or update a test in `tests/`.
- Keep new stochastic steps seedable (`seed=` argument) and deterministic.
- Public functions need a docstring stating the estimand (what is compared
  with what) and the columns of any returned frame.
- Run `pytest -q` and `ruff check` before pushing; CI runs on Python 3.10–3.12.

## Adding a dataset adapter

An adapter in `recoverr/io.py` maps a public source to `Telemetry(unit, seq,
outcome, covariates)`. Document the mapping, keep identifiers pseudonymous
(hash user names), and add a fixture-based test as in `tests/test_io.py` /
`tests/test_revision.py`.

## Code of conduct

Be respectful and constructive; see the Contributor Covenant
(https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
