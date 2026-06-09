# Contributing

Thanks for your interest in BC-QACL-ATD!

## Development setup
```bash
git clone https://github.com/passiondong/BC-QACL-ATD.git bcqacl-atd && cd bcqacl-atd
python -m venv .venv && . .venv/Scripts/activate     # bin/activate on macOS/Linux
pip install -e ".[gui,dev]"
pytest tests/ -q
```

## Repository layout
- `src/bcqacl_atd/` — the **clean, public** layer (edit here):
  `config`, `lb_law`, `model`, `flow`, `cli`, `data_io`, `recalibrate`,
  `app/wizard`.
- `src/bcqacl_atd/kernels/` — **vendored, numerically exact** research kernels
  (SG-CL field solver, six-port assembler, the exact CMA-ES Pareto flow).
  **Do not hand-edit these** except to de-hardcode a path; they are kept verbatim
  so results stay bit-faithful to the paper. Improvements belong in the clean
  layer.
- `tests/` — fast, kernel-free unit tests for the clean core.

## Guidelines
- Keep the public API small and documented; new behavior should be config-driven.
- Add a unit test for any clean-layer change (kernel-free where possible).
- Match the existing style (type hints, short docstrings, no hard-coded paths).
- Run `pytest tests/` before opening a PR; CI runs it on Python 3.10–3.12.

## Releasing to PyPI (maintainers)
```bash
python -m pip install --upgrade build twine
python -m build                  # builds sdist + wheel into dist/
python -m twine check dist/*
python -m twine upload dist/*    # requires a PyPI API token
```
Bump `version` in `pyproject.toml` and `__version__` in
`src/bcqacl_atd/__init__.py` first, tag the release (`git tag vX.Y.Z`), and update
the DOI/citation in `CITATION.cff` and `README.md` once the paper is published.

## Reporting issues
Please include: OS, Python version, the config you ran, and the full traceback.
