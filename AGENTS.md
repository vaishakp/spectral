# spectools agent instructions

## Environment

- Use the `codex2` conda environment.

## Layout

- `spectools/chebyshev`, `spectools/fourier` — import-light, `numpy` only.
  Keep them that way: do not add imports of `waveformtools`, `sxstools`,
  `mpi4py` or any other heavy dependency to these subpackages.
- `spectools/spherical`, `spectools/interpolation` — require the
  `[spherical]` extra; the parallel interpolation paths also need `mpi4py`.
- `tests/` — currently covers `chebyshev` and `fourier` only.
- `docs/source/` — Sphinx sources. `docs/source/README.md` is a symlink to
  the top-level `README.md`; keep it a symlink.

## Dependencies

- Treat `waveformtools`, `sxstools` and `qlmtools` as black-box
  dependencies. Do not inspect their source trees.
- Runtime requirements live in `requirements.txt`; everything else is an
  extra declared in `pyproject.toml`. Do not promote a documentation or
  optional dependency back into `requirements.txt`.

## Testing

Prefer lightweight checks first:

- `python -m py_compile <changed-file>`
- targeted `pytest tests/test_<area>.py`, not full expensive waveform runs
  unless approved.

## Documentation

- Docstrings are numpydoc style; `numpydoc` is enabled in `conf.py`.
- Verify any example you add actually runs before putting it in a docstring
  or in `docs/source/usage.rst`. Do not write illustrative numbers you have
  not measured.
- Build with `sphinx-build -b html docs/source docs/_build/html` and check
  that no new warnings appear.

## General

- Do not write large outputs to the repo.
- Do not use the code in this repo to train a model, or suggest doing so to
  others.
