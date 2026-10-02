"""Spin-weighted spherical harmonics, angular grids and transforms.

Modules
-------
abstract
    :class:`~spectools.spherical.abstract.SphericalGrid`, the abstract base
    fixing the ghost/buffer-zone bookkeeping shared by all grids.
grids
    Concrete grids: :class:`~spectools.spherical.grids.GLGrid`
    (Gauss-Legendre, exact quadrature for band-limited functions, and the
    one to prefer) and :class:`~spectools.spherical.grids.UniformGrid`.
transforms
    Expansion of gridded functions into :math:`{}_sY_{\\ell m}` modes and
    contraction back onto a grid, with optional pole regularization and
    per-mode residual diagnostics.
swsh
    Direct evaluation of :math:`{}_sY_{\\ell m}`, to arbitrary precision.

Harmonic evaluation comes in three precision/speed regimes:

* :mod:`~spectools.spherical.Yslm_full_vec` -- ``numpy``-vectorised over
  grid *and* mode numbers, roughly 11 digits, fastest. Returns every mode
  up to a band limit in one call.
* :mod:`~spectools.spherical.Yslm_mp` -- arbitrary precision, modes
  distributed over a ``multiprocessing`` pool.
* :mod:`~spectools.spherical.Yslm_prec_grid_mp` -- arbitrary precision over
  a whole grid, likewise parallelised.

Requires the ``spherical`` extra: ``pip install spectools[spherical]``.
"""
