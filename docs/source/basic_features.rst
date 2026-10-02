**************
Basic Features
**************

``spectools`` transforms 1D functions on :math:`\mathbb{R}` and 2D functions
on :math:`S^2` between their coordinate and spectral representations. The
implementations are cached and memory-efficient, and the parallel paths are
built on ``multiprocessing`` and MPI.

What is provided
================

Fourier transforms
    Fast Fourier transforms with the sign and normalisation conventions
    fixed once, in :mod:`spectools.fourier.transforms`, so you do not have
    to rediscover them. Mode numbers are kept in NumPy FFT order, and
    evaluation off the node set is supported.

Chebyshev transforms
    Collocation-point bases on an arbitrary interval, via matrix
    multiplication and direct methods, with first and second derivative
    matrices and spectral interpolation onto an arbitrary axis. See
    :mod:`spectools.chebyshev.chebyshev`.

Spin-weighted spherical harmonics
    :math:`{}_sY_{\ell m}` computed to arbitrary precision at a given
    coordinate :math:`(\theta, \phi)` or over a whole 2D grid, parallelised
    with ``multiprocessing``; plus a ``numpy``-vectorised path accurate to
    roughly 11 digits. Interoperates with ``spherepack`` (through
    ``pyspharm``) and the ``spherical`` package, with conversions in both
    directions.

Spin-weighted spherical transforms
    Expansion and contraction on uniform and Gauss-Legendre grids, with
    optional pole regularisation and per-mode residual diagnostics. See
    :mod:`spectools.spherical.transforms`.

Grids
    Gauss-Legendre and uniform angular grids, and Gauss-Lobatto collocation
    grids, in :mod:`spectools.spherical.grids`.

3D spectral interpolation
    Angular spin-weighted spherical harmonic expansion combined with a
    radial Chebyshev expansion, parallelised with MPI, in
    :mod:`spectools.interpolation`.

Convergence diagnostics
    Modal-power classification for both Chebyshev and Fourier expansions:
    is the series converging spectrally, algebraically, stalled, at the
    noise floor, or under-resolved, and at what order could it be
    truncated. See :mod:`spectools.chebyshev.diagnostics` and
    :mod:`spectools.fourier.diagnostics`.

Differentiation
    Spectral and finite differencing of 1D and 2D functions, up to
    11th order.

Where it is used
================

These are the building blocks for solving simple elliptic and hyperbolic
systems, BMS transformations (implemented in the sister package
``waveformtools``, from which this package was split out), handling grid
data produced by SpEC and other spectral codes, and 3D visualisation.

Dependencies at a glance
========================

:mod:`spectools.chebyshev` and :mod:`spectools.fourier` need only
``numpy``. :mod:`spectools.spherical` and :mod:`spectools.interpolation`
need the ``spherical`` extra (``pip install spectools[spherical]``), and the
parallel interpolation routines additionally need ``mpi4py``.
