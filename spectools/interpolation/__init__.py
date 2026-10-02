"""3D spectral interpolation on Gauss-Legendre x Chebyshev grids.

Interpolates a field sampled on concentric spherical shells onto arbitrary
points. Each shell is expanded in spin-weighted spherical harmonics, and
each resulting mode coefficient :math:`C_{\\ell m}(r)` is then expanded
radially in Chebyshev polynomials. Both stages are distributed over MPI
ranks.

Five variants of the same algorithm are provided, differing along two
axes -- whether a time axis is carried, and whether the per-mode loops are
vectorized:

.. list-table::
   :header-rows: 1

   * - Module
     - Time axis
     - Vectorized
   * - :mod:`~spectools.interpolation.interpolate`
     - yes
     - no
   * - :mod:`~spectools.interpolation.interpolate_new`
     - yes
     - no
   * - :mod:`~spectools.interpolation.interpolate_new_vectorized`
     - yes
     - yes
   * - :mod:`~spectools.interpolation.interpolate_single_t`
     - no
     - no
   * - :mod:`~spectools.interpolation.interpolate_single_t_vectorized`
     - no
     - yes

Each exposes a class named ``Interpolate3D``.

Which to use
------------
Prefer :mod:`~spectools.interpolation.interpolate_new_vectorized` for a
time series, or
:mod:`~spectools.interpolation.interpolate_single_t_vectorized` for a
single snapshot. The vectorized forms are faster but hold more of the
decomposition in memory; drop to the non-vectorized equivalent if that
becomes a constraint.

:mod:`~spectools.interpolation.interpolate` is the original
implementation, kept for reference. It sources its Chebyshev and spherical
machinery from ``qlmtools`` and ``waveformtools`` rather than from
:mod:`spectools`, so it does not track changes made here.

Requires the ``spherical`` extra and ``mpi4py``; several variants also
import ``qlmtools``, which is not declared as a dependency and must be
installed separately. A worked example is in
``support/examples/spectral_interpolator.ipynb``.
"""
