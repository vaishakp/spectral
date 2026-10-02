***********
Package API
***********

Reference documentation for every public module in :mod:`spectools`, grouped
by subpackage.

Subpackages
===========

.. autosummary::
   :toctree: _autosummary

   spectools
   spectools.chebyshev
   spectools.fourier
   spectools.spherical
   spectools.interpolation
   spectools.helper

Chebyshev
=========

Collocation-based Chebyshev bases, transforms and derivatives. Depends only
on ``numpy``.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   spectools.chebyshev.chebyshev
   spectools.chebyshev.basis
   spectools.chebyshev.basis_precise
   spectools.chebyshev.diagnostics

Fourier
=======

Discrete Fourier transforms with fixed sign and normalisation conventions,
plus convergence diagnostics. Depends only on ``numpy``.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   spectools.fourier.transforms
   spectools.fourier.diagnostics

Spherical
=========

Spin-weighted spherical harmonics, angular grids and the transforms between
them. Requires the ``spherical`` extra.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   spectools.spherical.abstract
   spectools.spherical.grids
   spectools.spherical.swsh
   spectools.spherical.transforms
   spectools.spherical.Yslm_mp
   spectools.spherical.Yslm_full_vec
   spectools.spherical.Yslm_prec_grid_mp

Interpolation
=============

3D spectral interpolation built from angular spin-weighted spherical
harmonic and radial Chebyshev expansions. Requires the ``spherical`` extra,
and MPI for the parallel variants.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   spectools.interpolation.interpolate
   spectools.interpolation.interpolate_new
   spectools.interpolation.interpolate_new_vectorized
   spectools.interpolation.interpolate_single_t
   spectools.interpolation.interpolate_single_t_vectorized

Helpers
=======

.. autosummary::
   :toctree: _autosummary
   :recursive:

   spectools.helper.factorial
