"""Chebyshev collocation spectral methods.

Modules
-------
chebyshev
    :class:`~spectools.chebyshev.chebyshev.ChebyshevSpectral`, the main
    entry point: collocation points on an interval, nodal/modal transform
    matrices, derivative matrices and spectral interpolation.
basis
    The underlying Chebyshev polynomial basis in double precision, and
    helpers for applying a transform matrix along chosen array axes.
basis_precise
    The same basis evaluated symbolically with ``sympy``, to arbitrary
    precision.
diagnostics
    Modal-power analysis: classify how the coefficients decay and
    recommend a truncation order.

Requires only ``numpy`` (``sympy`` for :mod:`~spectools.chebyshev.basis_precise`).
"""
