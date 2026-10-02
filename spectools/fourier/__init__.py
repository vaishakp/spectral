"""Discrete Fourier transforms with fixed conventions.

Modules
-------
transforms
    Nodes, mode numbers, forward and inverse transforms along chosen array
    axes, and evaluation at arbitrary points. The convention
    :math:`f(\\theta_j) = \\sum_k c_k e^{i k \\theta_j}` is fixed here, with
    coefficients kept in NumPy FFT order.
diagnostics
    Modal-power analysis: classify how the coefficients decay and
    recommend a truncation order, ranking modes by :math:`|m|`.

Requires only ``numpy``.
"""
