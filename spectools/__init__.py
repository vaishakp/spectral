# Copyright (c) 2020, Vaishak Prasad
# See [LICENSE](https://gitlab.com/vaishakp/spectral/-/blob/main/LICENSE) file for details.
"""
Spectral methods for numerical work.

Transforms 1D functions on the real line and 2D functions on the sphere
between their coordinate and spectral representations.

Subpackages
-----------
chebyshev
    Chebyshev collocation bases, nodal/modal transforms, derivative
    matrices and interpolation, plus convergence diagnostics. Requires
    only ``numpy``.
fourier
    Discrete Fourier transforms with fixed sign and normalisation
    conventions, and the matching convergence diagnostics. Requires only
    ``numpy``.
spherical
    Spin-weighted spherical harmonics to arbitrary precision, uniform and
    Gauss-Legendre angular grids, and the transforms between them.
    Requires the ``spherical`` extra.
interpolation
    A 3D spectral interpolator built from angular spin-weighted spherical
    harmonic and radial Chebyshev expansions, parallelised with MPI.
    Requires the ``spherical`` extra and ``mpi4py``.

Notes
-----
Importing :mod:`spectools` itself pulls in nothing beyond the standard
library; the subpackages carry their own dependencies, so a ``numpy``-only
installation can still use :mod:`spectools.chebyshev` and
:mod:`spectools.fourier`.

Examples
--------
>>> import numpy as np
>>> from spectools.chebyshev.chebyshev import ChebyshevSpectral
>>> cheb = ChebyshevSpectral(Nfuncs=32, a=-1.0, b=1.0)
>>> y = np.exp(cheb.collocation_points_physical)
>>> bool(np.allclose(cheb.MatrixD @ y, y, atol=1e-10))
True

"""
import os

package_directory = os.path.dirname(os.path.abspath(__file__))


def read_git_version():
    """Get the latest version number based on the last commit date,
    iff this is a git repo."""

    # print(package_directory)
    # Open the file
    vers = "-1"
    try:
        with open(
            package_directory + "/../public/version", "r", encoding="utf-8"
        ) as vers_file:
            vers = vers_file.readlines()[0]
    except Exception as excep:
        print(
            excep,
            "\n",
            "This is not a git repo! please use the version attribute instead!",
        )
    # with open(package_directory + "/../public/date.txt", "r") as vers_file:
    # vers = vers_file.read()[:10]

    return vers


__version__ = "2026.09.09"
