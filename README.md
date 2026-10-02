[[Project landing page]](https://sites.google.com/view/spectral/home)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/vaishakp/spectral/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/spectools/badge/?version=latest)](https://spectools.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/spectools.svg)](https://pypi.org/project/spectools/)

# spectools

Numerical implementation of various spectral methods.

This library transforms 1D functions on the real line and 2D functions on the
sphere from the coordinate basis to the spectral basis and back. It provides:

1. Fast Fourier transforms with the appropriate conventions, implemented
   using `numpy`.
2. Chebyshev transforms using collocation points, via matrix multiplication
   and direct methods, with derivative matrices and spectral interpolation.
3. Computation of spin-weighted spherical harmonics:
    1. to arbitrary precision at a given coordinate (theta, phi);
    2. to arbitrary precision on a 2D coordinate grid, parallelized using
       the `multiprocessing` library;
    3. to about 11 digits of precision, fast, via `numpy` vectorization;
    4. an API onto `spherepack` through `pyspharm`, with interconversions;
    5. an API onto the `spherical` package, with interconversions.
4. Spin-weighted spherical transforms on uniform and Gauss-Legendre grids.
5. Differentiation of 1D and 2D functions using spectral and finite
   differencing up to 11th order.
6. A 3D spectral interpolator combining angular spherical harmonic and
   radial Chebyshev transforms, parallelized with MPI.
7. Convergence diagnostics that classify modal decay and recommend a
   truncation order, for both Chebyshev and Fourier expansions.

These are the basic tools used in applications such as solutions to simple
elliptic and hyperbolic systems, BMS transformations (implemented in the
sister package `waveformtools`, which is the progenitor), handling and
computation of grid data output from SpEC and other spectral codes, and 3D
visualization.

# Installing

```sh
pip install spectools
```

That gives the **import-light** part of the package — `spectools.chebyshev`
and `spectools.fourier` — which needs only `numpy`. It installs anywhere
numpy does, including Windows.

The spherical-harmonic and interpolation modules need more:

```sh
pip install spectools[spherical]     # spectools.spherical, spectools.interpolation
pip install spectools[docs]          # to build the documentation
```

The parallel interpolation routines additionally require `mpi4py`, which is
not installed by any extra; install it yourself alongside a working MPI.

**Changed in 2026.09:** `waveformtools` and `sxstools` used to be
unconditional requirements, so `pip install spectools` pulled them in even
for a user who only wanted a Chebyshev basis. They pull `lalsuite`, which
publishes no Windows wheel and no source distribution, and `scri` ->
`spinsfast`, which has no Windows wheel and does not build under MSVC — so
the whole package was uninstallable on Windows. They are now the
`[spherical]` extra. `recommonmark` and `sphinx-rtd-theme` were also
runtime requirements and are documentation-build tools; they are now
`[docs]`. `numpy` was never declared at all and now is.

If you use `spectools.spherical` or `spectools.interpolation`, install
`spectools[spherical]`; a bare `pip install spectools` will no longer bring
their dependencies.

## Installing from source

Directly from the git repository:

```sh
pip install git+https://github.com/vaishakp/spectral@main
```

Or from a clone:

```sh
git clone https://github.com/vaishakp/spectral.git
cd spectral
pip install -e .
```

## Conda environment

To create an environment with the dependencies resolved for you, and
activate it:

```sh
conda env create -f docs/environment.yml
conda activate spectools
```

# Quick start

```python
import numpy as np
from spectools.chebyshev.chebyshev import ChebyshevSpectral

cheb = ChebyshevSpectral(Nfuncs=32, a=-1.0, b=1.0)

x = cheb.collocation_points_physical      # 32 Gauss-Lobatto nodes
y = np.exp(np.sin(3 * x))

coeffs = cheb.MatrixPhysToSpec @ y        # nodal -> modal
dydx = cheb.MatrixD @ y                   # spectral derivative
y_fine = cheb.GetOnAxis(y, np.linspace(-1, 1, 201))   # interpolation
```

See the [Usage guide](https://spectools.readthedocs.io/en/latest/usage.html)
for worked examples covering Fourier transforms, spherical harmonic
expansions and the convergence diagnostics.

# Documentation

The documentation is at
[spectools.readthedocs.io](https://spectools.readthedocs.io/en/latest/),
built automatically by Read the Docs from `docs/source`.

To build it locally:

```sh
pip install -e .[docs]
sphinx-build -b html docs/source docs/_build/html
```

Then open `docs/_build/html/index.html`.

# Citing this code

Please cite the latest version of this code if used in your work. This code
was developed for use in the following works:

1. [News from Horizons in Binary Black Hole Mergers](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.121101)
2. [Tidal deformation of dynamical horizons in binary black hole mergers](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.044019)

We request you to also cite these. Thanks!

# Bug tracker

If you run into any issues while using this package, please report them on
the [issue tracker](https://github.com/vaishakp/spectral/issues).

# Acknowledgements

This project is hosted on GitHub, and several GitHub tools are used in the
deployment, testing and version control of the code.

This work was developed in aid of my PhD at the
[Inter-University Centre for Astronomy and Astrophysics (IUCAA, Pune, India)](https://www.iucaa.in/).
The PhD is in part supported by the
[Shyama Prasad Mukherjee Fellowship](https://csirhrdg.res.in/Home/Index/1/Default/2006/59)
awarded to me by the
[Council of Scientific and Industrial Research (CSIR, India)](https://csirhrdg.res.in/).
Resources of IUCAA were used in part.
