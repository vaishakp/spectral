[[Project landing page]](https://sites.google.com/view/spectral/home)
[![pipeline status](https://github.com/vaishakp/spectral/badges/main/pipeline.svg)](https://github.com/vaishakp/spectral/commits/main)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/vaishakp/spectral/commits/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/spectral/badge/?version=latest)](https://spectral.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](docs/vers_badge.svg)](https://pypi.org/project/spectools/)
# spectral


This is a collection of various spectral methods for numerical appraches.


* Available Methods:

    * Chebyshev 

    * Fourier

    * Sphercical Harmonics

    * Spin weighted spherical harmonics
   

# Installing

```
pip install spectools
```

That gives the **import-light** part of the package -- `spectools.chebyshev`
and `spectools.fourier` -- which needs only `numpy`. It installs anywhere
numpy does, including Windows.

The spherical-harmonic and interpolation modules need more:

```
pip install spectools[spherical]     # spectools.spherical, spectools.interpolation
pip install spectools[docs]          # to build the documentation
```

**Changed in 2026.09:** `waveformtools` and `sxstools` used to be
unconditional requirements, so `pip install spectools` pulled them in even
for a user who only wanted a Chebyshev basis. They pull `lalsuite`, which
publishes no Windows wheel and no source distribution, and `scri` ->
`spinsfast`, which has no Windows wheel and does not build under MSVC -- so
the whole package was uninstallable on Windows. They are now the
`[spherical]` extra. `recommonmark` and `sphinx-rtd-theme` were also
runtime requirements and are documentation-build tools; they are now
`[docs]`. `numpy` was never declared at all and now is.

If you use `spectools.spherical` or `spectools.interpolation`, install
`spectools[spherical]`; a bare `pip install spectools` will no longer bring
their dependencies.


# Citing this code

Please cite the latest version of this code if used in your work. This code was developed for use in the following works:

1. [News from Horizons in Binary Black Hole Mergers](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.121101)
2. [Tidal deformation of dynamical horizons in binary black hole mergers](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.044019)

We request you to also cite these. Thanks!





# Installing this package

## Dependencies

This module has the following dependencies:

* Standard packages (come with full anaconda installation)
    * [`numpy`](http://www.numpy.org/)
    * [`scipy`](http://scipy.org/)
    * [`statistics`](https://docs.python.org/3/library/statistics.html)
    * [`matplotlib`](http://matplotlib.org/)
    * [`h5py`](http://www.h5py.org/)
    * [`termcolor`](https://pypi.org/project/termcolor/)
    * 
* Optional dependencies (labelled [EXT])
    * For use with PyCBC data analysis packages.
        * [`pyCBC`](https://pycbc.org/)
        * [`lalsuite`](https://git.ligo.org/lscsoft/lalsuite)
        * [`ligo-common`](https://git.ligo.org/lscsoft/ligo-common)
    * [`gmpy2`](https://gmpy2.readthedocs.io/en/latest/)


## Recommended method

I recommend installing this module through pypi:
```sh
pip install spectral
```
## Alternate method

Manual install directly from the git repo:

```pip install git+https://github.com/vaishakp/spectral@main```

Or from a clone:

* First, clone this repository:

```sh
git clone https://github.com/vaishakp/spectral.git

```
* Second, run python setup from the `spectral` directory:
```sh
cd spectral
python setup.py install --prefix="<path to your preferred installation dir>"
``` 

## Manually setup conda environment

* To create an environment with automatic dependency resolution and activate it, run
```sh
conda create env -f docs/environment.yml
conda activate wftools
```


# Using this code
```
# Documentation

The documentation for this module is available at [Link to the Documentation](https://spectral.readthedocs.io/en/latest/). This was built automatically using Read the Docs.

In some case where the repo has run out of github CI minutes, the documentation is not automatically built. In such cases, we request the user to access the documentation through the `index.html` file in `docs` directory.


# Bug tracker
If you run into any issues while using this package, please report the issue on the [issue tracker](https://github.com/vaishakp/spectral/-/issues).

 
# Acknowledgements

This project has been hosted, as you can see, on github. Several github tools are used in the deployment of the code, its testing, version control.

The work of this was developed in aiding my PhD work at Inter-University Centre for Astronomy and Astrophysics (IUCAA, Pune, India)](https://www.iucaa.in/). The PhD is in part supported by the [Shyama Prasad Nukherjee Fellowship](https://csirhrdg.res.in/Home/Index/1/Default/2006/59) awarded to me by the [Council of Scientific and Industrial Research (CSIR, India)](https://csirhrdg.res.in/). Resources of the [Inter-University Centre for Astronomy and Astrophysics (IUCAA, Pune, India)](https://www.iucaa.in/) were are used in part.
