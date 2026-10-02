*****
Usage
*****

Worked examples for each subpackage. Everything in the Chebyshev and
Fourier sections runs against a bare ``pip install spectools``; the
spherical section needs ``pip install spectools[spherical]``.

.. contents::
   :local:
   :depth: 2


Chebyshev
=========

Collocation points and transforms
---------------------------------

:class:`~spectools.chebyshev.chebyshev.ChebyshevSpectral` holds a Chebyshev
collocation basis on a physical interval ``[a, b]``. Sample your function at
:attr:`~spectools.chebyshev.chebyshev.ChebyshevSpectral.collocation_points_physical`
-- the transform matrices assume exactly those points.

.. code-block:: python

   import numpy as np
   from spectools.chebyshev.chebyshev import ChebyshevSpectral

   cheb = ChebyshevSpectral(Nfuncs=32, a=-1.0, b=1.0)

   x = cheb.collocation_points_physical   # 32 Gauss-Lobatto nodes, x[0] == -1.0
   y = np.exp(np.sin(3 * x))

   coeffs = cheb.MatrixPhysToSpec @ y     # nodal  -> modal
   values = cheb.MatrixSpecToPhys @ coeffs  # modal -> nodal, recovers y

The two matrices are inverses of one another, so ``values`` matches ``y`` to
machine precision.

Derivatives
-----------

:attr:`~spectools.chebyshev.chebyshev.ChebyshevSpectral.MatrixD` maps nodal
values straight to the nodal values of the derivative, with the
``2 / (b - a)`` interval scaling already applied.

.. code-block:: python

   dydx = cheb.MatrixD @ y
   exact = 3 * np.cos(3 * x) * np.exp(np.sin(3 * x))
   np.max(np.abs(dydx - exact))   # ~2e-08 at Nfuncs=32

:attr:`~spectools.chebyshev.chebyshev.ChebyshevSpectral.MatrixDD` is the
second derivative, with the same interval scaling applied:

.. code-block:: python

   d2ydx2 = cheb.MatrixDD @ y

.. versionchanged:: 2026.10.02

   ``MatrixDD`` previously carried a spurious ``((b - a) / 2) ** 2``
   factor, making it correct only on an interval of width 2. Code that
   divided the result by that factor to compensate must stop doing so.

Interpolation onto another axis
-------------------------------

``GetOnAxis`` evaluates the Chebyshev expansion of nodal data at arbitrary
points inside the interval -- it is a spectral interpolation, not a
resampling.

.. code-block:: python

   fine = np.linspace(-1.0, 1.0, 201)
   y_fine = cheb.GetOnAxis(y, fine)
   np.max(np.abs(y_fine - np.exp(np.sin(3 * fine))))   # ~7e-10

Checking resolution
-------------------

:func:`~spectools.chebyshev.diagnostics.diagnose_chebyshev_power` inspects
the modal power to tell you whether the expansion has converged, and at what
order you could truncate it.

.. code-block:: python

   from spectools.chebyshev.diagnostics import diagnose_chebyshev_power

   report = diagnose_chebyshev_power(coeffs)
   report.decay              # 'spectral'
   report.tail_ratio         # ~6e-18 -- the last modes carry no power
   report.recommended_order  # 19 -- 32 modes is more than this needs
   report.reason             # 'Chebyshev modal tail is below tolerance'

Under-resolve the same function and the classification changes:

.. code-block:: python

   coarse = ChebyshevSpectral(Nfuncs=8, a=-1.0, b=1.0)
   xc = coarse.collocation_points_physical
   report = diagnose_chebyshev_power(
       coarse.MatrixPhysToSpec @ np.exp(np.sin(3 * xc))
   )
   report.decay       # 'algebraic'
   report.tail_ratio  # ~7e-03 -- still significant power in the tail

``decay`` is a
:class:`~spectools.chebyshev.diagnostics.DecayClassification`: one of
``zero``, ``spectral``, ``algebraic``, ``stalled``, ``noisy`` or
``underresolved``. ``recommended_order`` is ``None`` when no truncation
meets the tolerance.


Fourier
=======

Transforms on a periodic interval
---------------------------------

The Fourier helpers fix the convention
``f(theta_j) = sum_k c_k exp(i k theta_j)`` on equispaced nodes, and keep
coefficients in NumPy FFT order.

.. code-block:: python

   import numpy as np
   from spectools.fourier.transforms import (
       fourier_nodes,
       fourier_mode_numbers,
       fourier_transform_axis,
       fourier_contract_axis,
   )

   theta = fourier_nodes(16)              # 16 nodes on [0, 2*pi), endpoint excluded
   f = np.sin(2 * theta) + 0.5 * np.cos(5 * theta)

   coeffs = fourier_transform_axis(f)     # normalised, FFT-ordered
   fourier_mode_numbers(16)               # [0, 1, ..., 7, -8, -7, ..., -1]

   recovered = fourier_contract_axis(coeffs)   # back onto the same nodes

``fourier_nodes`` excludes the right endpoint by default, which is what you
want for a periodic function; pass ``endpoint=True`` only if you need the
closed interval. Both ``fourier_transform_axis`` and
``fourier_contract_axis`` take an ``axis`` argument, and
``fourier_transform_axes`` applies the transform along several axes in turn.

Evaluating off the grid
-----------------------

``fourier_evaluate`` evaluates the expansion at arbitrary points, not only
at the nodes:

.. code-block:: python

   from spectools.fourier.transforms import fourier_evaluate

   points = np.array([0.1, 1.0, 3.0])
   values = fourier_evaluate(coeffs, points)   # complex; take .real for real input

For a non-default interval, pass the same ``a`` and ``b`` you gave
``fourier_nodes``.

Checking resolution
-------------------

:func:`~spectools.fourier.diagnostics.diagnose_fourier_power` mirrors the
Chebyshev diagnostic, ranking modes by ``|m|`` rather than by order:

.. code-block:: python

   from spectools.fourier.diagnostics import diagnose_fourier_power

   report = diagnose_fourier_power(coeffs)
   report.decay       # 'spectral'
   report.tail_ratio  # ~1e-31 -- band-limited signal, nothing in the tail


Spherical
=========

Requires ``pip install spectools[spherical]``.

Angular grids
-------------

:class:`~spectools.spherical.grids.GLGrid` builds a Gauss-Legendre grid from
a single band limit; its ``meshgrid`` gives the ``(theta, phi)`` arrays to
sample on.

.. code-block:: python

   from spectools.spherical.grids import GLGrid

   grid = GLGrid(L=10)
   grid.ntheta, grid.nphi   # (11, 22)
   theta, phi = grid.meshgrid
   grid.grid_type           # 'GL'

:class:`~spectools.spherical.grids.UniformGrid` provides an equispaced
alternative, specified by its extents rather than a band limit:

.. code-block:: python

   from spectools.spherical.grids import UniformGrid

   uniform = UniformGrid(nphi=160, ntheta=81)
   uniform.shape   # (77, 156) -- ghost zones excluded

.. versionchanged:: 2026.10.02

   ``UniformGrid.__init__`` previously forwarded hard-coded defaults to
   its base class instead of the arguments it was given, so only the
   default resolution of ``nphi=80, ntheta=41`` was reachable. Code that
   passed non-default values and silently got the defaults will now get
   the grid it asked for.

Expansion and reconstruction
----------------------------

.. code-block:: python

   import numpy as np
   from spectools.spherical.transforms import SHExpandSimple, SHContract
   from waveformtools.diagnostics import MethodInfo

   theta, phi = grid.meshgrid
   f = 1.0 + np.cos(theta) + np.sin(theta) * np.cos(phi)

   method = MethodInfo(ell_max=10, int_method="GL")
   modes = SHExpandSimple(f, grid, method)      # a waveformtools SingleMode
   reconstructed = SHContract(modes, grid, 10, method)

   np.max(np.abs(np.asarray(reconstructed) - f))   # ~3e-14

``SHExpandSimple`` can additionally regularise the function at the poles
(``reg=True``) and return per-mode residuals (``error_info=True``). Use
:func:`~spectools.spherical.transforms.SHExpandAuto` when you want the band
limit chosen for you rather than fixed in advance.

Spin-weighted harmonics
-----------------------

:mod:`spectools.spherical.swsh` computes :math:`{}_sY_{\ell m}` directly.
Three implementations trade precision against speed:

* :mod:`spectools.spherical.Yslm_full_vec` -- ``numpy``-vectorised, roughly
  11 digits, fastest.
* :mod:`spectools.spherical.Yslm_mp` -- arbitrary precision at given
  coordinates, parallelised with ``multiprocessing``.
* :mod:`spectools.spherical.Yslm_prec_grid_mp` -- arbitrary precision over a
  whole grid, parallelised.

For a single mode, :func:`~spectools.spherical.swsh.Yslm_vec` is the
simplest route. To get every mode up to a band limit at once, use
:class:`~spectools.spherical.Yslm_full_vec.Yslm_full_vec`, which vectorises
over the mode numbers as well as the grid:

.. code-block:: python

   from spectools.spherical.Yslm_full_vec import Yslm_full_vec

   harmonics = Yslm_full_vec(Grid=grid, ell_max=8)
   modes = harmonics.compute_modes()   # (9, 9, ntheta, nphi)
   harmonics.mode(3, -2)               # one mode, (ntheta, nphi)

Both use a pole-stable closed form, so they are finite at
:math:`\theta = 0, \pi`.

.. versionchanged:: 2026.10.02

   ``Yslm_full_vec.compute_modes`` previously raised ``NameError`` on
   every call and ``construct_modes_grid`` produced invalid negative
   :math:`\ell` values. Both now work; results agree with
   :func:`~spectools.spherical.swsh.Yslm_vec` to machine precision.


Interpolation
=============

Requires ``pip install spectools[spherical]``, plus ``mpi4py`` for the
parallel routines.

:mod:`spectools.interpolation` provides a 3D spectral interpolator: angular
directions are expanded in spin-weighted spherical harmonics and the radial
direction in Chebyshev polynomials, so data on a set of concentric spherical
shells can be evaluated at arbitrary points.

The ``_single_t`` modules handle one time slice; the others sweep a time
axis. The ``_vectorized`` variants trade memory for speed. A full worked
example is in ``support/examples/spectral_interpolator.ipynb``.
