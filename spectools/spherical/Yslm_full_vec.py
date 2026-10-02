"""Fully vectorised spin-weighted spherical harmonics.

Computes :math:`{}_sY_{\\ell m}` over an entire grid *and* over all modes
up to a band limit at once, using pure ``numpy`` in double precision. This
is the fastest of the three implementations in this subpackage, at roughly
11 digits of accuracy.

Use :mod:`spectools.spherical.Yslm_mp` or
:mod:`spectools.spherical.Yslm_prec_grid_mp` when more precision than
double is needed.

.. note::

   Only spin weight 0 is implemented; the constructor asserts on anything
   else.

Requires the ``spherical`` extra.
"""

import numpy as np
import math


def _safe_comb(n, k):
    """Binomial coefficient over integer arrays, zero where undefined.

    ``math.comb`` raises on a negative argument, but the Wigner-d sum
    needs those entries to simply drop out. This returns ``0.0`` wherever
    ``k < 0``, ``k > n`` or ``n < 0``.

    Parameters
    ----------
    n, k : array_like of int
        Broadcast against one another.

    Returns
    -------
    ndarray of float
        ``C(n, k)``, or ``0.0`` where the coefficient is undefined.
    """

    n_arr, k_arr = np.broadcast_arrays(np.asarray(n), np.asarray(k))
    out = np.zeros(n_arr.shape, dtype=float)

    valid = (n_arr >= 0) & (k_arr >= 0) & (k_arr <= n_arr)

    for index in np.argwhere(valid):
        pos = tuple(index)
        out[pos] = float(math.comb(int(n_arr[pos]), int(k_arr[pos])))

    return out


def _safe_factorial(n):
    """Factorial over an integer array, zero where the argument is negative.

    Parameters
    ----------
    n : array_like of int

    Returns
    -------
    ndarray of float
        ``n!`` as a float, or ``0.0`` where ``n < 0``.
    """

    n_arr = np.asarray(n)
    out = np.zeros(n_arr.shape, dtype=float)

    for index in np.argwhere(n_arr >= 0):
        pos = tuple(index)
        out[pos] = float(math.factorial(int(n_arr[pos])))

    return out


class Yslm_full_vec:
    """A container to compute the SWSH, vectorized over the mode numbers.

    Parameters
    ----------
    Grid : SphericalGrid
        The angular grid to evaluate on; its ``meshgrid`` supplies the
        :math:`(\\theta, \\phi)` arrays.
    ell_max : int
        Band limit. All modes with :math:`\\ell \\le` ``ell_max`` are
        computed.
    spin_weight : int, optional
        Spin weight of the harmonics. Must be ``0``; other values raise
        ``AssertionError``.

    Notes
    -----
    :meth:`compute_modes` calls :meth:`construct_modes_grid` itself if
    needed, so it can be invoked directly after construction.

    Examples
    --------
    >>> import numpy as np
    >>> from spectools.spherical.grids import GLGrid
    >>> from spectools.spherical.Yslm_full_vec import Yslm_full_vec
    >>> obj = Yslm_full_vec(Grid=GLGrid(L=6), ell_max=4)
    >>> modes = obj.compute_modes()
    >>> modes.shape[:2]
    (5, 5)
    >>> obj.mode(2, -1).shape
    (7, 14)
    """

    def __init__(
        self,
        Grid,
        ell_max,
        spin_weight=0,
    ):

        self._spin_weight = spin_weight
        self._Grid = Grid
        self._ell_max = ell_max
        self._modes_grid = None
        self._modes = None
        self._n_ells = None

        assert (
            spin_weight == 0
        ), "This has only been implemented for spin_weight 0"

    @property
    def Grid(self):
        """SphericalGrid: The angular grid the harmonics are evaluated on."""
        return self._Grid

    @property
    def ell_max(self):
        """int: Band limit; the largest :math:`\\ell` computed."""
        return self._ell_max

    @property
    def modes_grid(self):
        """tuple of ndarray: The ``(ell_grid, emm_grid)`` mode labels.

        Two 2D integer arrays giving the :math:`\\ell` and :math:`m` of
        every mode, laid out so the harmonic computation can be vectorised
        over modes. Set by :meth:`construct_modes_grid`; accessing it
        before that call raises ``AttributeError``.
        """
        return self._modes_grid

    @property
    def spin_weight(self):
        """int: Spin weight of the harmonics. Always ``0`` here."""
        return self._spin_weight

    @property
    def n_ells(self):
        """int: Number of distinct :math:`\\ell` values, ``ell_max + 1``.

        Set during :meth:`compute_modes`.
        """
        return self._n_ells

    def construct_modes_grid(self):
        """Build the ``(ell, emm)`` mode labels to vectorize over.

        Every mode with :math:`\\ell \\le` ``ell_max`` is placed at a
        distinct position of an ``(ell_max + 1, ell_max + 1)`` grid, using
        the standard ``ell``-major flat ordering

        .. math::

           n = \\ell^2 + \\ell + m

        reshaped row-major. That ordering is a bijection onto
        ``0 .. (ell_max + 1)^2 - 1``, so the square grid is filled exactly,
        with no padding and no mode repeated. The position of a given mode
        is therefore ``divmod(ell**2 + ell + emm, ell_max + 1)``.

        Sets :attr:`modes_grid` to the ``(ell_grid, emm_grid)`` pair.
        Returns nothing.

        .. versionchanged:: 2026.10.02

           Previously assigned ``meta_emm_grid = ell_grid.T``, which is a
           *view* rather than a copy, so writing the azimuthal numbers also
           overwrote the transposed entries of ``ell_grid``. The returned
           ``ell_grid`` consequently contained negative values, which are
           not valid :math:`\\ell`. The layout it attempted could not hold
           all modes either: column ``j`` needed rows ``0..2j``, which
           exceeds the grid for ``ell_max > 1``.

        Examples
        --------
        >>> import numpy as np
        >>> from spectools.spherical.grids import GLGrid
        >>> from spectools.spherical.Yslm_full_vec import Yslm_full_vec
        >>> obj = Yslm_full_vec(Grid=GLGrid(L=4), ell_max=3)
        >>> obj.construct_modes_grid()
        >>> ell_grid, emm_grid = obj.modes_grid
        >>> pairs = sorted(zip(ell_grid.ravel().tolist(),
        ...                    emm_grid.ravel().tolist()))
        >>> pairs == [(l, m) for l in range(4) for m in range(-l, l + 1)]
        True
        """

        ell_max = self.ell_max
        n_side = ell_max + 1

        ell_flat = np.zeros(n_side * n_side, dtype=int)
        emm_flat = np.zeros(n_side * n_side, dtype=int)

        for ell in range(n_side):
            for emm in range(-ell, ell + 1):
                index = ell * ell + ell + emm
                ell_flat[index] = ell
                emm_flat[index] = emm

        self._modes_grid = (
            ell_flat.reshape(n_side, n_side),
            emm_flat.reshape(n_side, n_side),
        )

    def compute_modes(self):
        """Evaluate all harmonics up to ``ell_max`` over the whole grid.

        Vectorized over both the grid and the mode numbers: the sum over
        the Wigner-d index ``aar`` is the only Python-level loop, and each
        of its ``ell_max + 1`` iterations updates every mode at every grid
        point at once.

        Returns
        -------
        ndarray
            Complex array of shape
            ``(ell_max + 1, ell_max + 1, ntheta, nphi)``. The leading two
            axes carry the mode, packed as described in
            :meth:`construct_modes_grid`; the trailing two are the grid.
            Also stored on the instance and returned by :attr:`modes`.

        Raises
        ------
        AssertionError
            If the grid is too coarse to resolve ``ell_max``, i.e. if
            ``ntheta - 1 < ell_max``.

        Notes
        -----
        Calls :meth:`construct_modes_grid` itself if that has not been
        done. Uses the pole-stable ``cos``/``sin`` closed form, so no
        nudging of :math:`\\theta` away from the poles is required and the
        result is finite at :math:`\\theta = 0, \\pi`.

        Accumulates in ``complex128``; the result is accurate to roughly
        11 digits, matching
        :func:`spectools.spherical.swsh.Yslm_vec`.

        .. versionchanged:: 2026.10.02

           Previously raised ``NameError``: the body had been adapted from
           a single-mode routine and still referred to loop variables
           ``ell`` and ``emm`` that were never bound in this scope. It also
           never returned a value, divided by ``tan(theta / 2)`` (singular
           at the poles), and accumulated in ``complex256``, which does not
           exist on Windows or some ARM builds of NumPy.

        Examples
        --------
        >>> import numpy as np
        >>> from spectools.spherical.grids import GLGrid
        >>> from spectools.spherical.swsh import Yslm_vec
        >>> from spectools.spherical.Yslm_full_vec import Yslm_full_vec
        >>> grid = GLGrid(L=6)
        >>> obj = Yslm_full_vec(Grid=grid, ell_max=4)
        >>> modes = obj.compute_modes()
        >>> modes.shape[:2]
        (5, 5)
        >>> theta, phi = grid.meshgrid
        >>> row, col = divmod(3 ** 2 + 3 + (-2), obj.ell_max + 1)
        >>> reference = Yslm_vec(0, 3, -2, theta, phi)
        >>> bool(np.allclose(modes[row, col], reference, atol=1e-10))
        True
        """

        theta_grid, phi_grid = self.Grid.meshgrid

        ntheta, nphi = theta_grid.shape

        grid_ell_max = ntheta - 1

        assert (
            grid_ell_max >= self.ell_max
        ), "The current grid does not support the requested ell_max"

        if self._modes_grid is None:
            self.construct_modes_grid()

        ell_grid, emm_grid = self.modes_grid

        self._n_ells = self.ell_max + 1

        theta = np.asarray(theta_grid, dtype=float)[np.newaxis, np.newaxis]
        phi = np.asarray(phi_grid, dtype=float)[np.newaxis, np.newaxis]

        abs_spin_weight = abs(self.spin_weight)

        # For negative spin weight the harmonic is obtained from the
        # positive-spin one at the antipode, with an ell-dependent sign.
        factor = 1.0
        if self.spin_weight < 0:
            factor = ((-1.0) ** ell_grid)[:, :, np.newaxis, np.newaxis]
            theta = np.pi - theta
            phi = phi + np.pi

        half_cos = np.cos(theta / 2)
        half_sin = np.sin(theta / 2)
        azimuth = np.exp(1j * emm_grid[:, :, np.newaxis, np.newaxis] * phi)

        Sum = np.zeros(
            (self.ell_max + 1, self.ell_max + 1, ntheta, nphi),
            dtype=np.complex128,
        )

        for aar in range(0, self.ell_max - abs_spin_weight + 1):

            term1 = _safe_comb(ell_grid - abs_spin_weight, aar)
            term2 = _safe_comb(
                ell_grid + abs_spin_weight, aar + abs_spin_weight - emm_grid
            )

            # A vanishing binomial kills the term; skipping those entries
            # is also what keeps a negative exponent below from being
            # raised on a zero base.
            keep = (term1 != 0) & (term2 != 0)

            if not keep.any():
                continue

            cos_exp = 2 * aar + abs_spin_weight - emm_grid
            sin_exp = 2 * ell_grid - cos_exp

            coeff = (
                term1
                * term2
                * np.power(-1.0, ell_grid - aar - abs_spin_weight)
            )
            coeff = np.where(keep, coeff, 0.0)

            powers = np.power(
                half_cos, np.clip(cos_exp, 0, None)[:, :, None, None]
            ) * np.power(half_sin, np.clip(sin_exp, 0, None)[:, :, None, None])

            Sum += coeff[:, :, np.newaxis, np.newaxis] * azimuth * powers

        norm = np.power(-1.0, emm_grid) * np.sqrt(
            _safe_factorial(ell_grid + emm_grid)
            * _safe_factorial(ell_grid - emm_grid)
            * (2 * ell_grid + 1)
            / (
                4
                * np.pi
                * _safe_factorial(ell_grid + abs_spin_weight)
                * _safe_factorial(ell_grid - abs_spin_weight)
            )
        )

        self._modes = factor * norm[:, :, np.newaxis, np.newaxis] * Sum

        return self._modes

    @property
    def modes(self):
        """ndarray: The harmonics computed by :meth:`compute_modes`.

        Shape ``(ell_max + 1, ell_max + 1, ntheta, nphi)``. ``None``
        before :meth:`compute_modes` has been called.
        """
        return self._modes

    def mode(self, ell, emm):
        """Return the computed harmonic for a single mode.

        Parameters
        ----------
        ell, emm : int
            Mode numbers :math:`\\ell` and :math:`m`, with
            ``-ell <= emm <= ell`` and ``ell <= ell_max``.

        Returns
        -------
        ndarray
            :math:`{}_sY_{\\ell m}` over the grid, of shape
            ``(ntheta, nphi)``.

        Raises
        ------
        ValueError
            If the mode is out of range, or if :meth:`compute_modes` has
            not been called yet.
        """
        if self._modes is None:
            raise ValueError("Call compute_modes() before requesting a mode")

        if not 0 <= ell <= self.ell_max:
            raise ValueError(
                f"ell must satisfy 0 <= ell <= {self.ell_max}, got {ell}"
            )

        if abs(emm) > ell:
            raise ValueError(f"abs(emm) must not exceed ell, got {emm}, {ell}")

        row, col = divmod(ell * ell + ell + emm, self.ell_max + 1)

        return self._modes[row, col]
