"""Chebyshev collocation spectral methods on an arbitrary interval.

Provides :class:`ChebyshevSpectral`, which bundles together the
Gauss-Lobatto collocation points of an interval ``[a, b]``, the matrices
that map between the physical (nodal) and spectral (modal) representation
of a function sampled there, the first and second derivative matrices, and
interpolation onto an arbitrary axis.

Every expensive object is computed lazily on first access and then cached
on the instance, so constructing a :class:`ChebyshevSpectral` is cheap and
only the matrices actually touched are ever built.

This module depends only on ``numpy``.
"""

import numpy as np

# from chebyshev_basis import ToSpecMatrix, ToSpecMatrixDirect, ToPhysMatrix, ChebDerPhysToPhysMatrix, ChebDerSpecToPhysMatrix, ChebBasisMem, ChebBasisDirect, ChebBasisRec
from spectools.chebyshev.basis import ChebyshevBasis


def message(*args, **kwargs):
    """Local no-op logger for import-light Chebyshev utilities."""

    return None


class ChebyshevSpectral:
    """A Chebyshev collocation basis on the physical interval ``[a, b]``.

    The collocation points are the ``Nfuncs`` Chebyshev-Gauss-Lobatto
    nodes, ordered from ``a`` to ``b``. A function is represented either
    *physically*, by its values at those nodes, or *spectrally*, by its
    coefficients in the Chebyshev polynomial basis; :attr:`MatrixPhysToSpec`
    and :attr:`MatrixSpecToPhys` map between the two.

    Parameters
    ----------
    Nfuncs : int
        Number of basis functions, equal to the number of collocation
        points. The highest resolved Chebyshev order is ``Nfuncs - 1``.
    a, b : float, optional
        Endpoints of the physical interval. Default to the logical
        interval ``[-1, 1]``.
    collocation_points_logical, collocation_points_physical : array_like, optional
        Precomputed collocation points, to reuse rather than recompute.
    MatrixPhysToSpec, MatrixSpecToPhys, MatrixD, MatrixDD : ndarray, optional
        Precomputed transform and derivative matrices, to reuse rather
        than recompute.

    Attributes
    ----------
    ChebyshevBasisSet : ChebyshevBasis
        The underlying basis object that supplies the raw matrices.

    Notes
    -----
    All matrices and point sets are computed on first access and cached.
    Passing them in through the constructor is purely an optimisation for
    reusing work across instances of the same ``Nfuncs``.

    Examples
    --------
    >>> import numpy as np
    >>> from spectools.chebyshev.chebyshev import ChebyshevSpectral
    >>> cheb = ChebyshevSpectral(Nfuncs=24, a=0.0, b=2.0)
    >>> x = cheb.collocation_points_physical
    >>> y = np.sin(x)
    >>> dy = cheb.MatrixD @ y
    >>> bool(np.allclose(dy, np.cos(x), atol=1e-10))
    True
    """

    def __init__(
        self,
        Nfuncs,
        a=-1,
        b=1,
        collocation_points_logical=None,
        collocation_points_physical=None,
        MatrixPhysToSpec=None,
        MatrixSpecToPhys=None,
        MatrixD=None,
        MatrixDD=None,
    ):

        # Find the basis coefficients.
        self._Nfuncs = Nfuncs
        self._a = a
        self._b = b
        self._collocation_points_logical = collocation_points_logical
        self._collocation_points_physical = collocation_points_physical
        self._MatrixPhysToSpec = MatrixPhysToSpec
        self._MatrixSpecToPhys = MatrixSpecToPhys
        self._MatrixD = MatrixD
        self._MatrixDD = MatrixDD

        self.ChebyshevBasisSet = ChebyshevBasis(Nfuncs=Nfuncs)

    @property
    def Nfuncs(self):
        """int: Number of basis functions, i.e. of collocation points."""
        return self._Nfuncs

    @property
    def a(self):
        """float: Left endpoint of the physical interval."""
        return self._a

    @property
    def b(self):
        """float: Right endpoint of the physical interval."""
        return self._b

    def basis_func(self, order):
        """Evaluate one Chebyshev basis function on the collocation points.

        Parameters
        ----------
        order : int
            Chebyshev polynomial order, in ``0 <= order < Nfuncs``.

        Returns
        -------
        ndarray
            ``T_order`` sampled at :attr:`collocation_points_logical`,
            of shape ``(Nfuncs,)``.
        """
        return self.ChebyshevBasisSet.ChebBasis(
            self.collocation_points_logical, order
        )

    @property
    def collocation_points_logical(self):
        """ndarray: Gauss-Lobatto nodes on the logical interval ``[-1, 1]``.

        The ``Nfuncs`` points ``-cos(n pi / (Nfuncs - 1))``, ascending from
        ``-1`` to ``1``. Computed on first access and cached.
        """

        if self._collocation_points_logical is None:

            message(
                "Computing logical collocation points", message_verbosity=2
            )
            Naxis = np.arange(self.Nfuncs)

            x_logical = -np.cos(Naxis * np.pi / (self.Nfuncs - 1))

            self._collocation_points_logical = x_logical

            message("Done", message_verbosity=2)
        return self._collocation_points_logical

    @property
    def collocation_points_physical(self):
        """ndarray: Collocation points mapped onto ``[a, b]``.

        :attr:`collocation_points_logical` affinely rescaled to the
        physical interval. These are the points at which a function must
        be sampled for the transform matrices to apply. Computed on first
        access and cached.
        """

        if self._collocation_points_physical is None:
            # Naxis = np.arange(self.Nfuncs+1)
            message(
                "Computing physical collocation points", message_verbosity=2
            )
            # x_logical = -np.cos(Naxis*np.pi/self.Nfuncs)

            self._collocation_points_physical = (
                self.a
                + (self.b - self.a) * (1 + self.collocation_points_logical) / 2
            )
            message("Done", message_verbosity=2)

        return self._collocation_points_physical

    @property
    def MatrixPhysToSpec(self):
        """ndarray: Nodal-to-modal transform matrix, shape ``(Nfuncs, Nfuncs)``.

        Maps the values of a function at :attr:`collocation_points_physical`
        to its coefficients in the Chebyshev polynomial basis::

            coeffs = cheb.MatrixPhysToSpec @ values

        Inverse of :attr:`MatrixSpecToPhys`. Computed on first access and
        cached.
        """

        if self._MatrixPhysToSpec is None:
            # N_coord_points = len(x_logical_axis)
            message("Computing ToSpec", message_verbosity=2)
            # self._MatrixPhysToSpec = ToSpecMatrixDirect(self.collocation_points_logical)
            self._MatrixPhysToSpec = self.ChebyshevBasisSet.ToSpecMatrix(
                self.collocation_points_logical
            )
            message("Done", message_verbosity=2)

        return self._MatrixPhysToSpec

    @property
    def MatrixSpecToPhys(self):
        """ndarray: Modal-to-nodal transform matrix, shape ``(Nfuncs, Nfuncs)``.

        Maps Chebyshev coefficients back to function values at
        :attr:`collocation_points_physical`::

            values = cheb.MatrixSpecToPhys @ coeffs

        Inverse of :attr:`MatrixPhysToSpec`. Computed on first access and
        cached.
        """

        if self._MatrixSpecToPhys is None:

            message("Computing ToPhys", message_verbosity=2)

            self._MatrixSpecToPhys = self.ChebyshevBasisSet.ToPhysMatrix(
                self.collocation_points_logical
            )

            message("Done", message_verbosity=2)

        return self._MatrixSpecToPhys

    @property
    def MatrixD(self):
        """ndarray: First derivative matrix, shape ``(Nfuncs, Nfuncs)``.

        Maps function values at :attr:`collocation_points_physical` to the
        values of the first derivative at the same points, including the
        ``2 / (b - a)`` factor for the physical interval::

            dydx = cheb.MatrixD @ y

        Computed on first access and cached.
        """

        if self._MatrixD is None:
            message("Computing MatrixD", message_verbosity=2)
            self._MatrixD = (
                2 / (self.b - self.a)
            ) * self.ChebyshevBasisSet.ChebDerPhysToPhysMatrix(
                self.collocation_points_logical
            )

            message("Done", message_verbosity=2)

        return self._MatrixD

    @property
    def MatrixDD(self):
        """ndarray: Second derivative matrix, shape ``(Nfuncs, Nfuncs)``.

        Maps function values at :attr:`collocation_points_physical` to the
        values of the second derivative at the same points::

            d2ydx2 = cheb.MatrixDD @ y

        Computed on first access and cached.

        .. versionchanged:: 2026.10.02

           Previously carried a spurious ``((b - a) / 2) ** 2`` prefactor
           that cancelled the ``2 / (b - a)`` scaling already applied by
           :attr:`MatrixD`, making the result correct only on an interval
           of width 2 (such as the default ``[-1, 1]``). On any other
           interval it was wrong by that factor. Code that compensated for
           this by rescaling the result must drop the correction.

        Examples
        --------
        >>> import numpy as np
        >>> from spectools.chebyshev.chebyshev import ChebyshevSpectral
        >>> cheb = ChebyshevSpectral(Nfuncs=28, a=0.0, b=4.0)
        >>> x = cheb.collocation_points_physical
        >>> y = np.sin(x)
        >>> bool(np.allclose(cheb.MatrixDD @ y, -np.sin(x), atol=1e-9))
        True
        """

        if self._MatrixDD is None:

            message("Computing MatrixDD", message_verbosity=2)

            D1Phys = self.MatrixD

            # D1Phys already carries the 2/(b-a) scaling, so squaring it
            # gives the physical second derivative directly. Any extra
            # interval prefactor here would cancel that scaling back out.
            self._MatrixDD = D1Phys @ D1Phys

            message("Done", message_verbosity=2)

        return self._MatrixDD

    def TransformPhysicalToLogical(self, x):
        """Transform the given coordinate value in physical space
        to logical space"""

        return 2 * (x - self.a) / (self.b - self.a) - 1

    def TransformLogicalToPhysical(self, x):
        """Transform the given coordinate value in logical space
        to physical space"""

        return self.a + (self.b - self.a) * (x + 1) / 2

    def EvaluateBasis(self, x, order):
        """Evaluate the Chebyshev basis of the required order
        at the requested physical point `x`.

        Parameters
        ----------
        x : float
            Physical coordinate, inside ``[a, b]``. A point just outside
            the interval is snapped back onto the boundary, to absorb
            rounding error in the caller's arithmetic.
        order : int
            Chebyshev polynomial order.

        Returns
        -------
        float
            The basis function of that order evaluated at ``x``.

        Raises
        ------
        ValueError
            If ``x`` lies more than ``1e-3`` outside the interval in
            logical coordinates.

        .. versionchanged:: 2026.10.02

           The snapping branch referenced an undefined name ``sgn`` in its
           log message and so raised ``NameError`` instead of correcting
           the coordinate, making any point marginally outside the
           interval -- the exact case the branch exists to handle -- fail.
        """

        x_log = self.TransformPhysicalToLogical(x)

        delta = abs(x_log) - 1

        if delta > 0:
            # tol = abs(x_log + 1)

            if delta > 1e-3:
                raise ValueError(
                    f"The requested Chebyshev collocation point is outside the domain! x_log {x_log}. Please check the input coord x {x}"
                )
            else:
                sgn = np.sign(x_log)

                message(
                    f"Manually setting x_log to {sgn} to correct for numerical precision error of {delta}",
                    message_verbosity=2,
                )

                x_log = sgn

        message(f"Value of x_log {x_log}", message_verbosity=4)

        # if x in self.collocation_points_logical:
        #    ind = np.argmin(abs(self.collocation_points_logical - x))
        #    basis_val = self.Bas

        return self.ChebyshevBasisSet.ChebBasisEval(x_log, order)

    def GetOnAxis(self, y_vals, x_axis):
        """Get the given `y_vals` originally at the physical colocation points
        onto a new `x_axis` by Chebyshev interpolations"""

        y_vals = np.array(y_vals)

        y_spec = self.MatrixPhysToSpec @ y_vals

        x_log = self.TransformPhysicalToLogical(np.asarray(x_axis))
        theta = np.arccos(np.clip(x_log, -1.0, 1.0))
        orders = np.arange(self.Nfuncs, dtype=np.float64)
        eval_matrix = np.cos(np.outer(theta, orders))

        return np.tensordot(eval_matrix, y_spec, axes=(1, 0))
