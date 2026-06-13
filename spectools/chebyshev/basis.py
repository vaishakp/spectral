import numpy as np

""" Deals with Chebyshev approximations of the first kind """

# Nmax = 25
# from numba import jit, njit
# cheb_basis_array = np.zeros(Nmax)

def message(*args, **kwargs):
    """Local no-op logger for import-light Chebyshev utilities."""

    return None


def chebyshev_transform_axis(values, matrix, axis=0):
    """Apply a one-dimensional Chebyshev transform along one array axis.

    Parameters
    ----------
    values:
        Array with collocation values. The length of ``values`` along ``axis``
        must match ``matrix.shape[1]``.
    matrix:
        Transform matrix with shape ``(n_modes, n_points)``. For physical to
        spectral transforms this is ``MatrixPhysToSpec``; for contractions back
        to collocation values this can be ``MatrixSpecToPhys``.
    axis:
        Axis of ``values`` to transform.

    Returns
    -------
    numpy.ndarray
        Array with the transformed axis kept in the same axis position.
    """

    values = np.asarray(values)
    matrix = np.asarray(matrix)
    moved = np.moveaxis(values, axis, 0)
    transformed = np.tensordot(matrix, moved, axes=(1, 0))
    return np.moveaxis(transformed, 0, axis)


def chebyshev_transform_axes(values, matrices, axes):
    """Apply several one-dimensional Chebyshev transforms sequentially."""

    transformed = np.asarray(values)
    for matrix, axis in zip(matrices, axes):
        transformed = chebyshev_transform_axis(
            transformed,
            matrix,
            axis=axis,
        )
    return transformed


def chebyshev_batched_transform_axis(values, matrices, domain_axis=0, axis=1):
    """Apply per-domain Chebyshev transforms along one axis.

    ``matrices`` may be either one shared transform matrix with shape
    ``(n_modes, n_points)`` or one transform per domain with shape
    ``(n_domains, n_modes, n_points)``.
    """

    values = np.asarray(values)
    matrices = np.asarray(matrices)

    if matrices.ndim == 2:
        return chebyshev_transform_axis(values, matrices, axis=axis)

    if matrices.ndim != 3:
        raise ValueError("matrices must have shape (modes, points) or "
                         "(domains, modes, points)")

    values_by_domain = np.moveaxis(values, domain_axis, 0)
    axis_in_domain = axis if axis < domain_axis else axis - 1
    if len(values_by_domain) != len(matrices):
        raise ValueError("number of domains in values and matrices differ")

    transformed = [
        chebyshev_transform_axis(domain_values, matrix, axis=axis_in_domain)
        for domain_values, matrix in zip(values_by_domain, matrices)
    ]
    return np.moveaxis(np.stack(transformed, axis=0), 0, domain_axis)


# @njit(parallel=True)


class ChebyshevBasis:
    """A chebyshev basis class comprising of polynomials of the first kind"""

    def __init__(
        self,
        Nfuncs=8,
        basis_calc_method="memoize",
    ):

        self._Nfuncs = Nfuncs
        self._basis_storage = {}
        self._basis_der_storage = {}
        self._to_phys_matrix_storage = {}
        self._to_spec_matrix_storage = {}
        self._basis_calc_method = basis_calc_method

        if self._basis_calc_method == "memoize":
            self.ChebBasis = self.ChebBasisMem
            self.ChebBasisEval = self.ChebBasisDirect

        elif self._basis_calc_method == "direct":
            self.ChebBasis = self.ChebBasisDirect
            self.ChebBasisEval = self.ChebBasisDirect

        elif self._basis_calc_method == "recursive":
            self.ChebBasis = self.ChebBasisRec
            self.ChebBasisEval = self.ChebBasisRec

        else:
            raise KeyError(
                "Unknown basis calculation method", self._basis_calc_method
            )

    @property
    def basis_storage(self):
        return self._basis_storage

    @property
    def basis_der_storage(self):
        return self._basis_der_storage

    @property
    def Nfuncs(self):
        return self._Nfuncs

    def CollocationPoints(self):
        Naxis = np.arange(self.Nfuncs)
        # Naxis = np.arange()

        return -np.cos(np.pi * Naxis / (self.Nfuncs - 1))
        # return np.cos(2*)

    # @njit(parallel=True)
    def MapToAB(self, x_axis, a, b):

        return a + (b - a) * (1 + x_axis) / 2

    # @njit(parallel=True, cache=True)
    def ChebBasisDirect(self, x_axis, order):
        """Return the chebyshev polynomial of First kind
        of order `order`"""
        # print(x_axis)

        return np.cos(order * np.arccos(x_axis))

    # @njit(parallel=True, cache=True)

    def ChebBasisRec(self, x_axis, order):
        """Return the chebyshev basis polynomial of First kind
        of order `order`"""

        x_axis = np.array(x_axis)

        if order == 0:
            return np.ones(len(x_axis))

        if order == 1:
            return x_axis

        else:
            return 2 * x_axis * self.ChebBasisRec(
                x_axis, order - 1
            ) - self.ChebBasisRec(x_axis, order - 2)

    def ChebBasisMem(self, x_axis, order):
        """Return the chebyshev basis polynomial of First kind
        of order `order`"""

        if order not in list(self.basis_storage.keys()):

            message(
                f"Constructing basis of order {order}", message_verbosity=3
            )

            if order == 0:
                self._basis_storage.update({0: np.ones(len(x_axis))})

            elif order == 1:
                self._basis_storage.update({1: x_axis})

            else:
                self._basis_storage.update(
                    {
                        order: 2
                        * x_axis
                        * self.ChebBasisMem(x_axis, order - 1)
                        - self.ChebBasisMem(x_axis, order - 2)
                    }
                )

        return self.basis_storage[order]

    # @njit(parallel=True)
    def ToPhys(self, x_axis, u_spec):
        """Transformation a vector from Chybyshev spectral
        to physical space"""

        Nmax = len(u_spec)

        u_coord = np.zeros(Nmax)

        for order in range(Nmax):

            u_coord += u_spec[order] * self.ChebBasis(x_axis, order)

        return u_coord

    # @njit(parallel=True)
    def ToPhysMatrix(self, x_axis):
        """Transformation matrix from physical
        to Chebyshev spectral space"""

        x_axis = np.asarray(x_axis, dtype=np.float64)
        key = self._axis_cache_key(x_axis)
        if key in self._to_phys_matrix_storage:
            return self._to_phys_matrix_storage[key]

        Nmax = len(x_axis)
        theta = np.arccos(np.clip(x_axis, -1.0, 1.0))
        orders = np.arange(Nmax, dtype=np.float64)
        matrix = np.cos(np.outer(theta, orders))
        self._to_phys_matrix_storage[key] = matrix
        return matrix

    # @njit(parallel=False)
    def ToSpecMatrix(self, x_axis):
        """Transformation matrix from the physical
        to spectral space"""

        x_axis = np.asarray(x_axis, dtype=np.float64)
        key = self._axis_cache_key(x_axis)
        if key not in self._to_spec_matrix_storage:
            self._to_spec_matrix_storage[key] = np.linalg.inv(
                self.ToPhysMatrix(x_axis)
            )
        return self._to_spec_matrix_storage[key]

    # @njit(parallel=True)
    def ToSpecMatrixDirect(self, x_axis):
        """Transformation matrix from the physical
        to spectral space directly using Gaussian quadrature
        over products of basis functions"""
        Nmax = len(x_axis)

        # cbar = 0.5*np.ones(Nmax)*(Nmax-1)
        # cbar = 0.5*np.ones(Nmax)*(Nmax+1)
        cbar = np.ones(Nmax)

        # cbar[0] = 1*(Nmax)
        # cbar[-1] = 1*(Nmax)

        cbar[0] = 2
        cbar[-1] = 2

        # cnorm = Nmax * np.ones((Nmax))/2
        # cnorm[0] = Nmax
        cnorm = (Nmax - 1) * np.ones(Nmax) / 2
        cnorm[0] = Nmax - 1
        cnorm[-1] = Nmax - 1

        Tmatrix = np.zeros((Nmax, Nmax))

        for index_i in range(Nmax):

            phys_basis_i = np.zeros(Nmax)
            phys_basis_i[index_i] = 1

            for index_j in range(Nmax):

                cheb_basis_j = self.ChebBasis(x_axis, index_j)

                Cij = np.dot(phys_basis_i, cheb_basis_j / cbar)

                Tmatrix[index_i, index_j] = Cij / cnorm[index_j]

        return Tmatrix.T

    # @njit(parallel=True, cache=True)
    def ChebBasisDer(self, x_axis, order):
        """Compute and return the derivative
        vector of a Chebyshev Basis function"""

        Nmax = len(x_axis)

        assert (
            Nmax == self.Nfuncs
        ), "The input Npoints does not agree with basis Nfuncs"

        if order not in list(self.basis_der_storage.keys()):

            message(
                f"Constructing basis derivative of order {order}",
                message_verbosity=3,
            )

            if order == 0:
                self._basis_der_storage.update({0: np.zeros(Nmax)})

            elif order == 1:
                self._basis_der_storage.update({1: np.ones(Nmax)})

            else:
                self._basis_der_storage.update(
                    {
                        order: 2 * self.ChebBasis(x_axis, order - 1)
                        + 2 * x_axis * self.ChebBasisDer(x_axis, order - 1)
                        - self.ChebBasisDer(x_axis, order - 2)
                    }
                )

        return self.basis_der_storage[order]

    # @njit(parallel=True)
    def ChebDerSpecToPhysMatrix(self, x_axis):
        """The operator to compute the derivative
        of a vector in spectral space, returning a
        vector in physical space.

        Takes in spectral
        Gives out physical

        """

        Nmax = len(x_axis)

        der_matrix = np.zeros((Nmax, Nmax))

        # Derivative of Basis vectors as the columns
        for order in range(Nmax):

            this_col = self.ChebBasisDer(x_axis, order)

            der_matrix[:, order] = this_col

        return der_matrix

    # @njit(parallel=False)
    def ChebDerPhysToPhysMatrix(self, x_axis):
        """The operator to compute the derivative
        of a vector in physical space, returning a
        vector in physical space"""

        der_mat_spec_to_phys = self.ChebDerSpecToPhysMatrix(x_axis)

        # t_matrix_spec_to_coord = self.ToPhysMatrix(x_axis)

        t_matrix_coord_to_spec = self.ToSpecMatrix(x_axis)

        return der_mat_spec_to_phys @ t_matrix_coord_to_spec

    def _axis_cache_key(self, x_axis):
        """Return a stable cache key for one collocation/evaluation axis."""

        axis = np.ascontiguousarray(x_axis, dtype=np.float64)
        return (axis.shape, axis.dtype.str, axis.tobytes())

    # ChebBasis = ChebBasisMem
    # ChebBasis = ChebBasisDirect
    # ChebBasis = ChebBasisRec
