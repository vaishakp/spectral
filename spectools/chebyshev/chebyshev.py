import numpy as np

# from chebyshev_basis import ToSpecMatrix, ToSpecMatrixDirect, ToPhysMatrix, ChebDerPhysToPhysMatrix, ChebDerSpecToPhysMatrix, ChebBasisMem, ChebBasisDirect, ChebBasisRec
from spectools.chebyshev.basis import ChebyshevBasis


def message(*args, **kwargs):
    """Local no-op logger for import-light Chebyshev utilities."""

    return None


class ChebyshevSpectral:

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
        return self._Nfuncs

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def basis_func(self, order):
        return self.ChebyshevBasisSet.ChebBasis(
            self.collocation_points_logical, order
        )

    @property
    def collocation_points_logical(self):

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
        """The matrix to transform from physical coordiate space
        representation of the function to the Chebyshev polynomial
        basis."""

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
        """The matrix to transform to physical coordiate space
        representation of the function from the Chebyshev polynomial
        basis."""

        if self._MatrixSpecToPhys is None:

            message("Computing ToPhys", message_verbosity=2)

            self._MatrixSpecToPhys = self.ChebyshevBasisSet.ToPhysMatrix(
                self.collocation_points_logical
            )

            message("Done", message_verbosity=2)

        return self._MatrixSpecToPhys

    @property
    def MatrixD(self):
        """The matrix of the derivative
        of the basis functions. Given the physical coeffs, this
        matrix maps them to the derivate in physical coords
        `x_physical_axis`"""

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
        """The matrix of the double derivative
        of the basis functions. Given the physical coeffs, this
        matrix maps them to the derivate in physical coords `x_logical_axis`"""

        if self._MatrixDD is None:

            message("Computing MatrixDD", message_verbosity=2)

            D1Phys = self.MatrixD

            self._MatrixDD = ((self.b - self.a) / 2) ** 2 * D1Phys @ D1Phys

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
        at the requested physical point `x`"""

        x_log = self.TransformPhysicalToLogical(x)

        delta = abs(x_log) - 1

        if delta > 0:
            # tol = abs(x_log + 1)

            if delta > 1e-3:
                raise ValueError(
                    f"The requested Chebyshev collocation point is outside the domain! x_log {x_log}. Please check the input coord x {x}"
                )
            else:
                message(
                    f"Manually setting x_log to {sgn} to correct for numerical precision error of {delta}",
                    message_verbosity=2,
                )

                x_log = np.sign(x_log)

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
