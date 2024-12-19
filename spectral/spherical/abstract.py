from abc import ABC, abstractmethod
import numpy as np


class SphericalGrid(ABC):
    """An abstract class for spherical grids"""

    # @abstractmethod
    def __init__(
        self,
        nphi=80,
        ntheta=41,
        nphimax=124,
        nthetamax=66,
        nghosts=2,
        integration_method="MP",
        grid_type="Abstract",
    ):

        # Number of gridpoints along phi direction including ghost points.
        self._nphi = nphi
        # Number of gridpoints along theta direction including ghost points.
        self._ntheta = ntheta
        # Total length of phi array used by ETK.
        self._nphi_max = nphimax
        # Total length of theta array used by ETK.
        self._ntheta_max = nthetamax
        # Number of ghost points in theta/phi direction.
        self._nghosts = nghosts
        # The default integration method
        self._integration_method = integration_method

        self._grid_type = grid_type

    @property
    def grid_type(self):
        return self._grid_type

    @property
    def nphi(self):
        return self._nphi

    @property
    def ntheta(self):
        return self._ntheta

    @property
    def nghosts(self):
        return self._nghosts

    @property
    def ntheta_max(self):
        return self._ntheta_max

    @property
    def nphi_max(self):
        return self._nphi_max

    @property
    def npix(self):
        # Return the total number of pixels, including the ghost zones present at one iteration.
        return (self.ntheta) * (self.nphi)

    @property
    def npix_act(self):
        # Return the actual number of pixels, excluding the ghost zones present at one iteration.
        return (self.ntheta - 2 * self.nghosts) * (self.nphi - 2 * self.nghosts)

    @property
    def npix_max(self):
        # Return the (max) total number of pixels, including the ghost and buffer zones at one iteration.
        return (self.ntheta_max) * (self.nphi_max)

    @property
    def ntheta_act(self):
        """Return the actual number of valid pixels,
        excluding the ghost and buffer zones, along the
        theta axis at one iteration."""
        return self.ntheta - 2 * self.nghosts

    @property
    def nphi_act(self):
        """Return the actual number of valid pixels,
        excluding the ghost and buffer zones,
        along the phi axis at one iteration."""
        return self.nphi - 2 * self.nghosts

    @property
    @abstractmethod
    def dtheta(self):
        # Return the coodinate spacing d\theta
        pass

    @property
    def dphi(self):
        # Return the coordinate spacing d\phi
        return 2 * np.pi / (self.nphi - 2 * self.nghosts)

    @property
    def nbuffer(self):
        # Return the buffer cells (excluding the ghost zones)
        return self.ntheta_max - self.ntheta

    @property
    def shape(self):
        """Return the shape of the grid excluding the
        ghost and buffer zones"""
        return (self.ntheta_act, self.nphi_act)

    @property
    @abstractmethod
    def theta_1d(self, theta_index=None):
        """Returns the coordinate value theta
        given the coordinate index. The coordinate
        index ranges from (0, ntheta). The actual
        indices without the ghost and extra zones
        is (nghosts, ntheta-nghosts).

        Parameters
        -----------
        theta_index : int/ 1d array
                      The theta coordinate index or axis.

        Returns
        -------
        theta_1d : float
                   The coordinate(s) :math:`\\theta` on the sphere.
        """

    @property
    def phi_1d(self, phi_index=None):
        """Returns the coordinate value theta given
        the coordinate index. The coordinate index lies
        in (0, nphi). The actual indices without
        the ghost and extra zones is (nghosts, nphi-nghosts).

        Parameters
        -----------
        phi_1d : int / 1d array
                 The phi coordinate index or axis.

        Returns
        -------
        phi_1d : float or 1d array
                 The coordinate(s) :math:`\\phi` on the sphere.

        """

        if not phi_index:
            phi_index = np.arange(self.nghosts, self.nphi - self.nghosts)

        return (
            (phi_index - self.nghosts)
            * 2
            * np.pi
            / (self.nphi - 2 * self.nghosts)
        )

    @property
    def meshgrid(self):
        """The (:math:`\\theta, \\phi)`: coordinate meshes.
        Excludes the ghost zones.

        Returns
        -------
        theta :	2d array
                The :math:`\\theta` coordinate matrix
                for vectorization.

        phi : 2d array
              The :math:`\\phi` coordinate matrix
              for vectorization.
        """

        theta, phi = np.meshgrid(self.theta_1d, self.phi_1d)

        return np.transpose(theta), np.transpose(phi)

    @property
    def integration_method(self):
        """The default integration method"""
        return self._integration_method

    @property
    def phi_1d_wrapped(self):
        "Returns 1d phi axis including the last element identified with the first"

        return self._phi_1d_wrapped

    @property
    def meshgrid_wrapped(self):
        "Return the meshgrid constructed from phi_1d_wrapped"

        return self._meshgrid_wrapped

    def create_wrapped_meshgrid(self):
        self._phi_1d_wrapped = np.array(
            list(self.phi_1d) + [self.phi_1d[-1] + self.dphi]
        )

        theta_grid_wrapped, phi_grid_wrapped = np.meshgrid(
            self.theta_1d, self.phi_1d_wrapped
        )
        self._meshgrid_wrapped = np.transpose(theta_grid_wrapped), np.transpose(
            phi_grid_wrapped
        )
