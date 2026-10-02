"""Abstract base class for grids on the sphere.

Defines :class:`SphericalGrid`, the interface shared by every angular grid
in :mod:`spectools.spherical.grids`. It fixes the bookkeeping that is
common to all of them -- grid extents, ghost and buffer zones, pixel counts
and coordinate spacings -- and leaves the actual coordinate placement to
the concrete subclasses.

Grid arrays carry two kinds of padding, inherited from the Einstein Toolkit
conventions this originally interoperated with:

ghost zones
    ``nghosts`` cells at each edge in each direction, included in
    ``ntheta`` and ``nphi`` but not part of the physical domain. The
    ``_act`` properties report the counts with these excluded.
buffer zones
    The difference between the allocated extents ``ntheta_max`` /
    ``nphi_max`` and the extents actually in use. See :attr:`nbuffer`.

Subclasses must implement :attr:`dtheta` and :attr:`theta_1d`.

Requires the ``spherical`` extra.
"""

from abc import ABC, abstractmethod
import numpy as np


class SphericalGrid(ABC):
    """An abstract class for spherical grids.

    Parameters
    ----------
    nphi : int, optional
        Gridpoints along :math:`\\phi`, including ghost points.
    ntheta : int, optional
        Gridpoints along :math:`\\theta`, including ghost points.
    nphimax, nthetamax : int, optional
        Total allocated array lengths along each direction, including
        buffer zones.
    nghosts : int, optional
        Ghost points at each edge, in each direction.
    integration_method : str, optional
        Default quadrature rule for integrals over this grid, e.g.
        ``"MP"`` for the midpoint rule.
    grid_type : str, optional
        Label identifying the concrete grid, e.g. ``"GL"`` or
        ``"Uniform"``.
    """

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
        """str: Label identifying the concrete grid, e.g. ``'GL'``."""
        return self._grid_type

    @property
    def nphi(self):
        """int: Gridpoints along :math:`\\phi`, including ghost points."""
        return self._nphi

    @property
    def ntheta(self):
        """int: Gridpoints along :math:`\\theta`, including ghost points."""
        return self._ntheta

    @property
    def nghosts(self):
        """int: Ghost points at each edge, in each direction."""
        return self._nghosts

    @property
    def ntheta_max(self):
        """int: Allocated array length along :math:`\\theta`, with buffers."""
        return self._ntheta_max

    @property
    def nphi_max(self):
        """int: Allocated array length along :math:`\\phi`, with buffers."""
        return self._nphi_max

    @property
    def npix(self):
        """int: Total pixels at one iteration, ghost zones included."""
        return (self.ntheta) * (self.nphi)

    @property
    def npix_act(self):
        """int: Pixels in the physical domain, ghost zones excluded."""
        return (self.ntheta - 2 * self.nghosts) * (
            self.nphi - 2 * self.nghosts
        )

    @property
    def npix_max(self):
        """int: Total allocated pixels, ghost and buffer zones included."""
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
        """float: Coordinate spacing :math:`d\\theta`.

        Abstract: each grid places its polar points differently, so the
        spacing is defined by the concrete subclass.
        """
        pass

    @property
    def dphi(self):
        """float: Coordinate spacing :math:`d\\phi`.

        Uniform in all grids here, since :math:`\\phi` is always
        equispaced: ``2 pi`` divided by the number of physical points.
        """
        return 2 * np.pi / (self.nphi - 2 * self.nghosts)

    @property
    def nbuffer(self):
        """int: Buffer cells along :math:`\\theta`, excluding ghost zones.

        The unused remainder of the allocated array,
        ``ntheta_max - ntheta``.
        """
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
        """Build the periodically wrapped ``(theta, phi)`` meshgrid.

        Appends one extra :math:`\\phi` point, one ``dphi`` past the last,
        so that the azimuthal direction closes on itself. Plotting and
        interpolation routines need this duplicated seam; quadrature does
        not, and should use the unwrapped ``meshgrid`` instead.

        Sets ``_phi_1d_wrapped`` and ``_meshgrid_wrapped`` on the
        instance, exposed through :attr:`phi_1d_wrapped` and
        :attr:`meshgrid_wrapped`. Returns nothing.
        """
        self._phi_1d_wrapped = np.array(
            list(self.phi_1d) + [self.phi_1d[-1] + self.dphi]
        )

        theta_grid_wrapped, phi_grid_wrapped = np.meshgrid(
            self.theta_1d, self.phi_1d_wrapped
        )
        self._meshgrid_wrapped = np.transpose(
            theta_grid_wrapped
        ), np.transpose(phi_grid_wrapped)
