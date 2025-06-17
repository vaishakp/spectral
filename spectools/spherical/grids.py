/""" Classes to hold grid information


Classes
-------
UniformGrid : grid info.
              Stores information on the 2d grid in spherical polar coordinates.
GLGrid : grid info
         Stores a Gauss-Legendre type grid on a spherical surface.
"""

# from enum import property
import numpy as np
from spectools.spherical.abstract import SphericalGrid

# from numba import jit, njit
# import numba as nb
# from numba.experimental import jitclass


class UniformGridOld:
    """A class to store the theta-phi grid info."""

    def __init__(
        self,
        nphi=80,
        ntheta=41,
        nphimax=124,
        nthetamax=66,
        nghosts=2,
        integration_method="MP",
        grid_type="Uniform",
    ):

        # Number of gridpoints along phi direction including ghost points.
        self.nphi = nphi
        # Number of gridpoints along theta direction including ghost points.
        self.ntheta = ntheta
        # Total length of phi array used by ETK.
        self.nphimax = nphimax
        # Total length of theta array used by ETK.
        self.nthetamax = nthetamax
        # Number of ghost points in theta/phi direction.
        self.nghosts = nghosts
        # The default integration method
        self._integration_method = integration_method

        self._grid_type = grid_type

    @property
    def grid_type(self):
        return self._grid_type

    @property
    def npix(self):
        # Return the total number of pixels, including the ghost zones present at one iteration.
        return (self.ntheta) * (self.nphi)

    @property
    def npix_act(self):
        # Return the actual number of pixels, excluding the ghost zones present at one iteration.
        return (self.ntheta - 2 * self.nghosts) * (
            self.nphi - 2 * self.nghosts
        )

    @property
    def npix_max(self):
        # Return the (max) total number of pixels, including the ghost and buffer zones at one iteration.
        return (self.nthetamax) * (self.nphimax)

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
    def dtheta(self):
        # Return the coodinate spacing d\theta
        return np.pi / (self.ntheta - 2 * self.nghosts)

    @property
    def dphi(self):
        # Return the coordinate spacing d\phi
        return 2 * np.pi / (self.nphi - 2 * self.nghosts)

    @property
    def nbuffer(self):
        # Return the coordinate spacing d\phi
        return self.nthetamax - self.ntheta

    @property
    def shape(self):
        """Return the shape of the grif"""
        return (self.ntheta_act, self.nphi_act)

    @property
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

        if not theta_index:
            theta_index = np.arange(self.nghosts, self.ntheta - self.nghosts)

        return (
            (theta_index - self.nghosts + 0.5)
            * np.pi
            / (self.ntheta - 2 * self.nghosts)
        )

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

        # theta = np.zeros((self.ntheta_act, self.nphi_act))
        # phi	= np.zeros((self.ntheta_act, self.nphi_act))

        # for theta_index, theta_val in enumerate(self.theta_1d()):
        # 	for phi_index, phi_val in enumerate(self.phi_1d()):
        # 		theta[theta_index, phi_index] = theta_val
        # 		phi[theta_index, phi_index] = phi_val

        return np.transpose(theta), np.transpose(phi)

    @property
    def integration_method(self):
        """The default integration method"""
        return self._integration_method

    def to_GLGrid(self):
        """Find the highest resolution
        closest equivalent GL grid of this
        grid
        """

        theta_min = min(self.theta_1d)

        possibleL = 1

        Lfound_flag = False

        while Lfound_flag is False:

            infoGL = GLGrid(L=possibleL)

            theta_gl_min = min(infoGL.theta_1d)

            if theta_gl_min < theta_min:
                Lfound_flag = True
                Lmax = possibleL - 1

            possibleL += 1

        Lmax = min(Lmax, self.ntheta_act - 1)

        infoGL = GLGrid(L=Lmax)

        self.equivalent_GLGrid = infoGL

        return infoGL

    def get_data_on_GLGrid(self, func, infoGL=None):
        """Get the data on a GLGrid given data on
        the uniform grid.

        Parameters
        ----------
        func : 2darray
               The function to be interpolated
               onto the GLGrid
        infoGL : grid_info, optional
                 The GLGrid onto which the function
                 is to be interpolated. If not given,
                 then the closeset equivalent GL grid
                 to this instance of UniformGrid will
                 be found and used.

        Returns
        -------
        infoGL : grid_info
                 The GLGrid used for interpolation

        func_on_GLGrid : 2darray
                         The function `func`
                         values on the GLGrid
        """

        if infoGL is None:
            infoGL = self.to_GLGrid()

        theta_grid_gl, phi_grid_gl = infoGL.meshgrid


class UniformGrid(SphericalGrid):
    """A class to store the theta-phi grid info."""

    def __init__(
        self,
        nphi=80,
        ntheta=41,
        nphimax=124,
        nthetamax=66,
        nghosts=2,
        integration_method="MP",
        grid_type="Uniform",
    ):
        super().__init__(
            nphi=80,
            ntheta=41,
            nphimax=124,
            nthetamax=66,
            nghosts=2,
            integration_method="MP",
            grid_type="Uniform",
        )

        self._grid_type = grid_type
        self.create_wrapped_meshgrid()

    @property
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

        if not theta_index:
            theta_index = np.arange(self.nghosts, self.ntheta - self.nghosts)

        return (
            (theta_index - self.nghosts + 0.5)
            * np.pi
            / (self.ntheta - 2 * self.nghosts)
        )

    @property
    def dtheta(self):
        # Return the coodinate spacing d\theta
        return np.pi / (self.ntheta - 2 * self.nghosts)

    def to_GLGrid(self):
        """Find the highest resolution
        closest equivalent GL grid of this
        grid
        """

        theta_min = min(self.theta_1d)

        possibleL = 1

        Lfound_flag = False

        while Lfound_flag is False:

            infoGL = GLGrid(L=possibleL)

            theta_gl_min = min(infoGL.theta_1d)

            if theta_gl_min < theta_min:
                Lfound_flag = True
                Lmax = possibleL - 1

            possibleL += 1

        Lmax = min(Lmax, self.ntheta_act - 1)

        infoGL = GLGrid(L=Lmax)

        self.equivalent_GLGrid = infoGL

        return infoGL

    def get_data_on_GLGrid(self, func, infoGL=None):
        """Get the data on a GLGrid given data on
        the uniform grid.

        Parameters
        ----------
        func : 2darray
               The function to be interpolated
               onto the GLGrid
        infoGL : grid_info, optional
                 The GLGrid onto which the function
                 is to be interpolated. If not given,
                 then the closeset equivalent GL grid
                 to this instance of UniformGrid will
                 be found and used.

        Returns
        -------
        infoGL : grid_info
                 The GLGrid used for interpolation

        func_on_GLGrid : 2darray
                         The function `func`
                         values on the GLGrid
        """

        if infoGL is None:
            infoGL = self.to_GLGrid()

        theta_grid_gl, phi_grid_gl = infoGL.meshgrid


class GLGridOld:
    """A class to store the coordinate grid on a sphere.

    Attributes
    ----------
    ntheta : int
             The number of angular points in the :math:`\\theta`
             direction, including ghost zones.
    nphi : int
           The number of angular points in the :math:`\\phi`
           direction, including ghost zones.
    nghosts : int
              The number of ghost zones at the end of
              each direction.
    meshgrid : tuple of 2d array
               The 2d array containing the meshgrid of
               (:math:`\\theta, \\phi`) angular points.
    theta_1d : 1d array
               The 1d array of angular points
               along the :math:`\\theta` axis.
    phi_1d : 1d array
             The 1d array of angular points
             along the :math:`\\phi` axis.
    dtheta : float
             The angular step size in the :math:`\\theta`
             direction.
    dphi : float
           The angular step size inthe :math:`\\phi`
           direction.
    npix_act : int
               The total number of gridpoints on the sphere,
               excluding the ghost points.
    meshgrid : tuple of 2darray
               Get the 2d angular grid.

    Methods
    -------
    theta_1d :
        Get the :math:`\\theta` axis.
    phi_1d :
        Get the :math:`\\phi` axis.

    Notes
    -----
    The total number of points on the sphere
    is assumed to be :math:`2 (L+1)^2`

    :math:`N_\\theta = L+1`

    :math:`N_\\phi = 2(L+1)`

    This integrates out spherical harmonics of degree L exactly,
    given a regular function on the sphere.

    In other words, given :math:`L+1` points in the :math:`\\theta`
    direction, one can resolve spherical harmonics upto degree
    :math:`L`.
    """

    def __init__(
        self,
        nphi=None,
        ntheta=None,
        nphi_act=None,
        ntheta_act=None,
        L=47,
        nghosts=2,
        integration_method="GL",
        grid_type="GL",
    ):

        # Number of gridpoints along phi direction including ghost points.
        self._nphi = nphi
        # Number of gridpoints along theta direction including ghost points.
        self._ntheta = ntheta
        self._nphi_act = nphi_act
        self._ntheta_act = ntheta_act
        # Total length of phi array used by ETK.
        # self._nphimax		= nphimax
        # Total length of theta array used by ETK.
        # self._nthetamax	= nthetamax
        # Number of ghost points in theta/phi direction.
        self._nghosts = nghosts
        self._L = L
        self._theta_1d = None
        self._phi_1d = None
        self._meshgrid = None
        self._integration_method = integration_method
        self._grid_type = grid_type

        if self._ntheta is None:
            if self._L is None:
                raise ValueError("Please specify L or angular points!")

            else:
                self._ntheta_act = L + 1
                self._nphi_act = 2 * self._ntheta_act

                self._ntheta = self._ntheta_act + 2 * self._nghosts
                self._nphi = self._nphi_act + 2 * self._nghosts

        elif self._L is None:
            if self.nthetha is None:
                raise ValueError("Please specify L or angular points!")
            else:
                self.L = self.ntheta_act - 1
                # assert(nphi%2==0)
                self.nphi_act = 2 * self.L + 1

        from scipy.special import roots_legendre

        cpoints, self._weights, self._sum_of_weights = roots_legendre(
            L + 1, mu=True
        )

        # xpoints = (np.pi-np.arccos(cpoints))
        xpoints = np.arccos(cpoints[::-1])
        self._theta_1d = xpoints

        dphi = 2 * np.pi / self._nphi_act

        self._phi_1d = np.linspace(0, 2 * np.pi - dphi, self._nphi_act)

        theta_grid, phi_grid = np.meshgrid(self._theta_1d, self._phi_1d)
        self._meshgrid = np.transpose(theta_grid), np.transpose(phi_grid)

        dtheta_axis = np.diff(self._theta_1d)

        dtheta_axis = np.append(dtheta_axis, np.pi - self._theta_1d[-1])
        dtheta_axis = np.insert(dtheta_axis, 0, self._theta_1d[0])

        self._dtheta_1d = dtheta_axis

        self._dphi = dphi  # self._phi_1d[1]

        self.create_wrapped_meshgrid()

    @property
    def grid_type(self):

        return self._grid_type

    def nphi(self):
        """Return the total number of gridpoints
        along the phi direction, including
        the ghost zones at one iteration."""
        if not self._nphi:
            return self._nphi_act + self.nghosts
        else:
            return self._nphi

    def ntheta(self):
        """Return the total number gridpoints
        along the theta direction, including
        the ghost zones at one iteration."""
        if not self._ntheta:
            return self._ntheta_act + self.nghosts
        else:
            return self._ntheta

    @property
    def nghosts(self):
        """Return the number of ghost zones
        at each end in each direction"""
        return self._nghosts

    @property
    def ntheta_act(self):
        """Return the actual number of physical data points
        along the theta direction excluding the ghost and
        buffer zones at one iteration."""
        if not self._ntheta_act:
            return self.ntheta - 2 * self.nghosts
        else:
            return self._ntheta_act

    @property
    def nphi_act(self):
        """Return the actual number of physical data points
        along the phi direction excluding the ghost and
        buffer zones at one iteration."""

        if not self._ntheta_act:
            return self.nphi - 2 * self.nghosts
        else:
            return self._nphi_act

    @property
    def npix_act(self):
        """Return the actual number of pixels,
        excluding the ghost zones present
        at one iteration"""
        return (self.ntheta_act) * (self.nphi_act)

    @property
    def dtheta_1d(self):
        """Return the non-uniform angular stepping in
        :math:`\theta` direction"""
        return self._dtheta_1d

    @property
    def dphi(self):
        """Return the uniform angular stepping in :math:`\\phi` direction"""
        return self._dphi

    @property
    def L(self):
        """Return the total number of pixels, including
        the ghost zones present at one iteration."""
        return self._L

    @property
    def npix(self):
        """Return the total number of pixels, including
        the ghost zones present at one iteration."""
        return (self.ntheta) * (self.nphi)

    @property
    def weights(self):
        """Return the integration weights along theta"""
        return self._weights

    @property
    def weights_grid(self):
        """Return the integration weights on the coordinate mesh grid"""
        return np.outer(self.weights, np.ones(self.nphi_act))

    @property
    def shape(self):
        """Return the shape of the grif"""
        return self.weights_grid.shape

    @property
    def theta_1d(self):
        """Returns the coordinate value theta given the coordinate index.
        The coordinate index ranges from (0, ntheta). The actual indices
        without the ghost and extra zones is (nghosts, ntheta-nghosts).

        Parameters
        -----------
        theta_index : int/ 1d array
                      The theta coordinate index or axis.

        Returns
        -------
        theta_1d : float
                   The coordinate(s) :math:`\\theta` on the sphere.
        """

        return self._theta_1d

    @property
    def phi_1d(self):
        """Returns the coordinate value theta given the coordinate index.
        The coordinate index lies in (0, nphi). The actual indices without
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

        return self._phi_1d

    @property
    def meshgrid(self):
        """The (:math:`\\theta, \\phi)`: coordinate meshes.
        Excludes the ghost zones.


        Returns
        -------
        theta :	2d array
                The :math:`\\theta` coordinate matrix for vectorization.

        phi : 2d array
              The :math:`\\phi` coordinate matrix for vectorization.
        """

        return self._meshgrid

    @property
    def integration_method(self):
        """The default integration method"""
        return self._integration_method


class GLGrid(SphericalGrid):
    """A class to store the coordinate grid on a sphere.

    Attributes
    ----------
    ntheta : int
             The number of angular points in the :math:`\\theta`
             direction, including ghost zones.
    nphi : int
           The number of angular points in the :math:`\\phi`
           direction, including ghost zones.
    nghosts : int
              The number of ghost zones at the end of
              each direction.
    meshgrid : tuple of 2d array
               The 2d array containing the meshgrid of
               (:math:`\\theta, \\phi`) angular points.
    theta_1d : 1d array
               The 1d array of angular points
               along the :math:`\\theta` axis.
    phi_1d : 1d array
             The 1d array of angular points
             along the :math:`\\phi` axis.
    dtheta : float
             The angular step size in the :math:`\\theta`
             direction.
    dphi : float
           The angular step size inthe :math:`\\phi`
           direction.
    npix_act : int
               The total number of gridpoints on the sphere,
               excluding the ghost points.
    meshgrid : tuple of 2darray
               Get the 2d angular grid.

    Methods
    -------
    theta_1d :
        Get the :math:`\\theta` axis.
    phi_1d :
        Get the :math:`\\phi` axis.

    Notes
    -----
    The total number of points on the sphere
    is assumed to be :math:`2 (L+1)^2`

    :math:`N_\\theta = L+1`

    :math:`N_\\phi = 2(L+1)`

    This integrates out spherical harmonics of degree L exactly,
    given a regular function on the sphere.

    In other words, given :math:`L+1` points in the :math:`\\theta`
    direction, one can resolve spherical harmonics upto degree
    :math:`L`.
    """

    def __init__(
        self,
        nphi=None,
        ntheta=None,
        nphi_act=None,
        ntheta_act=None,
        L=15,
        integration_method="GL",
    ):

        super().__init__(
            nphi=nphi,
            ntheta=ntheta,
            nghosts=0,
            integration_method=integration_method,
            grid_type="GL",
        )

        # Total length of phi array used by ETK.
        # self._nphimax		= nphimax
        # Total length of theta array used by ETK.
        # self._nthetamax	= nthetamax
        # Number of ghost points in theta/phi direction.
        self._L = L
        self._theta_1d = None
        self._phi_1d = None
        self._meshgrid = None
        self._integration_method = integration_method
        self._grid_type = "GL"

        if self._ntheta is None:
            if self._L is None:
                raise ValueError("Please specify L or angular points!")

            else:
                self._ntheta_act = L + 1
                self._nphi_act = 2 * self._ntheta_act

                self._ntheta = self._ntheta_act + 2 * self._nghosts
                self._nphi = self._nphi_act + 2 * self._nghosts

        elif self._L is None:
            if self.nthetha is None:
                raise ValueError("Please specify L or angular points!")
            else:
                self.L = self.ntheta_act - 1
                # assert(nphi%2==0)
                self.nphi_act = 2 * self.L + 1

        from scipy.special import roots_legendre

        cpoints, self._weights, self._sum_of_weights = roots_legendre(
            L + 1, mu=True
        )

        # xpoints = (np.pi-np.arccos(cpoints))
        xpoints = np.arccos(cpoints[::-1])
        self._theta_1d = xpoints

        dphi = 2 * np.pi / self._nphi_act

        self._phi_1d = np.linspace(0, 2 * np.pi - dphi, self._nphi_act)

        theta_grid, phi_grid = np.meshgrid(self._theta_1d, self._phi_1d)
        self._meshgrid = np.transpose(theta_grid), np.transpose(phi_grid)

        dtheta_axis = np.diff(self._theta_1d)

        dtheta_axis = np.append(dtheta_axis, np.pi - self._theta_1d[-1])
        dtheta_axis = np.insert(dtheta_axis, 0, self._theta_1d[0])

        self._dtheta_1d = dtheta_axis

        self._dphi = dphi  # self._phi_1d[1]

        self.create_wrapped_meshgrid()

    @property
    def dtheta(self):
        """Return the non-uniform angular stepping in
        :math:`\theta` direction"""
        return self._dtheta

    @property
    def L(self):
        """Return the total number of pixels, including
        the ghost zones present at one iteration."""
        return self._L

    @property
    def weights(self):
        """Return the integration weights along theta"""
        return self._weights

    @property
    def weights_grid(self):
        """Return the integration weights on the coordinate mesh grid"""
        return np.outer(self.weights, np.ones(self.nphi_act))

    @property
    def shape(self):
        """Return the shape of the grif"""
        return self.weights_grid.shape

    @property
    def theta_1d(self):
        """Returns the coordinate value theta given the coordinate index.
        The coordinate index ranges from (0, ntheta). The actual indices
        without the ghost and extra zones is (nghosts, ntheta-nghosts).

        Parameters
        -----------
        theta_index : int/ 1d array
                      The theta coordinate index or axis.

        Returns
        -------
        theta_1d : float
                   The coordinate(s) :math:`\\theta` on the sphere.
        """

        return self._theta_1d

    @property
    def phi_1d(self):
        """Returns the coordinate value theta given the coordinate index.
        The coordinate index lies in (0, nphi). The actual indices without
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

        return self._phi_1d

    @property
    def meshgrid(self):
        """The (:math:`\\theta, \\phi)`: coordinate meshes.
        Excludes the ghost zones.


        Returns
        -------
        theta :	2d array
                The :math:`\\theta` coordinate matrix for vectorization.

        phi : 2d array
              The :math:`\\phi` coordinate matrix for vectorization.
        """

        return self._meshgrid
