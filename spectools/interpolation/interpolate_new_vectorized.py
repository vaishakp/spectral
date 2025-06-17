from mpi4py import MPI
import numpy as np

from parallellib.parallel_mpi import ParallelClassTemplate
from waveformtools.waveformtools import flatten_3l, message, unsort
from spectools.spherical.grids import GLGrid
from waveformtools.diagnostics import method_info

from spectools.chebyshev.chebyshev import ChebyshevSpectral
from waveformtools.single_mode import SingleMode
from spectools.spherical.transforms import SHExpand, Yslm_vec
from waveformtools.waveforms import modes_array

import pickle

from qlmtools.sxs.transforms import (
    ToSphericalPolar,
    ReorganizeCoords,
    RContract,
    get_radial_clms,
    AngContract,
)

import time


class Interpolate3D(ParallelClassTemplate):
    """Interpolate any field on a spectral grid of
    type (GaussLegendre X Chebyshev) onto a requested
    set of points.

    This uses MPI to parallelize the calculations.

    Attributes
    ----------
    r_min, r_max : float
                   The radii of the smallest and
                   the largest spherical shells
                   in the Volume. These are the
                   endpoints of the Chebyshev
                   radial grid in the
                   physical space.

    coord_centres : list, optional
                    A list containing sets of
                    the three coordinate values of
                    the location of the geometric
                    centroid of the surface
                    onto which the data is
                    being interpolated.

                    If not supplied, then it is assumed
                    that they are zeros i.e. [[0, 0, 0], ...].

    sphp_output_grid : list, optional
                       A list containing sets of three sublists:
                       i.e. [[R, Theta, Phi],...] of the spherical polar
                       coordinates / mesh of
                       the interpolation surface.

                       If not specified, then `cart_output_grid`
                       is used.

    cart_output_grid : list/4darray, optional
                       A list containing three sublists:
                       i.e. [[X, Y, Z],...]of the cartesian
                       coordinates / mesh of
                       the interpolation surface.
                       This can also be a 4d array of the
                       shape (time, x, y, z)

                       If not specified, then `sphp_output_grid`
                       is used.

    raw_data        : ndarray
                      The array with the same
                      shape as that of the coordinates
                      containing the data to be interpolated.

    Notes
    -----
    Decoposition strategy
    1. Assumes the same number of radial and angular collocation points
       across all time steps.
    2. The data from shells from all time steps are pooled together before angular expansion happens
    3. Each shell is labelled with r and t
    4. Angular decompositions are fully vectorized over r and t i.e. one shell for
       each r and t is distributed to one MPI worker.
    4. After angular expansion, one has C_lmrt
    5. C_lm modes on all lines of sight in r and from across time steps are pooled together.
    6. Radial decompositions of the angular modes on all rays (over lm and time) are vectorized.
       i.e. C_lmrt -> Clmpt (radial physical to spectral) are fully vectorized over r and t.
    7. If needed, interpolation over time can be carried out using a spline interpolation.
    8. Only evaluations are parallelized.


    Evaluation strategy
    1. The spectral eval parameter space is (t, r, theta, phi)
    2. Evaluations can be distributed to MPI processors individually.
    3. Evaluations are MPI parallelized within each time step but not across
    it. So evaluations within a given time step are carried out by grouping coordinates
    having multiple angular positions at a single shell. Evaluations are vectorized
    within a shell/coordianate group but MPI parallelized across ccordinate groups.

    """

    def __init__(
        self,
        r_min,
        r_max,
        coord_centers=None,
        sphp_output_grid=None,
        cart_output_grid=None,
        raw_data=None,
        saved_interpolant_file=None,
        label="func",
        time_axis=None,
    ):
        super().__init__()

        # User input quantities
        self._cart_output_grid = cart_output_grid
        self._sphp_output_grid = sphp_output_grid
        self._time_axis = time_axis
        raw_data_shape = np.array(raw_data).shape

        if len(raw_data_shape) == 4:
            message(
                "The raw data has a time axis."
                "Assuming the leading axis"
                "to be the time axis"
            )

        elif len(raw_data_shape) == 3:
            message("The raw data is specified at a" "single time slice")

            raw_data = np.array(
                [
                    raw_data,
                ]
            )

        # self._output_grid = output_grid
        self._raw_data = np.array(raw_data)

        self._coord_centers = coord_centers
        self._r_min = r_min
        self._r_max = r_max
        self._label = label

        # Derived quantities
        self._shape = None
        self._interpolant = None
        self._interpolated_data = None
        self._input_ang_grid = None
        self._radial_grid = None
        self._method_info = None
        self._input_grid = None
        self._ang_args_order = None
        self._Ylm_cache = None
        self._local_Ylm_data_cache_dict = {}
        self._saved_interpolant_file = saved_interpolant_file
        self._axis_rotation_angles = None

        # Setup engine
        self.initialize_parallel_engine()
        self.pre_setup()

    @property
    def shape(self):
        """The shape of input raw data.
        The first axis is assumed to be the time axis,
        followed by x,y,z"""

        if (np.array(self._shape) == np.array(None)).all():
            self._shape = np.array(self.raw_data).shape

        return self._shape

    @property
    def ntime(self):
        return self.shape[0]

    @property
    def time_axis(self):
        """The time axis of the data set"""
        return self._time_axis

    @property
    def label(self):
        """A label for the run/dataset"""
        return self._label

    @property
    def coord_centers(self):
        """The coordinate center location
        of the coordinate mesh"""
        return self._coord_centers

    @property
    def sphp_output_grid(self):
        """The output coordinate grid
        in spherical polar coordinates
        used during interpolation"""
        return self._sphp_output_grid

    @property
    def cart_output_grid(self):
        """The output coordinate grid
        in cartesian coordinates
        for interpolation"""

        return self._cart_output_grid

    @property
    def r_min(self):
        """The radius of the smallest shell present
        in the input grid"""

        if self._r_min is None:
            raise KeyError("Please provide r_min of the input grid")

        return self._r_min

    @property
    def r_max(self):
        """The radius of the largest radial shell
        present in the input grid"""
        if self._r_max is None:
            raise KeyError("Please provide r_max of the input grid")

        return self._r_max

    @property
    def output_grid(self):
        """The output coordinate grid interpolated onto"""

        return self._output_grid

    @property
    def raw_data(self):
        """The data to be interpolated, in a specified
        coordinate system. The coordinate system is specified
         at xxx"""

        return self._raw_data

    @property
    def interpolant(self):
        """The unevaluated spectral interpolant"""
        return self._interpolant

    @property
    def interpolated_data(self):
        """The data interpolated onto the
        output cartesian grid"""
        return self._interpolated_data

    @property
    def input_ang_grid(self):
        """The angular grid of the input raw data.
        This is an object of the Gauss-Legendre grid
        class"""
        return self._input_ang_grid

    @property
    def input_grid(self):
        """A proxy to the combined (radial, angular) input grid"""
        return self._input_grid

    @property
    def radial_grid(self):
        """The radial collocation points of the
        input grid. These are the Gauss-Lobatto points"""
        return self._radial_grid

    @property
    def method_info(self):
        """The information class for methods used in spectral
        operations"""
        return self._method_info

    @property
    def ang_args_order(self):
        """A list contining the job ids/ indices of the ordered
        data"""
        return self._ang_args_order

    @property
    def coords_groups_list(self):
        """A list contining groups of coordinates. Each group is
        usually used by one MPI worker for evaluation routine ."""
        return self._coords_groups_list

    @property
    def num_coord_groups(self):
        """The number of groups in the coordinate group list. This is
        equal to the maximum number of MPI jobs that can run
        in parallel"""
        return len(self.coords_groups_list)

    @property
    def saved_interpolant_file(self):
        """Path to the saved interpolant"""
        return self._saved_interpolant_file

    @property
    def Ylm_cache(self):
        """The cached Ylm data for quick organized evaluation
        of the interpolant.

        In organized evaluation, the points to be interpolated onto
        lie along the same angular rays.

        Ylm_cache will be a 2 diemnsinal array
        """
        if np.array(self._Ylm_cache).any() == np.array(None):
            raise KeyError("Please create Ylm cache first!!")

        return self._Ylm_cache

    @property
    def axis_rotation_angles(self):
        """The rotation angles of the line joining the BHs.
        This will be a 1D array along time"""

        return self._axis_rotation_angles

    ####################################
    # Initialize
    ####################################

    def pre_setup(self):
        """Setup the angluar, radial spectral grid and
        associated expansion parameters"""

        # Input coordinate grid setup
        n_time, n_radii, n_theta, n_phi = self.shape

        self.message_root(
            f"Number of time steps {n_time}\n"
            f"Number of radial shells {n_radii}\n"
            f"Angular grid shape {n_theta} x {n_phi}"
        )

        self.message_root("Num of radial points", n_radii, message_verbosity=2)

        self._input_ang_grid = GLGrid(L=n_theta - 1)

        self.message_root(
            "L grid of expansion", self.input_ang_grid.L, message_verbosity=2
        )

        self._method_info = method_info(
            ell_max=self.input_ang_grid.L, int_method="GL"
        )

        self.message_root(
            f"method info ell max {self.method_info.ell_max}",
            message_verbosity=3,
        )

        self._radial_grid = ChebyshevSpectral(
            a=self.r_min, b=self.r_max, Nfuncs=self.shape[1]
        )

        self.message_root(
            "Created Chebyshev radial grid"
            f"with Nfuncs {self.radial_grid.Nfuncs}\n",
            f"Shape of radial collocation points \
                {self.radial_grid.collocation_points_logical.shape}",
            message_verbosity=3,
        )

        self._input_grid = [self.radial_grid, self.input_ang_grid]

        # Output coordinate grid setup
        if (np.array(self.sphp_output_grid) == np.array(None)).all():

            if np.array(self.cart_output_grid).all() == np.array(None):
                raise KeyError(
                    "Please provide the input grid"
                    "in either cartesian"
                    "or spherical polar coordinates"
                )
            else:
                self.message_root(
                    "Reaing in Cart3D Output grid...", message_verbosity=2
                )

                sphp_output_grid = []

                self.message_root(
                    "Transforming to spherical polar coordinates",
                    message_verbosity=2,
                )

                for t_step in range(n_time):
                    # Transform cart to spherical
                    X, Y, Z = self.cart_output_grid[t_step]

                    # Check if cood centres are given. If not
                    # initialize
                    if (np.array(self.coord_centers) == np.array(None)).all():
                        self._coord_centers = []  # [0, 0, 0]

                    # Check if coord centres are specified at each time
                    # step. If not assume them to be the initial value
                    # i.e. at the origin
                    if len(self.coord_centers) < (t_step + 1):
                        coord_centers = [0, 0, 0]
                        self._coord_centers.append(coord_centers)

                    else:
                        coord_centers = self.coord_centers[t_step]

                    # xcom, ycom, zcom = self.coord_centers
                    # Transform the cartesian output grid to
                    # spherical polar
                    sphp_output_grid.append(
                        ToSphericalPolar([X, Y, Z], coord_centers)
                    )

            self._sphp_output_grid = sphp_output_grid

        else:
            self.message_root(
                "Reading in SPHP Output Grid...", message_verbosity=2
            )

        assert len(self.sphp_output_grid) == n_time, (
            "The output grid must be"
            "specified at all time steps the input raw data"
            "is specified at"
        )

        self.message_root(
            "The output grid shape is ",
            self.sphp_output_grid[0][0].shape,
            message_verbosity=2,
        )

        self.compute_rotation_angles()
        self.align_coordinate_system()

    def initialize_interpolant(self):
        """Initialize a modes array to hold the spectral
        interpolant object"""

        self.message_root("Initializing the interpolant object...")

        if (np.array(self._interpolant) != np.array(None)).all():
            raise ValueError("The interpolant has already been initialized ! ")

        Clmrt = modes_array(
            ell_max=self._method_info.ell_max,
            extra_mode_axis_len=self.shape[1],
            data_len=self.ntime,
            spin_weight=0,
            time_axis=self.time_axis,
        )

        Clmrt.create_modes_array(
            ell_max=self.method_info.ell_max,
        )

        self._interpolant = Clmrt

    def compute_rotation_angles(self):
        """Find the optimum rotation angle
        to align the canonical coordinate system
        against the SpEC simulation coordinates"""

        phi_rotation_angles = []

        # ntheta_ah, nphi_ah = self.sphp_output_grid[0][1].shape
        # theta_axis_1d = self.sphp_output_grid[t_index][1][:, 0]
        # theta_equator_index = np.argmin(abs(theta_axis_1d - np.pi/2))
        # phi_rotation_angles =
        for t_index in range(self.ntime):
            _, Theta, Phi = self.sphp_output_grid[t_index]
            theta_axis_1d = self.sphp_output_grid[t_index][2][:, 0]
            theta_equator_index = np.argmin(abs(theta_axis_1d - np.pi / 2))
            phi_rotation_angles.append(Phi[theta_equator_index, 0])

        self._axis_rotation_angles = np.array(phi_rotation_angles)

    def align_coordinate_system(self):
        """Transform the canonical coordinate
        system to align with the simulation coordinates
        by applying the optimum rotation"""

        # Rotate the coordinates to match with SpEC's
        for t_index in range(self.ntime):

            rotated_phi_output_grid = (
                self._sphp_output_grid[t_index][2]
                - self.axis_rotation_angles[t_index]
            )

            self._sphp_output_grid[t_index][2] = rotated_phi_output_grid

    def dealign_coordinate_system(self):
        """Transform the canonical coordinate
        system to align with the simulation coordinates
        by applying the optimum rotation"""

        # Rotate the coordinates to match with SpEC's
        for t_index in range(self.ntime):

            derotated_phi_output_grid = (
                self._sphp_output_grid[t_index][2]
                + self.axis_rotation_angles[t_index]
            )

            self._sphp_output_grid[t_index][2] = derotated_phi_output_grid

    ################################################
    # Expansions
    ################################################

    def angular_expansion_at_r_index(self, r_index):
        """Expand the angular data at one radial
        collocation point (shell) in SH"""

        ang_data = self.raw_data[r_index, :, :]

        local_one_set_modes = SHExpand(
            func=ang_data,
            method_info=self.method_info,
            info=self.input_ang_grid,
        )

        return local_one_set_modes

    def angular_expansion_at_t_and_r_index(self, t_step, r_index):
        """Expand the angular data at one radial
        collocation point at one time step in SH"""

        self.message_root(
            "t_step inside ang exp at t and r index",
            t_step,
            message_verbosity=4,
        )
        self.message_root(
            "r_index inside ang exp at t and r index",
            r_index,
            message_verbosity=4,
        )

        ang_data = self.raw_data[t_step, r_index, :, :]

        local_one_set_modes = SHExpand(
            func=ang_data,
            method_info=self.method_info,
            info=self.input_ang_grid,
        )

        return local_one_set_modes

    def radial_expansion_at_ell_emm(self, ell, emm):
        """Carry out radial expansion of Clm modes with chebyshev
        polynomials"""

        this_r_modes_local = get_radial_clms(self._modes_r_ordered, ell, emm)

        this_Clmr = self.radial_grid.MatrixPhysToSpec @ np.array(
            this_r_modes_local
        )

        message(f" This Clmr l{ell}, m{emm}", this_Clmr, message_verbosity=4)

        return this_Clmr

    def radial_expansion_at_t_and_ell_emm(self, t_step, ell, emm):
        """Carry out radial expansion of Clm modes with Chebyshev polynomials
        along a single ray at a single instant of time"""

        # Fetch Clm modes at all radial shells for the requested
        # t_step, ell and emm
        this_r_modes_local = get_radial_clms(
            self._reordered_Clm_modes_t_r_flat_list[t_step], ell, emm
        )

        this_Clmr = self.radial_grid.MatrixPhysToSpec @ np.array(
            this_r_modes_local
        )

        message(f" This Clmr l{ell}, m{emm}", this_Clmr, message_verbosity=4)

        return this_Clmr

    def radial_decompose(self):
        """Carry out the radial decomposition of the
        Clm s at all t steps"""

        n_time = self.ntime

        local_radial_decomp_list = []

        for t_step in range(n_time):
            if t_step % self.mpi_nprocs == self.mpi_rank:

                single_step_spectrum = self.get_radial_spectrum_vec(t_step)

                local_radial_decomp_list.append([t_step, single_step_spectrum])

        self.mpi_comm.Barrier()

        full_Clmr_t_modes_data_group_list = self.mpi_comm.gather(
            local_radial_decomp_list, root=0
        )

        if self.mpi_rank == 0:

            full_Clmr_t_modes_data = np.array(
                self.reorganize_mpi_job_output(
                    full_Clmr_t_modes_data_group_list
                ),
                dtype=np.complex128,
            )

            # full_Clmr_t_modes_data = np.array([item[1] for item in full_Clmr_t_modes_data_flat_list], dtype=np.complex128)

            message(
                "Full Clmr_t_modes_data shape", full_Clmr_t_modes_data.shape
            )

            self.assign_modes_data_to_modes_array(full_Clmr_t_modes_data)

    def AngContractVec(self, Clm_interp, ang_jobid):
        """Carry out transformation from angular spectral
        space to physical space i.e. evaluate a set of
        SH modes at a given angle. The angle is given
        by the jobid, which is basically the element number
        in the zipped list of the flattened angular corrdinates
        of the output grid (Theta, Phi)"""

        self.message_root(
            "Clm interp modes shape",
            Clm_interp._modes_data.shape,
            message_verbosity=4,
        )

        Ylm_cached_1d_array = self.get_cached_Ylm(ang_jobid)

        Clm_modes_values = Clm_interp._modes_data

        message(
            "Clm modes values shape",
            Clm_modes_values.shape,
            message_verbosity=4,
        )
        message(
            "Ylm_raw_data_array shape",
            Ylm_cached_1d_array.shape,
            message_verbosity=4,
        )

        func_value = np.sum(Ylm_cached_1d_array * Clm_modes_values)

        message("Func vals shape", func_value.shape, message_verbosity=4)
        return func_value

    ###########################################################
    # Re organization
    ###########################################################

    def flatten_coords(self, t_step):
        """Flatten the 2d coordinate arrays into 1d lists
        and zip them."""
        Rf, Th, Ph = self.sphp_output_grid[t_step]

        Rf_list = np.array(Rf).flatten()
        Th_list = np.array(Th).flatten()
        Ph_list = np.array(Ph).flatten()

        coords_list = zip(Rf_list, Th_list, Ph_list)

        return coords_list

    def reorganize_coords(self):
        """Reorganize coords for evaluation of the interpolant.
        For every time step, this organizes the interpolation points
        into a bunch of lists. Each of the lists contains arrays of
        coords at the same radial position.

        Each list is composed of R_1d_array, theta_1d_array, phi_1d_array.

        If every points is at a different r like on an AH, then the
        each of the 1d lists has length one.
        """

        self.message_root("Reorganizing coords", message_verbosity=2)

        coords_groups_list = []
        ang_args_order_list = []

        for t_step in range(self.ntime):

            one_coords_groups_list, one_ang_args_order = ReorganizeCoords(
                self.sphp_output_grid[t_step]
            )

            coords_groups_list.append(one_coords_groups_list)
            ang_args_order_list.append(one_ang_args_order)

        self._coords_groups_list = coords_groups_list
        self._ang_args_order = ang_args_order_list

    def reorder_ang_modes_x_r_list(self, modes_r_set_group):
        """Reorder the Clm angular modes data from MPI workers
        and create a copy in each for later use.

        The input data will be grouped. Works only
        with 3d input data i.e. no time axis"""

        if self.mpi_rank == 0:
            modes_r_set = flatten_3l(modes_r_set_group)
            r_ind_set = [item[0] for item in modes_r_set]
            r_modes_order = np.argsort(r_ind_set)

            modes_r_ordered = [
                modes_r_set[index][1] for index in r_modes_order
            ]
        else:
            modes_r_ordered = None

        modes_r_ordered = self.mpi_comm.bcast(modes_r_ordered, root=0)

        self.message_root(
            "Synchronizing before assigining modes r ", message_verbosity=3
        )

        self.mpi_comm.Barrier()

        self.message_root(
            "Finished synchronizing before assigining modes r ",
            message_verbosity=3,
        )

        self._modes_r_ordered = modes_r_ordered

    def regroup_ang_modes_in_time(self, modes_r_t_set_group):
        """From the modes on every sphereical shell at all
        radial collocation points and all time steps, regoup
        them into sets of modes at different collocation points
        at separate instants of time.
        """

    def order_Clmrt_list_in_time(self, modes_Clmr_t_flat_list):
        """Order the segments in time. This takes in a flattened list
        of all segments at all times and angular coords.

        """
        # Order in time
        segment_numbers = [item[0] for item in modes_Clmr_t_flat_list]

        seg_order = np.argsort(segment_numbers)

        time_ordered_segments = np.array(modes_Clmr_t_flat_list)[seg_order]

        message(
            "Length of time ordered segments",
            len(time_ordered_segments),
            message_verbosity=2,
        )

        message(
            "Shape of one segment data",
            time_ordered_segments[0][3].shape,
            message_verbosity=3,
        )

        return time_ordered_segments

    def order_Clmr_list_in_ell_emm(self, this_seg_data):
        """Order elements in angles at a given
        segment. The input is a flattened list of
        all elements, at all times. Each element has
        three ids corresponding to the time and two angular
        indices. Returns a flattend list of the same
        reordered elements
        """

        ell_max = self.method_info.ell_max

        time_indices = [item[0] for item in this_seg_data]

        ell_indices = [item[1] for item in this_seg_data]
        ell_ind_order = np.argsort(ell_indices)

        ell_ordered_this_seg_data = np.array(this_seg_data)[ell_ind_order]

        ell_ind_ordered = [item[1] for item in ell_ordered_this_seg_data]

        ordered_data_segment = []

        for ell_ind in range(ell_max + 1):

            this_ell_data = [
                item
                for item in ell_ordered_this_seg_data
                if item[1] == ell_ind
            ]

            emm_indices = [item[2] for item in this_ell_data]

            emm_ind_order = np.argsort(emm_indices)

            emm_ordered_data = np.array(this_ell_data)[emm_ind_order]

            emm_indices_ordered = [item[2] for item in emm_ordered_data]

            emm_ordered_data = np.array(
                [this_ell_data[ind][3] for ind in emm_ind_order]
            )

            single_ordered_data = np.array(emm_ordered_data)

            ordered_data_segment.append(single_ordered_data)

        ordered_data_segment = np.array(ordered_data_segment)

        print(
            "Single time step reordered data shape", ordered_data_segment.shape
        )

        return ordered_data_segment

    def reorganize_mpi_job_output(self, job_output):

        job_output = flatten_3l(job_output)

        message(
            "Job output ele shape", job_output[0][1].shape, message_verbosity=2
        )

        jobids = [item[0] for item in job_output]

        order = np.argsort(jobids)

        job_vals_list_ordered = np.array(
            [item[1] for item in job_output], dtype=object
        )[order]

        message(
            "Reorg mpi job list shape",
            job_vals_list_ordered.shape,
            message_verbosity=4,
        )

        message(
            "One ele shape",
            job_vals_list_ordered[0].shape,
            message_verbosity=4,
        )

        return job_vals_list_ordered

    #####################################################################
    # Fetch
    #####################################################################

    def get_Clm_modes_at_t_step(self, t_step, Clm_modes_r_t_flat_list):
        """Fetch the Clm modes at all radial shells at
        respective collocation points at a given time step.

        The input data is the time index and the set of all modes
        at all times and radial shells (flattened)

        Please note that the `modes_r_t_set_list` is of the form
        [job_id, SingleMode]. Here job_id is t_step * n_radii + r_index

        """

        message(
            "modes r t set list", Clm_modes_r_t_flat_list, message_verbosity=4
        )

        n_radii = self.shape[1]

        message(f"Fectching all Clm modes at time step {t_step}")

        Clm_modes_r_list_at_given_t_step = [
            item
            for item in Clm_modes_r_t_flat_list
            if int(item[0] / n_radii) == t_step
        ]

        message(
            f"Length of the Clm modes at t step {t_step} is",
            len(Clm_modes_r_list_at_given_t_step),
            message_verbosity=3,
        )

        return Clm_modes_r_list_at_given_t_step

    def get_Clmr_modes_at_t_step(self, t_step, Clmr_modes_t_flat_list):
        """Fetch the Clmr modes at a given time step.

        The input data is the time index and the list of all 3d modes
        at all times.

        Please note that the `Clmr_modes_r_t_set_list` is of the form
        [t_step, ell, emm, a list of Clmr modes along the radial axis].

        Also, please note that this is different from the previous
        function `get_Clm_modes_at_t_step` which only deals with
        the 2D Clm modes on shells before radial decomposition
        placed at radial collocation points
        and not the 3D Clmr modes.
        """

        message(f"Fectching all Clmr modes at time step {t_step}")

        Clmr_modes_list_at_given_t_step = [
            item for item in Clmr_modes_t_flat_list if item[0] == t_step
        ]

        message(
            f"Length of the Clmr modes at t step {t_step} is",
            len(Clmr_modes_list_at_given_t_step),
        )

        return Clmr_modes_list_at_given_t_step

    def get_radial_spectrum_vec(self, t_step):
        """Get the radial spectum of Clm modes
        for a given t_step from the list of ordered
        single_modes obj"""

        this_t_Clm_r_modes = self._reordered_Clm_modes_t_r_flat_list[t_step]

        message(
            "Len of stacked Clm_r modes list ",
            len(this_t_Clm_r_modes),
            message_verbosity=2,
        )

        this_t_Clm_r_modes_data = np.array(
            [item._modes_data for item in this_t_Clm_r_modes]
        )

        message(
            f"This Clm_r modes data shape {this_t_Clm_r_modes_data.shape}",
            message_verbosity=2,
        )

        # this_Clmr = self.radial_grid.MatrixPhysToSpec @ np.array(this_t_Clm_r_modes_data).transpose(1, 2, 0)
        this_Clmr = np.einsum(
            "ij,jkl->ikl",
            self.radial_grid.MatrixPhysToSpec,
            np.array(this_t_Clm_r_modes_data),
        )

        message(f"This Clmr shape {this_Clmr.shape}", message_verbosity=2)

        return this_Clmr

    def get_ang_ind_from_ang_jobid(self, t_step, ang_jobid):
        """Get the angular indices from the job id and time step.

        This would be primarily used to fetch the cached Ylm values
        on a point on a sphere."""

        ntheta_ah, nphi_ah = self.sphp_output_grid[t_step][1].shape

        theta_index = int(ang_jobid / nphi_ah)
        phi_index = ang_jobid % nphi_ah

        return theta_index, phi_index

    def get_cached_Ylm(self, ang_jobid):
        """For scattered evaluation, get all the values of Ylm
        correspondig to

        1. all ell, m
        2. one time step (implicit).
        3. particular angular coordinates corresponding to
        ang_jobid.

        at angular coordinate of the interpolation point
        specified by ang_jobid.

        ang_jobid is the index of the flattened coordinate
        array at the respective time step.
        """

        cached_chunk = self.Ylm_cache._modes_data[:, :, ang_jobid]

        message("Cached chunk shape", cached_chunk.shape, message_verbosity=4)
        return cached_chunk

    def get_number_from_ell_emm(self, ell, emm):
        number = (ell + 1) * (ell - 1) + emm + ell + 1

        return number

    def get_ell_emm_from_number(self, number):
        ell = int(np.sqrt(number))

        emm = number - ell**2 - ell

        return ell, emm

    def get_coor_group_from_jobid(self, jobid):
        return self.coords_groups_list[jobid]

    def get_Ylm_set_from_jobid(self, jobid):
        """Given a jobid, this returns the
        SingleMode object for all the angular positions
        in that job"""

    ####################
    # Assign
    ####################

    def flatten_reorder_assign_Clm_ang_modes_r_t_grouped_list(
        self, Clm_modes_t_r_grouped_list
    ):
        """Reorder the angular modes data at different radii and time steps
        from MPI workers and create a copy in each for
        radial spectral decomposition later

        The input data is the set of grouped Clm modes.

        axis 0: time
        axis 1: radius
        axis 2: theta
        axis 3: phi

        Please note that radial spectral decomposition has not been carried out
        yet.

        """

        if self.mpi_rank == 0:
            reordered_Clm_modes_t_r_flat_list = []

            Clm_modes_t_r_flat_list = flatten_3l(Clm_modes_t_r_grouped_list)

            # Iterate over t steps to get modes
            # at a given t step at all radial
            # collocation points.

            message(
                "Received t and r set modes list size",
                len(Clm_modes_t_r_flat_list),
                message_verbosity=4,
            )

            n_time = self.ntime
            n_radii = self.shape[1]

            for t_step in range(n_time):
                # Get all the Clm modes on all shells at a particular
                # t_step
                Clm_modes_r_flat_list_at_given_t_step = (
                    self.get_Clm_modes_at_t_step(
                        t_step, Clm_modes_t_r_flat_list
                    )
                )

                self.message_root(
                    f"Modes at t step {t_step} length",
                    len(Clm_modes_r_flat_list_at_given_t_step),
                    message_verbosity=4,
                )

                jobid_set_t_step = np.array(
                    [item[0] for item in Clm_modes_r_flat_list_at_given_t_step]
                )

                self.message_root(
                    "job id set at t step",
                    jobid_set_t_step,
                    message_verbosity=4,
                )

                # Get shell numbers at this t_step
                r_ind_set = jobid_set_t_step - t_step * n_radii

                self.message_root(
                    f"r_ind_set at t slice {t_step}",
                    r_ind_set,
                    message_verbosity=4,
                )

                # Reorder shells
                r_modes_order = np.argsort(r_ind_set)

                # Get just the modes at this t step into a list
                Clm_modes_r_ordered_at_t_step = [
                    Clm_modes_r_flat_list_at_given_t_step[index][1]
                    for index in r_modes_order
                ]

                reordered_Clm_modes_t_r_flat_list.append(
                    Clm_modes_r_ordered_at_t_step
                )

        else:
            reordered_Clm_modes_t_r_flat_list = None

        # Here, changes may be needed as we need to avoid saving a copy
        # of the full angular modes on each mpi rank
        reordered_Clm_modes_t_r_flat_list = self.mpi_comm.bcast(
            reordered_Clm_modes_t_r_flat_list, root=0
        )

        self.message_root(
            "Synchronizing before assigining modes r ", message_verbosity=3
        )

        self.mpi_comm.Barrier()

        self.message_root(
            "Finished synchronizing before assigining modes r ",
            message_verbosity=3,
        )

        self._reordered_Clm_modes_t_r_flat_list = (
            reordered_Clm_modes_t_r_flat_list
        )

    def assign_Clmr_to_modes(self, modes_Clmr_list_group):
        """Create a SingleMode object from the gathered
        Clmr modes list from MPI workers. Assign the interpolant"""

        modes_Clmr_list_flattened = flatten_3l(modes_Clmr_list_group)

        # Set mode data
        modes_Clmr = SingleMode(
            ell_max=self._method_info.ell_max,
            extra_mode_axis_len=self.shape[1],
        )

        for item in modes_Clmr_list_flattened:
            ell, emm, mode_data = item
            modes_Clmr.set_mode_data(ell, emm, mode_data)

        self._interpolant = modes_Clmr

    def flatten_assign_Clmr_to_modes_array(self, modes_Clmr_t_grouped_list):
        """Create a ModesArray object from the gathered
        Clmr modes list from MPI workers. Assign the interpolant.


        Here, the interpolant in spectral space is represented as
        a time series of modes in a ModesArray object. The data
        component of this object is basically the 4d array of
        Clmr at different instants of time."""

        # This contains the jobid, ell, emm and the Clmr mode
        modes_Clmr_t_flat_list = flatten_3l(modes_Clmr_t_grouped_list)

        message(
            "Flattened Clmr vs time list",
            modes_Clmr_t_flat_list,
            message_verbosity=4,
        )

        # Assign modes for every ell, emm at every time step
        for t_step in range(self.ntime):

            modes_Clmr_at_t_step = self.get_Clmr_modes_at_t_step(
                t_step, modes_Clmr_t_flat_list
            )

            self.message_root(
                f"Setting mode data at time step {t_step} ",
                message_verbosity=4,
            )
            # Set mode data

            self.message_root(
                f"Length of mode Clmr at tstep {t_step} ",
                len(modes_Clmr_at_t_step),
                message_verbosity=4,
            )

            # Here mode_data is a list of modes at the same r_index
            # at a given time step

            self._interpolant._time_axis[t_step] = t_step

            for item in modes_Clmr_at_t_step:
                tstep2, ell, emm, mode_data = item

                self.message_root(
                    f"Nested t step {t_step}", message_verbosity=4
                )

                self.message_root(
                    f"Setting l {ell} m {emm} mode data", message_verbosity=4
                )

                self.message_root(
                    "Mode data before",
                    self.interpolant.mode(ell, emm),
                    message_verbosity=4,
                )

                self._interpolant.set_mode_data_at_t_step(
                    t_step=t_step,
                    time_stamp=t_step,
                    ell=ell,
                    emm=emm,
                    data=mode_data,
                )

                self.message_root(
                    "Mode data after",
                    self.interpolant.mode(ell, emm),
                    message_verbosity=4,
                )

    def flatten_assign_Clmr_to_modes_array_v2(self, modes_Clmr_t_grouped_list):
        """Create a ModesArray object from the gathered
        Clmr modes list from MPI workers. Assign the interpolant.


        Here, the interpolant in spectral space is represented as
        a time series of modes in a ModesArray object. The data
        component of this object is basically the 4d array of
        Clmr at different instants of time."""

        # This contains the jobid, ell, emm and the Clmr mode
        modes_Clmr_t_flat_list = flatten_3l(modes_Clmr_t_grouped_list)

        message(
            "Flattened Clmr vs time list",
            modes_Clmr_t_flat_list,
            message_verbosity=4,
        )

        job_indices = [item[0] for item in modes_Clmr_t_flat_list]
        # Assign modes for every ell, emm at every time step

        for t_step in range(self.ntime):

            modes_Clmr_at_t_step = self.get_Clmr_modes_at_t_step(
                t_step, modes_Clmr_t_flat_list
            )

            self.message_root(
                f"Setting mode data at time step {t_step} ",
                message_verbosity=4,
            )

            # Set mode data
            self.message_root(
                f"Length of mode Clmr at tstep {t_step} ",
                len(modes_Clmr_at_t_step),
                message_verbosity=4,
            )

            # Here mode_data is a list of modes at the same r_index
            # at a given time step

            self._interpolant._time_axis[t_step] = t_step

            ell_inds = [item[1] for item in modes_Clmr_at_t_step]

            ell_inds_order = np.argsort(ell_inds)

            ell_ordered_modes_Clmr_at_t_step = [
                modes_Clmr_at_t_step[ind] for ind in ell_inds_order
            ]

            for item in modes_Clmr_at_t_step:
                tstep2, ell, emm, mode_data = item

                self.message_root(
                    f"Nested t step {t_step}", message_verbosity=4
                )

                self.message_root(
                    f"Setting l {ell} m {emm} mode data", message_verbosity=4
                )

                self.message_root(
                    "Mode data before",
                    self.interpolant.mode(ell, emm),
                    message_verbosity=4,
                )

                self._interpolant.set_mode_data_at_t_step(
                    t_step=t_step,
                    time_stamp=t_step,
                    ell=ell,
                    emm=emm,
                    data=mode_data,
                )

                self.message_root(
                    "Mode data after",
                    self.interpolant.mode(ell, emm),
                    message_verbosity=4,
                )

    def assign_interpolated_data(self, func_vals_list_ordered):
        func_vals = np.array(func_vals_list_ordered, dtype=np.complex128)

        message("Assiging interpolated data.. ", message_verbosity=2)
        self._interpolated_data = func_vals.reshape(
            np.array(self.sphp_output_grid[0]).shape
        )

    def assign_modes_data_to_modes_array(self, modes_Clmr_t_data):
        """Assign the modes data array to a modes array object"""

        message(
            "Shape of the modes data",
            modes_Clmr_t_data.shape,
            message_verbosity=2,
        )

        self.initialize_interpolant()

        self._interpolant._modes_data = modes_Clmr_t_data.transpose(2, 3, 1, 0)

    ##############################
    # Main
    ##############################

    def construct_interpolant(self, diagnostics=True):
        """Setup an interpolant that would be used to
        interpolate the `raw_data` specified on the
        given `input_grid` onto the `output_grid`
        to obtain the `interpolated_data`

        Returns
        -------
        self._interpolant : SingleMode
                            The 3d spectral interpolant
        """
        ################################
        # Angular decomposition

        # Radial set of Clm modes.
        # Each element is a singlemode obj from
        # a single shell at a single instant
        # of time

        # Construct Clm(r).
        # SHExpand at every radial shell

        # Radial shells
        # Assumes the same number of radial
        # shells through all time steps

        if self.saved_interpolant_file is not None:
            self.message_root("Loading interpoland from file")
            self.load_interpolant()

            return 1

        n_radii = self.shape[1]
        r_indices_at_t_step = np.arange(n_r)

        n_time = self.ntime

        total_nang_decomps = n_radii * n_time

        n_decomps_parallel = min(total_nang_decomps, self.mpi_nprocs)

        # Use only nang_decoms num of mpi workers for
        # the angular decomposition

        self.message_root(
            "Expanding radial shells in angular modes...", message_verbosity=2
        )
        self.message_root(
            "---------------------------------------------",
            message_verbosity=2,
        )

        self.message_root(
            "r indices per t step", r_indices_at_t_step, message_verbosity=3
        )

        if self.mpi_rank == 0:
            if self.mpi_nprocs > total_nang_decomps:
                message(
                    f"Using only {n_decomps_parallel} processors "
                    "for angular decompositions",
                    message_verbosity=2,
                )

            else:
                message(
                    f"Using all {self.mpi_nprocs} processors "
                    "for angular decompositions",
                    message_verbosity=2,
                )

        local_Clm_modes_t_r_list = []

        # Parallel angular decompose
        for t_step in range(n_time):
            for r_index in r_indices_at_t_step:

                job_index = t_step * n_radii + r_index

                if job_index % self.mpi_nprocs == self.mpi_rank:
                    message(
                        f"Interpolating element {job_index} of {n_time*n_radii} ",
                        message_verbosity=2,
                    )

                    message(
                        f"Angular decomposition at time step {t_step} r_index {r_index}",
                        message_verbosity=3,
                    )

                    local_Clm_modes = self.angular_expansion_at_t_and_r_index(
                        t_step=t_step, r_index=r_index
                    )

                    local_Clm_modes_t_r_list.append(
                        [job_index, local_Clm_modes]
                    )

        # Wait for ang decomp to finish
        self.mpi_comm.Barrier()

        # Gather
        self.message_root("Gathering modes r ", message_verbosity=2)

        Clm_modes_t_r_grouped_list = self.mpi_comm.gather(
            local_Clm_modes_t_r_list, root=0
        )

        self.mpi_comm.Barrier()

        if self.mpi_rank == 0:
            self.message_root(
                "Finished gathering modes t and r. Length "
                f"{len(Clm_modes_t_r_grouped_list)}",
                message_verbosity=3,
            )

        # Re order and recollect angular modes x 1d radial
        self.message_root(
            "Total number of local angular decomposition"
            f"jobs {len(local_Clm_modes_t_r_list)}",
            message_verbosity=2,
        )

        self.message_root("Reorganize modes in t and r ", message_verbosity=2)

        self.flatten_reorder_assign_Clm_ang_modes_r_t_grouped_list(
            Clm_modes_t_r_grouped_list
        )

        self.message_root(
            "Finished reordering and gathering modes r ", message_verbosity=3
        )

        self.message_root(
            "Finished expanding radial shells in angular modes...",
            message_verbosity=2,
        )

        #########################
        # Radial transformation
        ########################

        self.message_root(
            "Expanding angular modes in Chebyshev spectrum...",
            message_verbosity=2,
        )

        # Construct PClm
        modes_Clmr_list_local = []

        if self.mpi_rank == 0:
            if n_time < self.mpi_nprocs:
                message(
                    f"Using only {n_time}"
                    "processors for radial decomposition",
                    message_verbosity=2,
                )

        # Clmr modes list for all r for this particular
        # time slice, ell, emm.
        # The SingleMode object has been converted to raw data
        # within radial expansion at r and ell emm within
        # the function, where Clm modes are fetched for all
        # r_shells for radial spectral transform.

        self.radial_decompose()

        self.message_root("Synchronizing", message_verbosity=3)

        self.mpi_comm.Barrier()

        self.message_root(
            "Finished expanding angular modes" "in Chebyshev spectrum...",
            message_verbosity=2,
        )
        #############################
        # Construct the interpolant
        ###########################
        # Here modes_Clmr_list group_all_t is only
        # present in the root process. Hence
        # moving self.mpi_rank to inside the method
        # is not viable

        if self.mpi_rank == 0:
            # self.initialize_interpolant()
            # self.flatten_assign_Clmr_to_modes_array(modes_Clmr_list_group_all_t)

            self.message_root(
                "Checks on the constructed inteproland from rank 0",
                message_verbosity=4,
            )
            self.message_root(
                "Shape of interpoland modes",
                self.interpolant._modes_data.shape,
                message_verbosity=4,
            )
            self.message_root(
                "Time axis of the interolated modes",
                self.interpolant.time_axis,
                message_verbosity=4,
            )
            self.message_root(
                "Mode 2, 2", self.interpolant.mode(2, 2), message_verbosity=4
            )

        self.mpi_comm.Barrier()

        self.message_root(
            "Broadcasting constructed interpolant", message_verbosity=3
        )

        self._interpolant = self.mpi_comm.bcast(self.interpolant, root=0)

        self.message_root(
            "Successfully broadcasted constructed interpolant",
            message_verbosity=2,
        )

        self.save_interpolant()

        self.message_root(
            "Finished constructing interpolator...", message_verbosity=2
        )

    def evaluate_interpolant(self):
        """Find the values of the function in spectral space represented
        by its modes `modes_Clmr` on a requested set of coordinates .

        Parameters
        ----------
        modes_Clmr : dict
                     The modes dictionary with keys in the
                     format `lxny` and values being the radial
                     spectral Cehnyshev coefficients for that
                     mode.
        cart_coords : list of three 3darrays, optional
                      The meshgrid `X,Y,Z` of cartesian coordinates onto
                      which to evaluate the function.

        centers : list
                  The coordinate center of the cartesian
                  coodinate arrays.

        sphp_coords : list of three 3darrays, optional
                      The meshgrid `r, theta, phi` of
                      spherical polar coordinates onto
                      which to evaluate the function.
        Returns
        -------
        func_vals : 2darray
                    The value of the function on the requested
                    cartesian gridpoints.
        """
        self.message_root(
            "Synchronizing before beginning evaluation routine.. ",
            message_verbosity=3,
        )

        self.mpi_comm.Barrier()

        self.message_root(
            "Finished synchronizing before beginning" " evaluation routine.. ",
            message_verbosity=2,
        )

        evaluated_interpolant = []

        for t_step in range(self.ntime):
            R, Th, Ph = self.sphp_output_grid[t_step]

            self.reorganize_coords()

            if (
                len(self.coords_groups_list[t_step])
                > len(np.array(Th).flatten()) / 2
            ):

                self.create_Ylm_cache(t_step, overwrite=True)

                evaluated_interpolant.append(
                    self.evaluate_scattered(t_step=t_step)
                )

            else:
                # self.create_Ylm_cache(t_step, overwrite=True)
                # raise NotImplementedError(
                #    "The reorganized evaluation is not checked, hence this message"
                # )

                #
                #
                #
                #
                #
                evaluated_interpolant.append(self.evaluate_reorganized(t_step))
                # evaluated_interpolant.append(self.evaluate_scattered(t_step=t_step))

        if self.mpi_comm.rank == 0:
            evaluated_interpolant_array_ts = []

            message(
                "Shape of interpolated grid",
                self.sphp_output_grid[t_step][0].shape,
                message_verbosity=4,
            )

            for t_step, item in enumerate(evaluated_interpolant):
                message(
                    f"Length of one time step interpolated data at {t_step}",
                    len(item),
                    message_verbosity=4,
                )

                evaluated_interpolant_array = np.array(item).reshape(
                    self.sphp_output_grid[t_step][0].shape
                )

                message(
                    f"Shape of one time step interpolated data array {t_step}",
                    item.shape,
                    message_verbosity=4,
                )
                evaluated_interpolant_array_ts.append(
                    evaluated_interpolant_array
                )

            self._interpolated_data = evaluated_interpolant_array_ts

            message(
                "Number of time steps in interpolated data",
                len(evaluated_interpolant_array_ts),
                message_verbosity=4,
            )
            # message("Shape of interpolated data array", np.array(evaluated_interpolant_array).shape,  message_verbosity=2)

    def evaluate_scattered(self, t_step=None):
        """Evaluate the interpolant at a given time step
        point wise by parallelizing over points. Each MPI process
        would handle one angular point.

        Strategy
        First, create a cache of Ylm values over l, m, theta, phi
        Second, point by point, get Clm
        within each iter, contract with Ylm cache

        For MPI implementation, parallelize over points
        Angular contraction per point will employ openmp
        """

        self.message_root(
            "Evaluating by scattering and vectorization...",
            message_verbosity=2,
        )

        # coords_list
        # List zipped list of coords at particular time step
        # No reorganization.
        # Reorganization will only be
        # used in evaluate_reorganized

        coords_list = self.flatten_coords(t_step)

        job_out_list_local = []

        self.message_root("Evaluating at points...", message_verbosity=3)

        # local_counts = 0

        for ang_jobid, item in enumerate(coords_list):
            # Iterate over every point at this time
            # step
            if ang_jobid % self.mpi_nprocs == self.mpi_rank:
                message(
                    f"Job {ang_jobid} executed by rank {self.mpi_rank}",
                    message_verbosity=4,
                )

                # On scattered, r, theta, phi will
                # just be single float values
                radius, theta, phi = item

                message("At radius", radius, message_verbosity=4)

                message(
                    f"Angular coords Theta {theta} \n Phi {phi}",
                    message_verbosity=4,
                )

                message("r interpolating", message_verbosity=3)

                start = time.time()
                # Here Clm interp is a SingleMode object
                # that contains info about rhe Clm modes
                # on a sphere at particular radius.
                # at a single instant of time.
                Clm_interp = RContract(
                    self.interpolant,
                    radius,
                    cs=self.radial_grid,
                    t_step=t_step,
                )

                end = time.time()

                message(
                    f"R Contraction done in {end-start}", message_verbosity=3
                )

                start = time.time()
                fval = self.AngContractVec(Clm_interp, ang_jobid)

                end = time.time()

                message(f"Ang cont done in {end-start}", message_verbosity=3)

                message(f"fval shape {fval.shape}", message_verbosity=4)

                message(f"val {fval}", message_verbosity=4)

                # local list of evaluation job results
                # at this time step.
                job_out_list_local.append([ang_jobid, fval])

                # local_counts+=1

                # if self.mpi_rank==0:

                #    total_counts = self.mpi_comm.reduce(local_counts, root=0)
                #    progress = 100*total_counts/len(coords_list)

                # message(f" Progress : {progress} %", message_verbosity=2)
                #    print(f" Progress : {progress} %")

        self.message_root(
            "Synchronizing before gathering func values.. ",
            message_verbosity=3,
        )

        self.mpi_comm.Barrier()

        # gather
        job_out_list_grouped = self.mpi_comm.gather(job_out_list_local, root=0)

        if self.mpi_rank == 0:
            message(
                "Finished synchronizing after gathering func values.",
                message_verbosity=2,
            )

            # This list undoes the group structure from different mpi ranks
            # The result is basically a list of entries.
            # The entries contain jobid, list of values of func

            func_vals_list_ordered = self.reorganize_mpi_job_output(
                job_out_list_grouped
            )

            # Preserve original input order

            # First, undo the grouping in same r

            message(f"Evluation Done for t step {t_step}", message_verbosity=2)

            return func_vals_list_ordered

        else:
            return None

    def evaluate_reorganized(self, t_step=None):
        # Strategy
        # First, group coords with same r together
        # Second, evaluate 2d function value over angular elements
        # within each const r set
        # Contraction in r required when group is changed

        # For MPI implementation, parallelize over groups
        # Angular evaluation within a group would be serial

        job_out_list_local = []

        if self.mpi_rank == 0:
            message("Evaluating by reorganizing mesh...", message_verbosity=2)

            ngroups = len(self.coords_groups_list[t_step])

            if ngroups < self.mpi_nprocs:
                message(
                    f"Using only {ngroups}" "processors for evaluation",
                    message_verbosity=2,
                )

            message(f"Total num of points {ngroups}", message_verbosity=1)

        for jobid, item in enumerate(self.coords_groups_list[t_step]):
            if jobid % self.mpi_nprocs == self.mpi_rank:
                message(
                    f"Job {jobid} executed by rank {self.mpi_rank}",
                    message_verbosity=4,
                )

                Rf, Th, Ph = item

                ri = Rf[0]

                message("At radius", ri, message_verbosity=4)
                message(
                    f"Angular list Theta {Th} \n Phi {Ph}", message_verbosity=4
                )
                message("r interpolating", message_verbosity=4)

                Clm_interp = RContract(
                    self.interpolant, ri, cs=self.radial_grid, t_step=t_step
                )

                message("R Interpolation done", message_verbosity=4)

                # This evaluates Ylm at all points specified by
                # theta, phi again in a vectorized manner.
                # and does not use Ylm cache
                fval = AngContract(Clm_interp, Th, Ph)

                job_out_list_local.append([jobid, fval])

        self.message_root(
            "Synchronizing before gathering func values.. ",
            message_verbosity=3,
        )

        self.mpi_comm.Barrier()

        # gather
        job_out_list_grouped = self.mpi_comm.gather(job_out_list_local, root=0)

        if self.mpi_rank == 0:
            message(
                "Finished synchronizing after gathering func values.",
                message_verbosity=2,
            )

            func_vals_list = flatten(
                self.reorganize_mpi_job_output(job_out_list_grouped)
            )
            # Preserve original input order

            # First, undo the grouping in same r
            # Then unsort
            func_vals_list_ordered = unsort(
                func_vals_list, self.ang_args_order[t_step]
            )

            message(
                f"Evaluation Done for t step {t_step}", message_verbosity=2
            )

            return func_vals_list_ordered

        else:
            return None

    def create_Ylm_cache(self, t_step, Th=None, Ph=None, overwrite=False):
        """Create an array of Ylm at all required angular points
        to be used later for faster vectorized evaluation.

        At a given t_step, there exists a list of coords
        R, Th, Ph that represents the interpolation coordinates.

        This function evaluates and keeps all ell, m modes of
        Ylm cache evaluated at the interpolation points at that
        time step. This is then stored in self._Ylm_cache. Please
        note that this variable will be overwritten on the respective
        MPI processor every time this is called. This is OK because
        one MPI processor can only work on one evaluation at any
        instant of time.

        The created object self._Ylm_cache is a SingeMode obj

        Please note that the SingleMode object has the shape
        given by (ell, 2 * ell +1, flattened angular coords)
        """

        calculate = False

        if overwrite is False:
            if np.array(self._Ylm_cache).any() == np.array(None):
                calculate = True
            else:
                message(
                    "Ylm Cache already calculated."
                    " Please supply overwrite=True to re calculate"
                )
        else:
            calculate = True

        if calculate:
            self.message_root("Creating Ylm cache", message_verbosity=3)

            if np.array(Th).any() == np.array(None):
                Th = self.sphp_output_grid[t_step][1]

            if np.array(Ph).any() == np.array(None):
                Ph = self.sphp_output_grid[t_step][2]

            # Job list to parallelize over ell, emm
            jobs_list = []
            for ell in range(self.method_info.ell_max + 1):
                for emm in range(-ell, ell + 1):
                    jobs_list.append([ell, emm])

            Ylm_local_set = []

            for jobid, mode_set in enumerate(jobs_list):
                if jobid % self.mpi_nprocs == self.mpi_rank:
                    ell, emm = mode_set

                    Ylm = Yslm_vec(
                        spin_weight=0,
                        theta_grid=Th,
                        phi_grid=Ph,
                        ell=ell,
                        emm=emm,
                    )

                    Ylm_local_set.append([jobid, Ylm])

            self.mpi_comm.Barrier()

            Ylm_job_set = self.mpi_comm.gather(Ylm_local_set, root=0)

            if self.mpi_rank == 0:
                # This is an array of shape len(jobs_list), ntheta, nphi

                Ylm_set = self.reorganize_mpi_job_output(Ylm_job_set)

                # This has to be recast into
                # (ell_max + 1, 2 * (ell_max) + 1, ntheta, nphi)
                # Root cache
                # The 2d angular axis is flattened into
                # 1d array and saved as extra mode axis
                # in SingeMode obj
                # Its modes data has shape
                # ell x m x coord axis

                Ylm_cache = SingleMode(
                    ell_max=self.method_info.ell_max,
                    extra_mode_axis_len=len(np.array(Th).flatten()),
                )

                # Now assign modes based on ordered job id
                for jobid, item in enumerate(Ylm_set):
                    ell, emm = self.get_ell_emm_from_number(jobid)

                    Ylm_cache.set_mode_data(ell, emm, item.flatten())

                self._Ylm_cache = Ylm_cache

            # Store a copy of the cache onto all processors
            # Also, memory foorprint can be improved by
            # scattering only required data to procs

            self._Ylm_cache = self.mpi_comm.bcast(self._Ylm_cache, root=0)

            self.mpi_comm.Barrier()

            self.message_root("Created Ylm cache", message_verbosity=3)
            self.message_root(
                "Created Ylm cache shape",
                self.Ylm_cache._modes_data.shape,
                message_verbosity=4,
            )

        message(
            "Created Ylm cache modes data shape ",
            self.Ylm_cache._modes_data.shape,
            message_verbosity=4,
        )

    def distribute_Ylm_cache(self):
        """Distribute Ylm cache amongst all ranks"""
        if self.mpi_rank == 0:
            njobs = len(self.Ylm_cache._modes_data[0, 0, :])

            for jobid in range(njobs):
                message(f"Distributing packet {jobid}", message_verbosity=3)

                for rank_id in range(self.mpi_nprocs):
                    message(f"Contacting rank {rank_id}", message_verbosity=3)

                    if jobid % self.mpi_nprocs == rank_id:
                        message(
                            f"Eligible by rank {rank_id}", message_verbosity=3
                        )

                        sub_cache = self.Ylm_cache._modes_data[:, :, jobid]

                        self.mpi_comm.Send(
                            sub_cache.copy(), dest=rank_id, tag=jobid
                        )
                        # req.wait()

                        message(
                            f"Captured by rank {rank_id}", message_verbosity=3
                        )

                    break

        for jobid in range(len(jobs_list)):
            message(
                f"Attempting to receive packet {jobid}", message_verbosity=3
            )

            if jobid % self.mpi_nprocs == self.mpi_rank:
                data_chunk = np.empty(
                    (
                        self.method_info.ell_max + 1,
                        2 * self.method_info.ell_max + 1,
                    ),
                    dtype=np.complex128,
                )

                self.mpi_comm.Recv(data_chunk, source=0, tag=jobid)

                message(
                    f"Packet {jobid} received by rank {jobid}",
                    message_verbosity=3,
                )

                self._local_Ylm_data_cache_dict.update({jobid: data_chunk})

    ##########################################
    # IO
    ##########################################

    def save_interpolant(self, fname=None):
        """Save the interpolant to file"""

        nt, _, _, _ = self.shape

        if self.mpi_rank == 0:
            if fname is None:
                tstamp = time.time()

                fname = f"spectral_interpolant_{self.label}_{nt}_{tstamp}.dump"

            import pickle

            with open(fname, "wb") as sf:
                pickle.dump(self.interpolant, sf)

    def load_interpolant(self):
        """Load an interpolator saved to a file"""

        if self.mpi_rank == 0:
            with open(self.saved_interpolant_file, "rb") as lf:
                self._interpolant = pickle.load(lf)

        self._interpolant = self.mpi_comm.bcast(self.interpolant, root=0)

        self.message_root("Successfully loaded interpolant from file")
