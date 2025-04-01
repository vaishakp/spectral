import multiprocessing
import os
from multiprocessing import Pool, TimeoutError, cpu_count
from waveformtools.single_mode import SingleMode
import numpy as np
from spectral.spherical.swsh import Yslm_vec as Yslm
from waveformtools.waveformtools import message
from warnings import warn

#


class Yslm_mp:
    """Evaluate the spin weighted spherical harmonics
    asynchronously on multiple processors at any precision
    required

    Attributes
    ----------
    ell_max : int
              The max :math:`\\ell` to use to evaluate the harmonic coefficients.
    Grid : Grid
                An object of the Grid class, that is used to setup
                the coordinate grid on which to evaluate the spherical
                harmonics.
    prec : int
           The precision to maintain in order to evaluate the spherical
           harmonics.
    nprocs : int
             The number of processors to use. Defaults to half the available.
    """

    _Yslm_mp_cache = {}

    def __init__(
        self,
        ell_max=6,
        Grid=None,
        prec=16,
        nprocs=None,
        spin_weight=0,
        theta=None,
        phi=None,
        cache=True,
    ):

        self._ell_max = ell_max
        self._Grid = Grid
        self._prec = prec
        self._nprocs = nprocs
        self._spin_weight = spin_weight
        self._cache = cache

        if Grid is None:
            if theta is None and phi is None:
                raise KeyError("Please specify the grid, or theta and phi")
            else:
                self._theta = theta
                self._phi = phi
            # Dont cache if grid type in unknown
            self._cache = False
        else:
            if Grid.grid_type != "GL":

                warn(
                    "Caching is only currently supported for Gauss-Legendre type grids. \n Turning of caching."
                )
                # Dont cache if grid type is not GL.
                self._cache = False

            self._theta, self._phi = Grid.meshgrid

        self.setup_env()
        self._job_list = None
        self._result_list = []

    @property
    def ell_max(self):
        return self._ell_max

    @property
    def prec(self):
        return self._prec

    @property
    def Grid(self):
        return self._Grid

    @property
    def nprocs(self):
        return self._nprocs

    @property
    def spin_weight(self):
        return self._spin_weight

    @property
    def job_list(self):
        return self._job_list

    @property
    def result_list(self):
        return self._result_list

    @property
    def theta(self):
        return self._theta

    @property
    def phi(self):
        return self._phi

    @property
    def sYlm_modes(self):
        return self._sYlm_modes

    @property
    def cache(self):
        return self._cache

    def mode(self, ell, emm):
        return self.sYlm_modes.mode(ell, emm)

    def setup_env(self):
        """Imports"""
        from multiprocessing import Pool, TimeoutError, cpu_count

    def get_max_nprocs(self):
        """Get the available number of processors on the system"""
        max_ncpus = cpu_count()
        return max_ncpus

    def create_job_list(self):
        """Create a list of jobs (modes) for distributing
        computing to different processors"""

        job_list = []

        mode_count = 0
        for ell in range(abs(self.spin_weight), self.ell_max + 1):
            for emm in range(-ell, ell + 1):
                job_list.append([mode_count, ell, emm])
                mode_count += 1

        self._job_list = job_list

    def log_results(self, result):
        """Save result to memory"""
        self._result_list.append(result)

    def initialize(self):
        """Initialize the workers / pool"""

        if self.nprocs is None:
            max_ncpus = self.get_max_nprocs()
            self._nprocs = int(max_ncpus / 2)

        self.create_job_list()
        self.pool = multiprocessing.Pool(processes=self.nprocs)

    def run(self):
        """Compute the SHSHs, cache results, and create modes"""

        if not self.is_available_in_cache():
            self.initialize()
            multiple_results = self.pool.map(self.compute_Yslm, self.job_list)
            self._result_list = multiple_results
            self.pool.close()
            self.pool.join()
            self.store_as_modes()
            self.update_cache()
            self.pool.close()

        else:
            self._sYlm_modes = self._Yslm_mp_cache[self.spin_weight][
                self.ell_max
            ]

    def compute_Yslm(self, task):
        """Compute the SHSH for the given mode number"""

        mode_count, ell, emm = task
        return [
            mode_count,
            np.array(
                Yslm(
                    theta_grid=self.theta,
                    phi_grid=self.phi,
                    spin_weight=self.spin_weight,
                    ell=ell,
                    emm=emm,
                ),
                dtype=np.complex128,
            ),
        ]

    def test_mp(self, mode_number):
        """Print a simple test output message"""
        message(
            f"This is process {os.getpid()} processing mode {mode_number}\n",
            message_verbosity=1,
        )
        return 1

    def __getstate__(self):
        """Refresh Pool state"""
        self_dict = self.__dict__.copy()
        del self_dict["pool"]
        return self_dict

    def __setstate__(self, state):
        """Set Pool state"""
        self.__dict__.update(state)

    def is_available_in_cache(self):
        """Check if the current parameters are available in cache"""

        availability = False

        if self.cache:
            if self.spin_weight in self._Yslm_mp_cache.keys():
                if self.ell_max in self._Yslm_mp_cache[self.spin_weight].keys():
                    # print("Retrieving from cache")
                    availability = True

        # print(availability, self._Yslm_mp_cache)

        return availability

    def update_cache(self):
        """Update cache after computation for faster retrieval"""
        # print("Updating cache")
        if self.cache:
            Yslm_mp._Yslm_mp_cache.update(
                {self.spin_weight: {self.ell_max: self._sYlm_modes}}
            )

        # print(Yslm_mp._Yslm_mp_cache)

    def store_as_modes(self):
        """Store the results as modes"""

        self._sYlm_modes = SingleMode(
            ell_max=self.ell_max, spin_weight=self.spin_weight
        )

        mode_nums = [item[0] for item in self.result_list]
        mode_vals = [item[1] for item in self.result_list]
        diffs = np.diff(mode_nums)

        if not (diffs == 1).all():
            message("Sorting the results!")
            args_order = np.argsort(mode_nums)
            mode_vals = np.array(mode_vals)[args_order]

        self._sYlm_modes._modes_data = np.array(mode_vals)
        # self.__sYlm_modes._extra_mode_axes_shape = theta
