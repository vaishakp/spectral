"""Parallel arbitrary-precision harmonics over a whole grid.

Like :mod:`spectools.spherical.Yslm_mp`, but delegating each mode to
``waveformtools``' ``Yslm_prec_grid`` rather than evaluating it here. One
``multiprocessing`` job is created per :math:`(\\ell, m)` mode, and results
are returned as ``[mode_count, values]`` pairs in mode-count order.

Computed mode sets are memoised in :data:`Yslm_prec_grid_mp_cache`, keyed
by spin weight and band limit, so repeating a run with the same parameters
skips the pool entirely.

.. note::

   The cache is keyed only by ``spin_weight`` and ``ell_max``, not by the
   grid or the requested precision. Two runs over different grids, or at
   different ``prec``, collide: the second returns the first's values.
   Construct in a fresh process, or clear
   :data:`Yslm_prec_grid_mp_cache`, when changing either.

Requires the ``spherical`` extra.
"""

import multiprocessing
import os
from multiprocessing import Pool, TimeoutError, cpu_count

import numpy as np
from waveformtools.transforms import Yslm_prec_grid
from waveformtools.waveformtools import message

#: Module-level memo of computed mode sets, keyed as
#: ``[spin_weight][ell_max]``. Never evicted from.
Yslm_prec_grid_mp_cache = {}


class Yslm_prec_grid_mp:
    """Evaluate the spin weighted spherical harmonics
    asynchronously on multiple processors at any precision
    required

    Attributes
    ----------
    ell_max : int
              The max :math:`\\ell` to use to evaluate the harmonic coefficients.
    grid_info : Grid
                An object of the Grid class, that is used to setup
                the coordinate grid on which to evaluate the spherical
                harmonics.
    prec : int
           The precision to maintain in order to evaluate the spherical
           harmonics.
    nprocs : int
             The number of processors to use. Defaults to half the available.
    """

    def __init__(
        self, ell_max=6, grid_info=None, prec=16, nprocs=None, spin_weight=0
    ):

        self._ell_max = ell_max
        self._grid_info = grid_info
        self._prec = prec
        self._nprocs = nprocs
        self._spin_weght = spin_weight

        self.setup_env()

        self._job_list = None
        self._result_list = []

        self.initialize()

    @property
    def ell_max(self):
        """int: Band limit; harmonics are computed for all :math:`\\ell` up to this."""
        return self._ell_max

    @property
    def prec(self):
        """int: Requested digits of precision.

        .. note::

           Stored but not forwarded: :meth:`compute_Yslm_prec` calls
           ``Yslm_prec_grid`` without a ``prec`` argument, so that
           function's own default is used regardless of what is set here.
        """
        return self._prec

    @property
    def grid_info(self):
        """SphericalGrid: The grid to evaluate on, via its ``meshgrid``."""
        return self._grid_info

    @property
    def nprocs(self):
        """int: Worker processes in the pool.

        Resolved by :meth:`initialize` to half the available CPUs when not
        given explicitly.
        """
        return self._nprocs

    @property
    def spin_weight(self):
        """int: Spin weight :math:`s` of the harmonics."""
        return self._spin_weght

    @property
    def job_list(self):
        """list or None: One ``[mode_count, ell, emm]`` entry per mode.

        ``None`` until :meth:`create_job_list` is called, which
        :meth:`initialize` does.
        """
        return self._job_list

    @property
    def result_list(self):
        """list: Per-mode ``[mode_count, values]`` pairs, in mode order."""
        return self._result_list

    def setup_env(self):
        """Import the ``multiprocessing`` names used by the worker pool.

        .. note::

           A no-op in practice: the imports are local to this method and
           go out of scope the moment it returns, and the same names are
           already imported at module level. Retained because
           ``__init__`` calls it.
        """

        from multiprocessing import Pool, TimeoutError, cpu_count

    def get_max_nprocs(self):
        """Get the available number of processors on the CPU"""

        max_ncpus = cpu_count()

        return max_ncpus

    def create_job_list(self):
        """Enumerate every mode up to ``ell_max`` as a parallel job.

        Builds :attr:`job_list` as one ``[mode_count, ell, emm]`` entry per
        mode, with ``mode_count`` a running index used to restore ordering
        once the workers return out of order. Returns nothing.
        """

        job_list = []

        mode_count = 0
        for ell in range(self.ell_max + 1):
            for emm in range(-ell, ell + 1):
                job_list.append([mode_count, ell, emm])
                mode_count += 1

        self._job_list = job_list

    def log_results(self, result):
        """Append one worker's result to :attr:`result_list`.

        Intended as an ``apply_async`` callback.

        Parameters
        ----------
        result : list
            A ``[mode_count, values]`` pair from
            :meth:`compute_Yslm_prec`.

        Notes
        -----
        Unused on the current code path: :meth:`run` uses ``pool.map``,
        which returns results in order and assigns them directly.
        """

        self._result_list.append(result)

    def initialize(self):
        """Initialize the workers / pool"""

        if self.nprocs is None:

            max_ncpus = self.get_max_nprocs()

            self._nprocs = int(max_ncpus / 2)

        self.create_job_list()

        self.pool = multiprocessing.Pool(processes=self.nprocs)

    def run(self):
        """Compute every mode, in parallel, and store the results.

        Returns immediately from :data:`Yslm_prec_grid_mp_cache` if this
        ``(spin_weight, ell_max)`` combination has already been computed.
        Otherwise maps :meth:`compute_Yslm_prec` over :attr:`job_list`
        across the pool, stores the results in :attr:`result_list`, closes
        the pool and updates the cache. Returns nothing.

        Notes
        -----
        The pool is closed and joined here, so an instance can only be
        :meth:`run` once; construct a new one to compute again.
        """

        if not self.is_available_in_cache():

            # if __name__ == '__main__':

            multiple_results = self.pool.map(
                self.compute_Yslm_prec, self.job_list
            )
            # multiple_results = self.pool.map(self.test_mp, self.job_list)
            # launching multiple evaluations asynchronously *may* use more processes
            # multiple_results = self.pool.apply_async(self.compute_Yslm_prec, self.job_list, callback=self.log_results)

            # for item in self.job_list:
            # multiple_results = [self.pool.apply_async(self.test_mp, args=(one_mode,), callback=self.log_results) for one_mode in self.job_list]

            self._result_list = multiple_results

            self.pool.close()

            self.pool.join()

            # results = [job.get() for job in multiple_results]

            self.update_cache()

        else:
            self._result_list = Yslm_prec_grid_mp_cache[self.spin_weight][
                self.ell_max
            ]

    def compute_Yslm_prec(self, mode_number):
        """Compute one mode. Runs inside a worker process.

        Parameters
        ----------
        mode_number : list
            A ``[mode_count, ell, emm]`` entry from :attr:`job_list`.

        Returns
        -------
        list
            ``[mode_count, values]``, where ``values`` is
            :math:`{}_sY_{\\ell m}` over the grid as a ``complex128``
            array. ``mode_count`` is carried through so results can be put
            back in order.
        """

        mode_count, ell, emm = mode_number

        print(f"Job {mode_count} Computing Yslm for l{ell} m{emm}\n")
        theta_grid, phi_grid = self.grid_info.meshgrid

        return [
            mode_count,
            np.array(
                Yslm_prec_grid(
                    theta_grid=theta_grid,
                    phi_grid=phi_grid,
                    spin_weight=self.spin_weight,
                    ell=ell,
                    emm=emm,
                ),
                dtype=np.complex128,
            ),
        ]

    def test_mp(self, mode_number):
        """Print which process picked up a mode. Diagnostic only.

        A stand-in for :meth:`compute_Yslm_prec` for checking that work is
        actually being distributed across the pool. Does no computation
        and returns nothing.

        Parameters
        ----------
        mode_number : list
            A ``[mode_count, ell, emm]`` entry from :attr:`job_list`.
        """

        print(f"This is process {os.getpid()} processing mode {mode_number}\n")

    def __getstate__(self):
        """Return picklable state, dropping the unpicklable pool.

        Bound methods are sent to the workers by pickling ``self``, and a
        ``multiprocessing.Pool`` cannot itself be pickled, so it is
        excluded here.
        """
        self_dict = self.__dict__.copy()
        del self_dict["pool"]
        return self_dict

    def __setstate__(self, state):
        """Restore state in a worker process, without the pool."""
        self.__dict__.update(state)

    def is_available_in_cache(self):
        """Check if the current parameters are available in cache"""
        available = False

        if self.spin_weight in Yslm_prec_grid_mp_cache.keys():

            if (
                self.ell_max
                in Yslm_prec_grid_mp_cache[self.spin_weight].keys()
            ):
                available = True

        return available

    def update_cache(self):
        """Update cache after computation for faster retrieval"""

        Yslm_prec_grid_mp_cache.update(
            {self.spin_weight: {self.ell_max: self.result_list}}
        )
