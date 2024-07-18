import multiprocessing
import os
from multiprocessing import Pool, TimeoutError, cpu_count
from waveformtools.single_mode import SingleMode
import numpy as np
from waveformtools.transforms import Yslm_vec as Yslm
from waveformtools.waveformtools import message

#


class Yslm_mp:
    ''' Evaluate the spin weighted spherical harmonics 
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
    '''

    Yslm_mp_cache = {}
    
    def __init__(self,
                 ell_max=6,
                 grid_info=None,
                 prec=16,
                 nprocs=None,
                 spin_weight=0,
                 theta=None,
                 phi=None):
        
        self._ell_max = ell_max
        self._grid_info = grid_info
        self._prec = prec
        self._nprocs = nprocs
        self._spin_weight = spin_weight
        
        if grid_info is None:
            if theta is None and phi is None:
                raise KeyError("Please specify the grid, or theta and phi")
            
            else:
                self._theta = theta
                self._phi = phi
        else:
            self._theta, self._phi = grid_info.meshgrid

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
    def grid_info(self):
        return self._grid_info
    
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
    
    def setup_env(self):

        from multiprocessing import Pool, TimeoutError, cpu_count

    def get_max_nprocs(self):
        ''' Get the available number of processors on the CPU '''

        max_ncpus = cpu_count()

        return max_ncpus
    
    def create_job_list(self):
        
        job_list = []
        
        mode_count = 0
        for ell in range(self.ell_max+1):
            for emm in range(-ell, ell+1):
                job_list.append([mode_count, ell, emm])
                mode_count+=1
                
        self._job_list = job_list
        
    def log_results(self, result):
        
        self._result_list.append(result)
        
        
    def initialize(self):
        ''' Initialize the workers / pool '''

        if self.nprocs is None:

            max_ncpus = self.get_max_nprocs()

            self._nprocs = int(max_ncpus/2)

        self.create_job_list()
        
        self.pool = multiprocessing.Pool(processes=self.nprocs)
        
        
    def run(self):
        
        #if not self.is_available_in_cache():

        #if __name__ == '__main__':
        self.initialize()
        multiple_results = self.pool.map(self.compute_Yslm, self.job_list)

        #multiple_results = self.pool.map(self.test_mp, self.job_list)
        # launching multiple evaluations asynchronously *may* use more processes
        #multiple_results = self.pool.apply_async(self.compute_Yslm, self.job_list, callback=self.log_results)
        
        #for item in self.job_list:
        #multiple_results = [self.pool.apply_async(self.test_mp, args=(one_mode,), callback=self.log_results) for one_mode in self.job_list]
        
        self._result_list = multiple_results
        
        
        self.pool.close()
        
        self.pool.join()
        
        # results = [job.get() for job in multiple_results]
        
        #self.update_cache()

            

        #else:
        #    self._result_list = Yslm_mp_cache[self.spin_weight][self.ell_max]

        self.store_as_modes()

    def compute_Yslm(self, mode_number):
        
        
        mode_count, ell, emm = mode_number
        
        #print(f"Job {mode_count} Computing Yslm for l{ell} m{emm}\n")

        #theta_grid, phi_grid = self.grid_info.meshgrid
        
        return [mode_count, np.array(Yslm(theta_grid=self.theta, phi_grid=self.phi, spin_weight=self.spin_weight, ell=ell, emm=emm), dtype=np.complex128)]

    
    def test_mp(self, mode_number):
        
        print(f"This is process {os.getpid()} processing mode {mode_number}\n")
        
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def is_available_in_cache(self):
        ''' Check if the current parameters are available in cache '''
        available = False

        if self.spin_weight in Yslm_mp_cache.keys():

            if self.ell_max in Yslm_mp_cache[self.spin_weight].keys():
                available=True

        return available

    def update_cache(self):
        ''' Update cache after computation for faster retrieval'''

        Yslm_mp_cache.update({self.spin_weight : {self.ell_max : self.result_list}})


    def store_as_modes(self):
        ''' Store the results as modes '''

        sYlm_modes = SingleMode(ell_max=self.ell_max,
                                spin_weight=self.spin_weight
                                ) 
        
        for mode_num, mode_val in self.result_list:

            ell_filled = int(np.sqrt(mode_num)) -1
            ell = ell_filled+1

            nfilled = (ell_filled+1)**2

            emm = mode_num -nfilled - ell 
            
            #print(ell, emm, mode_num)
            sYlm_modes.set_mode_data(ell, emm, mode_val)

            
        self._sYlm_modes = sYlm_modes
