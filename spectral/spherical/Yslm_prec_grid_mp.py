import multiprocessing
import os
from multiprocessing import Pool, TimeoutError, cpu_count

import numpy as np
from waveformtools.transforms import Yslm_prec_grid
from waveformtools.waveformtools import message

Yslm_prec_grid_mp_cache = {}


class Yslm_prec_grid_mp:
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

    def __init__(self,
                 ell_max=6,
                 grid_info=None,
                 prec=16,
                 nprocs=None,
                 spin_weight=0):
        
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
        return self._spin_weght
    
    @property
    def job_list(self):
        return self._job_list
    
    @property
    def result_list(self):
        return self._result_list
    
    
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
        
        if not self.is_available_in_cache():

            #if __name__ == '__main__':

            multiple_results = self.pool.map(self.compute_Yslm_prec, self.job_list)
            #multiple_results = self.pool.map(self.test_mp, self.job_list)
            # launching multiple evaluations asynchronously *may* use more processes
            #multiple_results = self.pool.apply_async(self.compute_Yslm_prec, self.job_list, callback=self.log_results)
            
            #for item in self.job_list:
            #multiple_results = [self.pool.apply_async(self.test_mp, args=(one_mode,), callback=self.log_results) for one_mode in self.job_list]
            
            self._result_list = multiple_results
            
            
            self.pool.close()
            
            self.pool.join()
            
            # results = [job.get() for job in multiple_results]
            
            self.update_cache()

        else:
            self._result_list = Yslm_prec_grid_mp_cache[self.spin_weight][self.ell_max]

    def compute_Yslm_prec(self, mode_number):
        
        
        mode_count, ell, emm = mode_number
        
        print(f"Job {mode_count} Computing Yslm for l{ell} m{emm}\n")
        theta_grid, phi_grid = self.grid_info.meshgrid
        
        return [mode_count, np.array(Yslm_prec_grid(theta_grid=theta_grid, phi_grid=phi_grid, spin_weight=self.spin_weight, ell=ell, emm=emm), dtype=np.complex128)]

    
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

        if self.spin_weight in Yslm_prec_grid_mp_cache.keys():

            if self.ell_max in Yslm_prec_grid_mp_cache[self.spin_weight].keys():
                available=True

        return available

    def update_cache(self):
        ''' Update cache after computation for faster retrieval'''

        Yslm_prec_grid_mp_cache.update({self.spin_weight : {self.ell_max : self.result_list}})


    