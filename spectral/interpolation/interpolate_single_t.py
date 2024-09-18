import mpi4py
from mpi4py import MPI
import numpy as np
from waveformtools.waveformtools import flatten, message, unsort
from spectral.spherical.grids import GLGrid
from waveformtools.diagnostics import method_info

from chebyshev import ChebyshevSpectral
from waveformtools.single_mode import SingleMode
from spectral.spherical.transforms import SHExpand, SHContract
from qlmtools.sxs.transforms import ToSphericalPolar, ReorganizeCoords, RContract, get_radial_clms, AngContract

import matplotlib.pyplot as plt


class Interpolate3D:

    def __init__(self,
                 r_min,
                 r_max,
                 coord_centers=None,
                 sphp_output_grid=None, 
                 cart_output_grid=None,
                 raw_data=None,
                 ):

        # User input quantities
        self._cart_output_grid = cart_output_grid
        self._sphp_output_grid = sphp_output_grid
        #self._output_grid = output_grid
        self._raw_data = raw_data
        self._coord_centers = coord_centers
        self._r_min = r_min
        self._r_max = r_max
        
        # Derived quantities
        self._shape = None
        self._interpolant = None
        self._interpolated_data = None
        self._ang_grid = None
        self._radial_grid = None
        self._method_info = None
        self._input_grid = None
        
        self._mpi_rank=None
        self._mpi_nprocs=None
        
        
        self.initialize_parallel_engine()
        self.pre_setup()
        
    @property
    def shape(self):
        ''' The number of input grid points '''
        if np.array(self._shape).all() == np.array(None):
            self._shape = self.raw_data.shape
                
        return self._shape

    @property
    def coord_centers(self):
        ''' The coordinate center location '''
        return self._coord_centers
    
    @property
    def sphp_output_grid(self):
        ''' The output coordinate grid 
        in spherical polar coordinates 
        for interpolation '''

        return self._sphp_output_grid
    
    @property
    def cart_output_grid(self):
        ''' The output coordinate grid 
        in cartesian coordinates 
        for interpolation '''

        return self._cart_output_grid
    
    @property
    def r_min(self):
        ''' The smallest radial shell present
        in the input grid '''
        if self._r_min is None:
            raise KeyError("Please provide r_min of the input grid")
            #self._r_min = np.amin(self.sphp_output_grid[0])
        
        return self._r_min

    @property
    def r_max(self):
        ''' The largest radial shell present
        in the input grid '''
        if self._r_max is None:
            raise KeyError("Please provide r_max of the input grid")
        
        return self._r_max
    
    @property
    def output_grid(self):
        ''' The output coordinate grid interpolated onto '''

        return self._output_grid

    @property
    def raw_data(self):
        ''' The data to be interpolated '''

        return self._raw_data


    @property
    def interpolated_data(self):
        ''' The resulting data after interpolation '''

        return self._interpolated_data

    @property
    def mpi_rank(self):
        return self._mpi_rank

    @property
    def mpi_nprocs(self):
        return self._mpi_nprocs

    @property
    def mpi_comm(self):
        return self._mpi_comm
    
    @property
    def interpolant(self):
        return self._interpolant
    
    @property
    def interpolated_data(self):
        return self._interpolated_data
    
    @property
    def ang_grid(self):
        
        return self._ang_grid
    
    @property
    def input_grid(self):
    
        return self._input_grid
    
    @property
    def radial_grid(self):
        
        return self._radial_grid
    
    @property
    def method_info(self):
        return self._method_info
    
    
    def initialize_parallel_engine(self):
        ''' Initialize the parallel compute engine
        for interpolation '''

        self._mpi_comm = MPI.COMM_WORLD
        #rank = comm.Get_rank()
        #size = comm.Get_size()
        
        self._mpi_nprocs = self.mpi_comm.Get_size()
        self._mpi_rank = self.mpi_comm.Get_rank()
        
        if self.mpi_rank==0:
            message(f"Engine using {self.mpi_nprocs} processors", 
                    message_verbosity=2)
        
        
    def pre_setup(self):
        ''' Setup the angluar spectral grid and
        expansion parameters '''
    
        n_r, n_theta, n_phi = self.raw_data.shape
        
        if self.mpi_rank==0:
            message('Num of radial points', n_r, message_verbosity=2)

        self._ang_grid = GLGrid(L=n_theta-1)

        #message(f"grid info ell max {grid_info.ell_max}")
        if self.mpi_rank==0:
            message('L grid', self.ang_grid.L, message_verbosity=2)


        self._method_info = method_info(ell_max=self.ang_grid.L, 
                                        int_method='GL')
        
        if self.mpi_rank==0:
            message(f"method info ell max {self.method_info.ell_max}", 
                    message_verbosity=2)
        
        self._radial_grid = ChebyshevSpectral(a=self.r_min, b=self.r_max, 
                                              Nfuncs=self.shape[0])

        if self.mpi_rank==0:
            message("Created Chebyshev radial grid"
                f"with Nfuncs {self.radial_grid.Nfuncs}\n", 
                f"Shape of collocation points \
                {self.radial_grid.collocation_points_logical.shape}", 
                message_verbosity=2)

        self._input_grid = [self.radial_grid, self.ang_grid]
        
        
        # Output coordinate grid setup
        if np.array(self.sphp_output_grid).all() == np.array(None):
            if np.array(self.cart_output_grid).all() == np.array(None):
                
                raise KeyError("Please provide the input grid"
                               "in either cartesian"
                               "or spherical polar coordinates")
            else:
                # Transform cart to spherical
                X, Y, Z = self.cart_output_grid

                if np.array(self.coord_centers).all() == np.array(None):
                    self._coord_centers = [0, 0, 0]

                #xcom, ycom, zcom = self.coord_centers
                message('Transforming to spherical polar coordinates', 
                        message_verbosity=2)

                self._sphp_output_grid = ToSphericalPolar(
                        self.cart_output_grid, 
                        self.coord_centers
                )
    
                
    
    
    def angular_expansion_at_r_index(self, r_index):
        ''' Expand the angular data at one radial
        collocation point in SH '''
        
        ang_data = self.raw_data[r_index, :, :]
        local_one_set_modes = SHExpand(func=ang_data, 
                                       method_info=self.method_info, 
                                       info=self.ang_grid)
        
        return local_one_set_modes
    
    def radial_expansion_at_ell_emm(self, ell, emm):
        ''' Carry out radial expansion of Clm modes '''
        
        this_r_modes_local = get_radial_clms(self._modes_r_ordered, ell, emm)
        this_Clmq = self.radial_grid.MatrixPhysToSpec@np.array(this_r_modes_local)
        
        
        message(f' This Clmq l{ell}, m{emm}', this_Clmq, message_verbosity=4)
        
        return this_Clmq
                
    def reorder_ang_modes_x_r_list(self, modes_r_set_group):
        ''' Reorder the angular modes data from MPI workers
        and create a copy in each for later use '''
        
        if self.mpi_rank==0:
            modes_r_set = flatten(modes_r_set_group)
            r_ind_set = [item[0] for item in modes_r_set]
            r_modes_order = np.argsort(r_ind_set)
            
            modes_r_ordered = [modes_r_set[index][1] for index in r_modes_order]
        else:
            modes_r_ordered = None
            
        modes_r_ordered = self.mpi_comm.bcast(modes_r_ordered, root=0)
        
        if self.mpi_rank==0:
            message("Synchronizing before assigining modes r ", 
                    message_verbosity=2)
        self.mpi_comm.Barrier()
        
        if self.mpi_rank==0:
            message("Finished synchronizing before assigining modes r ", 
                    message_verbosity=2)
        
        self._modes_r_ordered = modes_r_ordered
            
            
    def assign_Clmq_to_modes(self, modes_Clmq_list_group):
        ''' Create a SingleMode object from the gathered 
        Clmq modes list from MPI workers. Assign the interpolant '''
        
        modes_Clmq_list_flattened = flatten(modes_Clmq_list_group)
            
        # Set mode data
        modes_Clmq = SingleMode(ell_max=self._method_info.ell_max, 
                                modes_dim=self.shape[0])
        
        for item in modes_Clmq_list_flattened:
            ell, emm, mode_data = item
            
            #this_ell_modes.update({f'm{emm}' : this_Clmq})
            modes_Clmq.set_mode_data(ell, emm, mode_data)
    
        #return modes_Clmq
    
        self._interpolant = modes_Clmq
        
        
        
        
    def construct_interpolant(self, diagnostics=True):
        ''' Setup an interpolant that would be used to
        interpolate the `raw_data` specified on the 
        given `input_grid` onto the `output_grid`
        to obtain the `interpolated_data` 
        
        Returns
        -------
        
        self._interpolant : SingleMode
                            The 3d spectral interpolant
        '''
        ################################
        # Angular decomposition
        #######################
        
        
        # Radial set of Clm modes.
        # Each element is a singlemode obj
        local_modes_r_set = []
        
        # Construct Clm(r).
        # SHExpand at every radial shell
        
        # Radial shells
        r_indices = np.arange(self.shape[0])
        nang_decoms = min(self.shape[0], self.mpi_nprocs)
        
        
        # Use only nang_decoms num of mpi workers for 
        # the angular decomposition
        if self.mpi_rank==0:
            message("Expanding radial shells in angular modes...", 
                    message_verbosity=2)
            
            if self.mpi_nprocs>nang_decoms:
                message(f"Using only {nang_decoms} processors"
                        "for angular decompositions", 
                        message_verbosity=2)
        
        for r_index in r_indices:
            #if self.mpi_rank<nang_decoms:
            if r_index%self.mpi_nprocs==self.mpi_rank:
                    
                local_one_set_modes = self.angular_expansion_at_r_index(
                        r_index
                    )
                local_modes_r_set.append([r_index, local_one_set_modes])
                
        # Wait for ang decomp to finish
        self.mpi_comm.Barrier()
        
        # Gather
        if self.mpi_rank==0:
            message("Gathering modes r ", message_verbosity=2)
        
        modes_r_set_group = self.mpi_comm.gather(local_modes_r_set, root=0) 
            
        #self.mpi_comm.Barrier()
        
        
        if self.mpi_rank==0:
            message("Finished gathering modes r ", 
                    message_verbosity=2)
            # Re order and recollect angular modes x 1d radial
        
            message("Reorganize modes r ", message_verbosity=2)
        
        self.reorder_ang_modes_x_r_list(modes_r_set_group)
        
        if self.mpi_rank==0:
            message("Finished reordering and gathering modes r ", 
                message_verbosity=2)
        
        
            message("Finished expanding radial shells in angular modes...", 
                    message_verbosity=2)
        #self.comm.bcast(modes_r_ordered, root=0)
        
        ##############################
        # Radial transformation 
        ########################
        
        if self.mpi_rank==0:
            message("Expanding angular modes in Chebyshev spectrum...", 
                    message_verbosity=2)
            
        # Job list
        radial_decom_job_list = []
        
        for ell in range(self.method_info.ell_max +1):
            for emm in range(-ell, ell+1):
                
                radial_decom_job_list.append([ell, emm])
        
        
        # Construct PClm
        
        #modes_Clmq = SingleMode(ell_max=minfo.ell_max, modes_dim=n_r)

        modes_Clmq_list_local = []

        if self.mpi_rank==0:
            if len(radial_decom_job_list)<self.mpi_nprocs:
                message(f"Using only {len(flat_set_of_coords_groups)}"
                        "processors for radial decomposition", 
                        message_verbosity=2)
                
                
        for jobid, template in enumerate(radial_decom_job_list):

            if jobid%self.mpi_nprocs==self.mpi_rank:
            
                ell, emm = template

                this_Clmq = self.radial_expansion_at_ell_emm(ell, emm)

                message(f'This Clmq l{ell}, m{emm}', 
                        this_Clmq, message_verbosity=3)

                modes_Clmq_list_local.append([ell, emm, this_Clmq])
                
                #if ell==2 and emm==0:
                #    this_r_modes_recon = cs.MatrixSpecToPhys@this_Clmq

                #    delta = np.mean((this_r_modes_recon - this_r_modes)**2)
                #    message(f'Spectral diagnostics l{ell} m{emm} Delta', delta, message_verbosity=2)
                
        
        if self.mpi_rank==0:
            message("Synchronizing", message_verbosity=2)
        
        self.mpi_comm.Barrier()
        
        modes_Clmq_list_group = self.mpi_comm.gather(modes_Clmq_list_local, 
                                                     root=0)
        
        if self.mpi_rank==0:
            message("Finished expanding angular modes"
                    "in Chebyshev spectrum...", 
                    message_verbosity=2)
        #############################
        # Construct the interpolant
        ###########################
        
        
        
        if self.mpi_rank==0:
            self.assign_Clmq_to_modes(modes_Clmq_list_group)
        
        
        self.mpi_comm.Barrier()
        
        if self.mpi_rank==0:
            message("Broadcasting constructed interpolant", 
                    message_verbosity=2)
        
        self._interpolant = self.mpi_comm.bcast(self.interpolant, root=0)
        
        if self.mpi_rank==0:
            message("Successfully broadcasted constructed interpolant", 
                message_verbosity=2)
        
        
        if self.mpi_rank==0:
            message("Finished constructing interpolator...", 
                    message_verbosity=2)
    
    def evaluate_interpolant(self):
        ''' Find the values of the function in spectral space represented
        by its modes `modes_Clmq` on a requested set of coordinates .

        Parameters
        ----------
        modes_Clmq : dict
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
        '''
        if self.mpi_rank==0:
            message("Synchronizing before beginning evaluation routine.. ", 
                message_verbosity=2)
        
        self.mpi_comm.Barrier()
        
        if self.mpi_rank==0:
            message("Finished synchronizing before beginning"
                " evaluation routine.. ", message_verbosity=2)
        
        R, Th, Ph = self.sphp_output_grid #self.sphp_output_grid

        flat_set_of_coords_groups, args_order = ReorganizeCoords(
            self.sphp_output_grid
        )
        
        # Strategy
        ## First, group coords with same r together
        ## Second, evaluate 2d function value over angular elements
        ### within each const r set
        ## Contraction in r required when group is changed
        
        ## For MPI implementation, parallelize over groups
        ## Angular evaluation within a group would be serial

        func_vals_list_local = []
        
        index = 0
        
        #modes_Clmq = self.interpolant
        
        #prev_ri = np.amin(R)

        #prev_Clm_interp = RContract(modes_Clmq, prev_ri, self.r_min, self.r_max, Nfuncs)
        
        if self.mpi_rank==0:
            if len(flat_set_of_coords_groups)<self.mpi_nprocs:
                message(f"Using only {len(flat_set_of_coords_groups)}"
                        "processors for evaluation", message_verbosity=2)
                
        
        if self.mpi_rank==0:
            message(f"Total num of points {len(flat_set_of_coords_groups)}", message_verbosity=1)
            
        #local_count=0
        for jobid, item in enumerate(flat_set_of_coords_groups):
                
            if jobid%self.mpi_nprocs==self.mpi_rank:
                message(f"Job {jobid} executed by rank {self.mpi_rank}", 
                        message_verbosity=3)
                
                
                Rf, Th, Ph = item
                #print("Rf shape")
                #print(Rf, Rf.shape)
                
                ri = Rf[0]

                message('At radius', ri, message_verbosity=4)
                message(f'Angular list Theta {Th} \n Phi {Ph}', message_verbosity=4)
                message('r interpolating', message_verbosity=4)

                Clm_interp = RContract(self.interpolant, 
                                       ri,  
                                       cs=self.radial_grid)
                
                message('R Interpolation done', message_verbosity=4)
                
                fval = AngContract(Clm_interp, Th, Ph)
                print("fval", fval)
                func_vals_list_local.append([jobid, fval])
                
                #local_count+=1
            
                #if local_count%100==0:
                    
                    #total_counts=None
                    #total_counts = self.mpi_comm.reduce(local_count, root=0)
                    
                    #self.mpi_comm.Barrier()
            
                #if self.mpi_rank==0:
                    #print("Progress ", 100*total_counts/len(flat_set_of_coords_groups), "%")
        
        
        if self.mpi_rank==0:
            message("Synchronizing before gathering func values.. ", 
                message_verbosity=2)
        
        self.mpi_comm.Barrier()
        
        # gather
        func_vals_list = self.mpi_comm.gather(func_vals_list_local, root=0)
        
        
        
        if self.mpi_rank==0:
            message("Finished synchronizing after gathering func values.", 
                message_verbosity=2)
        
       
            print("Func vals list", func_vals_list)
            func_vals_list = flatten(func_vals_list)
            
            print("Func vals list", func_vals_list)
            
            message("Reordering func values.. ", message_verbosity=2)
            
            func_vals_jobids = [item[0] for item in func_vals_list]
            
            func_vals_order = np.argsort(func_vals_jobids)
            
            print("Func cals order", func_vals_order)
            print("Func vals shape", np.array(func_vals_list).shape)
            
            # Reorder according to jobs, and also preserve original input order
            func_vals_list_ordered = np.array(func_vals_list, dtype=object)[func_vals_order]
            
            func_values_only = [item[1] for item in func_vals_list]
            
            func_values_only_ordered = [unsort(item, args_order) \
                                        for item in func_values_only]
            
            func_values_only_ordered_flattened = flatten(
                func_values_only_ordered
            )

            func_vals = np.array(func_values_only_ordered_flattened, 
                                 dtype=np.complex128)
            
            
            message("Assiging interpolated data.. ", message_verbosity=2)
            self._interpolated_data =  func_vals.reshape(
                np.array(
                    self.sphp_output_grid[0]
                ).shape
            )
            
            #diff = abs(self.interpolated_data - self.raw_data[0]).flatten()
            
            #plt.plot(diff)
            #plt.savefig("deviation.png")
            #plt.close()
            
            message("Eavluation Done", message_verbosity=2)
        
