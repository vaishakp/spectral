import numpy as np
import sympy as sp
from waveformtools.waveformtools import message

""" Deals with Chebyshev approximations of the first kind """

#Nmax = 25
#from numba import jit, njit
#cheb_basis_array = np.zeros(Nmax)

#@njit(parallel=True)


class ChebyshevBasis():
    ''' A chebyshev basis class comprising of polynomials of the first kind '''
    def __init__(self,
                Nfuncs=8,
                basis_calc_method='memoize',
                ):
        
        self._Nfuncs = Nfuncs
        self._basis_storage = {}
        self._basis_der_storage = {}
        self._basis_calc_method = basis_calc_method
        
        
        if self._basis_calc_method=='memoize':
            self.ChebBasis = self.ChebBasisMem
            self.ChebBasisEval = self.ChebBasisDirect
            
        elif self._basis_calc_method=="direct":
            self.ChebBasis = self.ChebBasisDirect
            self.ChebBasisEval = self.ChebBasisDirect
            
        elif self._basis_calc_method=='recursive':
            self.ChebBasis = self.ChebBasisRec
            self.ChebBasisEval = self.ChebBasisRec
            
        else:
            raise KeyError("Unknown basis calculation method", self._basis_calc_method)
            
    @property
    def basis_storage(self):
        return self._basis_storage
    
    @property
    def basis_der_storage(self):
        return self._basis_der_storage
    
    @property
    def Nfuncs(self):
        return self._Nfuncs
    
    def CollocationPoints(self):
        Naxis = np.arange(self.Nfuncs)
        #Naxis = np.arange()

        return -np.cos(np.pi*Naxis/(self.Nfuncs-1))
        #return np.cos(2*)

    #@njit(parallel=True)
    def MapToAB(self, x_axis, a, b):

        return a+ (b-a)*(1+x_axis)/2
    
    #@njit(parallel=True, cache=True)
    def ChebBasisDirect(self, x_axis, order):
        ''' Return the chebyshev polynomial of First kind
        of order `order` '''
        #print(x_axis)
        
        return np.cos(order*np.arccos(x_axis))

    #@njit(parallel=True, cache=True)
    
    def ChebBasisRec(self, x_axis, order):
        ''' Return the chebyshev basis polynomial of First kind
        of order `order` '''
        
        x_axis = np.array(x_axis)
        
        if order==0:
            return np.ones(len(x_axis))

        if order==1:
            return x_axis

        else:
            return 2*x_axis*self.ChebBasisRec(x_axis, order-1) - self.ChebBasisRec(x_axis, order-2)


    def ChebBasisMem(self, x_axis, order):
        ''' Return the chebyshev basis polynomial of First kind
        of order `order` '''


        if order not in list(self.basis_storage.keys()):

            message(f"Constructing basis of order {order}", message_verbosity=3)
        
            if order==0:
                self._basis_storage.update({0 : np.ones(len(x_axis))})

            elif order==1:
                self._basis_storage.update({1 : x_axis})

            else:
                self._basis_storage.update({order : 2*x_axis*self.ChebBasisMem(x_axis, order-1) - self.ChebBasisMem(x_axis, order-2)})

        return self.basis_storage[order]
    
    #@njit(parallel=True)
    def ToPhys(self, x_axis, u_spec):
        ''' Transformation a vector from Chybyshev spectral 
        to physical space '''

        Nmax = len(u_spec)
        
        u_coord = np.zeros(Nmax)

        for order in range(Nmax):

            u_coord+=u_spec(order)*self.ChebBasis(x_axis, order)

        return u_coord

    #@njit(parallel=True)
    def ToPhysMatrix(self, x_axis):
        ''' Transformation matrix from physical 
        to Chebyshev spectral space '''

        Nmax = len(x_axis)
        #matrix = np.array([Nmax, Nmax], dtype=np.float64)

        matrix = np.zeros((Nmax, Nmax), dtype=np.float64)
        #spec_vec = np.zeros(Nmax)

        for order in range(Nmax):

            u_coord=self.ChebBasis(x_axis, order)

            matrix[:, order] = u_coord[:]    
            #atrix = np.concatenate((matrix, u_coord))

        return matrix

    #@njit(parallel=False)
    def ToSpecMatrix(self, x_axis):
        ''' Transformation matrix from the physical
        to spectral space '''
        return np.linalg.inv(self.ToPhysMatrix(x_axis))

    #@njit(parallel=True)
    def ToSpecMatrixDirect(self, x_axis):
        ''' Transformation matrix from the physical
        to spectral space directly using Gaussian quadrature
        over products of basis functions '''
        Nmax = len(x_axis)

        #cbar = 0.5*np.ones(Nmax)*(Nmax-1)
        #cbar = 0.5*np.ones(Nmax)*(Nmax+1)
        cbar = np.ones(Nmax)

        #cbar[0] = 1*(Nmax)
        #cbar[-1] = 1*(Nmax)

        cbar[0] = 2
        cbar[-1] = 2

        #cnorm = Nmax * np.ones((Nmax))/2
        #cnorm[0] = Nmax
        cnorm = (Nmax-1)*np.ones(Nmax)/2
        cnorm[0] = Nmax-1
        cnorm[-1] = Nmax-1

        Tmatrix = np.zeros((Nmax, Nmax))

        for index_i in range(Nmax):

            phys_basis_i = np.zeros(Nmax)
            phys_basis_i[index_i] = 1

            for index_j in range(Nmax):

                cheb_basis_j =  self.ChebBasis(x_axis, index_j)

                Cij = np.dot(phys_basis_i, cheb_basis_j/cbar)

                Tmatrix[index_i, index_j] = Cij/cnorm[index_j]

        return Tmatrix.T

    #@njit(parallel=True, cache=True)
    def ChebBasisDer(self, x_axis, order):
        ''' Compute and return the derivative 
        vector of a Chebyshev Basis function '''

        Nmax = len(x_axis)

        assert Nmax==self.Nfuncs, "The input Npoints does not agree with basis Nfuncs"
        
        if order not in list(self.basis_der_storage.keys()):

            message(f"Constructing basis derivative of order {order}", message_verbosity=3)

            if order==0:
                self._basis_der_storage.update({0:np.zeros(Nmax)})

            elif order==1:
                self._basis_der_storage.update({1:np.ones(Nmax)})

            else:
                self._basis_der_storage.update({ 
                                                order : 2*self.ChebBasis(x_axis, order-1)  
                                               + 2*x_axis*self.ChebBasisDer(x_axis, order-1) 
                                               - self.ChebBasisDer(x_axis, order-2) 
                                              })

        return self.basis_der_storage[order]

    #@njit(parallel=True)
    def ChebDerSpecToPhysMatrix(self, x_axis):
        ''' The operator to compute the derivative 
        of a vector in spectral space, returning a
        vector in physical space.
        
        Takes in spectral
        Gives out physical
        
        '''

        Nmax = len(x_axis)

        der_matrix = np.zeros((Nmax, Nmax))

        # Derivative of Basis vectors as the columns
        for order in range(Nmax):

            this_col = self.ChebBasisDer(x_axis, order)

            der_matrix[:, order] = this_col

        return der_matrix

    #@njit(parallel=False)
    def ChebDerPhysToPhysMatrix(self, x_axis):
        ''' The operator to compute the derivative 
        of a vector in physical space, returning a
        vector in physical space '''

        der_mat_spec_to_phys = self.ChebDerSpecToPhysMatrix(x_axis)

        # t_matrix_spec_to_coord = self.ToPhysMatrix(x_axis)

        t_matrix_coord_to_spec = self.ToSpecMatrix(x_axis)


        return der_mat_spec_to_phys@t_matrix_coord_to_spec

    #ChebBasis = ChebBasisMem
    #ChebBasis = ChebBasisDirect
    #ChebBasis = ChebBasisRec