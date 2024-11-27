import sys
import numpy as np
import dolfin as dl

class PDEModel:
    
    def __init__(self, Vm, Vu, \
                 prior_sampler, seed = 0):
        
        self.seed = seed

        # prior and transform parameters
        self.prior_sampler = prior_sampler
        
        # FE setup
        self.Vm = Vm
        self.Vu = Vu
        
        self.mesh = self.Vm.mesh()
        self.m_nodes = self.mesh.coordinates()
        self.u_nodes = self.m_nodes
        
        self.Vm_v2d = dl.vertex_to_dof_map(self.Vm)
        self.Vm_d2v = dl.dof_to_vertex_map(self.Vm)

        self.Vu_v2d = dl.vertex_to_dof_map(self.Vu)
        self.Vu_d2v = dl.dof_to_vertex_map(self.Vu)

        self.m_dim = self.Vm.dim()
        self.u_dim = self.Vu.dim()

        # store transformed m where input is from Gaussian prior
        self.m_transformed = None 
        self.m_mean = None
        
        # input and output functions (will be updated in solveFwd)
        self.m_fn = None
        self.u_fn = None

        # variational form
        self.u_trial = None
        self.u_test = None
        
        self.a_form = None
        self.L_form = None
        
        self.bc = None

        # assemble matrix and vector
        self.lhs = None
        self.rhs = None 

    @staticmethod
    def boundaryU(x, on_boundary):
        print("boundaryU method not implemented. Should be defined by inherited class.")
        pass
    
    @staticmethod
    def is_point_on_dirichlet_boundary(x):
        print("is_point_on_dirichlet_boundary method not implemented. Should be defined by inherited class.")
        pass

    def assemble(self, assemble_lhs = True, assemble_rhs = True):
        print("assemble method not implemented. Should be defined by inherited class.")
        pass

    def empty_u(self):
        return np.zeros(self.u_dim)
    
    def empty_m(self):
        return np.zeros(self.m_dim)
    
    def compute_mean(self, m):
        print("compute_mean method not implemented. Should be defined by inherited class.")
        pass

    def solveFwd(self, u = None, m = None, transform_m = False):
        print("solveFwd method not implemented. Should be defined by inherited class.")
        pass

    def samplePrior(self, m = None, transform_m = False):
        print("samplePrior method not implemented. Should be defined by inherited class.")
        pass
        
        