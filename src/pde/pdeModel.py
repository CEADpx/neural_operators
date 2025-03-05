import numpy as np
import dolfin as dl
from fenicsUtilities import build_vector_vertex_maps

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
        
        # vertex to dof vector and dof vector to vertex maps
        self.Vm_vec2vv, self.Vm_vv2vec = build_vector_vertex_maps(self.Vm)
        self.Vu_vec2vv, self.Vu_vv2vec = build_vector_vertex_maps(self.Vu)

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
    
    def function_to_vertex(self, m_fn, m_vv = None, is_m = True):
        if is_m:
            if m_vv is None:
                return m_fn.vector().get_local()[self.Vm_vec2vv].copy()
            else:
                m_vv = m_fn.vector().get_local()[self.Vm_vec2vv].copy()
                return m_vv
        else:
            if m_vv is None:
                return m_fn.vector().get_local()[self.Vu_vec2vv].copy()
            else:
                m_vv = m_fn.vector().get_local()[self.Vu_vec2vv].copy()
                return m_vv
            
    def vertex_to_function(self, m_vv, m_fn = None, is_m = True):
        if is_m:
            if m_fn is None:
                m_fn = dl.Function(self.Vm)
                m_fn.vector().set_local(m_vv[self.Vm_vv2vec])
                return m_fn
            else:
                m_fn.vector().set_local(m_vv[self.Vm_vv2vec])
                return m_fn
        else:
            if m_fn is None:
                m_fn = dl.Function(self.Vu)
                m_fn.vector().set_local(m_vv[self.Vu_vv2vec])
                return m_fn
            else:
                m_fn.vector().set_local(m_vv[self.Vu_vv2vec])
                return m_fn
            
    def function_to_vector(self, m_fn, m_vec = None, is_m = True):
        if is_m:
            if m_vec is None:
                return m_fn.vector().get_local().copy()
            else:
                m_vec = m_fn.vector().get_local().copy()
                return m_vec
        else:
            if m_vec is None:
                return m_fn.vector().get_local().copy()
            else:
                m_vec = m_fn.vector().get_local().copy()
                return m_vec
            
    def vector_to_function(self, m_vec, m_fn = None, is_m = True):
        if is_m:
            if m_fn is None:
                m_fn = dl.Function(self.Vm)
                m_fn.vector().set_local(m_vec)
                return m_fn
            else:
                m_fn.vector().set_local(m_vec)
                return m_fn
        else:
            if m_fn is None:
                m_fn = dl.Function(self.Vu)
                m_fn.vector().set_local(m_vec)
                return m_fn
            else:
                m_fn.vector().set_local(m_vec)
                return m_fn
    
    def compute_mean(self, m):
        print("compute_mean method not implemented. Should be defined by inherited class.")
        pass

    def solveFwd(self, u = None, m = None, transform_m = False):
        print("solveFwd method not implemented. Should be defined by inherited class.")
        pass

    def samplePrior(self, m = None, transform_m = False):
        print("samplePrior method not implemented. Should be defined by inherited class.")
        pass
        
        