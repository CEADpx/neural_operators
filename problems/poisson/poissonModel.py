import sys
import numpy as np
import dolfin as dl

# local utility methods
util_path = "../../utilities/"
sys.path.append(util_path)
from priorSampler import PriorSampler

class PoissonModel:
    
    def __init__(self, Vm, Vu, \
                 prior_sampler, \
                 logn_scale = 1., logn_translate = 0., \
                 seed = 0):
        
        self.seed = seed

        # prior and transform parameters
        self.prior_sampler = prior_sampler
        self.logn_scale = logn_scale
        self.logn_translate = logn_translate
        
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

        # Boundary conditions
        self.f = dl.Expression("1000*(1-x[1])*x[1]*(1-x[0])*(1-x[0])", degree=2)
        self.q = dl.Expression("50*sin(5*pi*x[1])", degree=2)

        # store transformed m where input is from Gaussian prior
        self.m_transformed = None 
        self.m_mean = None
        self.m_mean = self.compute_mean(self.m_mean)

        # input and output functions (will be updated in solveFwd)
        self.m_fn = dl.Function(self.Vm)
        self.u_fn = dl.Function(self.Vu)
        self.m_fn.vector().set_local(self.m_mean[self.Vm_d2v])

        # variational form
        self.u_trial = dl.TrialFunction(self.Vu)
        self.u_test = dl.TestFunction(self.Vu)
        
        self.a_form = self.m_fn*dl.inner(dl.nabla_grad(self.u_trial), dl.nabla_grad(self.u_test))*dl.dx 
        self.L_form = self.f*self.u_test*dl.dx \
                 + self.q*self.u_test*dl.ds # boundary term
        
        self.bc = [dl.DirichletBC(self.Vu, dl.Constant(0), self.boundaryU)]

        # assemble matrix and vector
        self.lhs = None
        self.rhs = None 
        self.assemble()
    
    def assemble(self):
        self.lhs = dl.assemble(self.a_form)
        self.rhs = dl.assemble(self.L_form)
        for bc in self.bc:
            bc.apply(self.lhs, self.rhs)

    def empty_u(self):
        return np.zeros(self.u_dim)
    
    def empty_m(self):
        return np.zeros(self.m_dim)
    
    def compute_mean(self, m):
        return self.transform_gaussian_pointwise(self.prior_sampler.mean, m)

    @staticmethod
    def boundaryU(x, on_boundary):
        return on_boundary and x[0] < 1. - 1e-10
    
    @staticmethod
    def is_point_on_dirichlet_boundary(x):
        # locate boundary nodes
        tol = 1.e-10
        if np.abs(x[0]) < tol \
            or np.abs(x[1]) < tol \
            or np.abs(x[0] - 1.) < tol \
            or np.abs(x[1] - 1.) < tol:
            # select all boundary nodes except the right boundary
            if x[0] < 1. - tol:
                return True
        return False
    
    def transform_gaussian_pointwise(self, w, m_local = None):
        if m_local is None:
            self.m_transformed = self.logn_scale*np.exp(w) + self.logn_translate
            return self.m_transformed.copy()
        else:
            m_local = self.logn_scale*np.exp(w) + self.logn_translate
            return m_local

    def solveFwd(self, u = None, m = None, transform_m = False):

        if m is None:
            m = self.samplePrior()
        
        # see if we need to transform m vector (it is vertex_dof ordered)
        if transform_m:
            self.m_transformed = self.transform_gaussian_pointwise(m, self.m_transformed)
        else:
            self.m_transformed = m

        # set m
        self.m_fn.vector().zero()
        self.m_fn.vector().set_local(self.m_transformed[self.Vm_d2v])

        # reassamble (don't need to reassemble L)
        self.lhs = dl.assemble(self.a_form)
        # self.rhs = dl.assemble(self.L_form)
        for bc in self.bc:
            # bc.apply(self.lhs, self.rhs)
            bc.apply(self.lhs)
        
        # solve
        dl.solve(self.lhs, self.u_fn.vector(), self.rhs)

        if u is not None:
            # set the solution (dolfin vector to vertex_dof ordered vector)
            u = self.u_fn.vector().get_local()[self.Vu_v2d] 
            return u
        else:
            return self.u_fn.vector().get_local()[self.Vu_v2d]

    def samplePrior(self, m = None, transform_m = False):
        if transform_m:
            self.m_transformed = self.transform_gaussian_pointwise(self.prior_sampler()[0], self.m_transformed)
        else:
            self.m_transformed = self.prior_sampler()[0]

        if m is None:
            return self.m_transformed.copy()
        else:
            m = self.m_transformed.copy()
            return m
        
        