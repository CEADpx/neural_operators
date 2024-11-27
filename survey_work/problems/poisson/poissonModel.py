import sys
import numpy as np
import dolfin as dl

# local utility methods
src_path = "../../../src/"
sys.path.append(src_path + 'pde/')
from pdeModel import PDEModel

class PoissonModel(PDEModel):
    
    def __init__(self, Vm, Vu, \
                 prior_sampler, \
                 logn_scale = 1., \
                 logn_translate = 0., \
                 seed = 0):
        
        super().__init__(Vm, Vu, prior_sampler, seed)

        # prior transform parameters
        self.logn_scale = logn_scale
        self.logn_translate = logn_translate
        
        # Boundary conditions
        self.f = dl.Expression("1000*(1-x[1])*x[1]*(1-x[0])*(1-x[0])", degree=2)
        self.q = dl.Expression("50*sin(5*pi*x[1])", degree=2)

        # store transformed m where input is from Gaussian prior
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
        self.assemble()
    
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
   
    def assemble(self, assemble_lhs = True, assemble_rhs = True):
        if assemble_lhs or self.lhs is None:
            self.lhs = dl.assemble(self.a_form)
        if assemble_rhs or self.rhs is None:
            self.rhs = dl.assemble(self.L_form)

        for bc in self.bc:
            if assemble_lhs and assemble_rhs:
                bc.apply(self.lhs, self.rhs)
            elif assemble_rhs:
                bc.apply(self.rhs)
            elif assemble_lhs:
                bc.apply(self.lhs)

    def transform_gaussian_pointwise(self, w, m_local = None):
        if m_local is None:
            self.m_transformed = self.logn_scale*np.exp(w) + self.logn_translate
            return self.m_transformed.copy()
        else:
            m_local = self.logn_scale*np.exp(w) + self.logn_translate
            return m_local

    def compute_mean(self, m):
        return self.transform_gaussian_pointwise(self.prior_sampler.mean, m) 
    
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
        self.assemble(assemble_lhs = True, assemble_rhs = False)
        
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
        
        