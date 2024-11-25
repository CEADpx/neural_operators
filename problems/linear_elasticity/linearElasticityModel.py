import sys
import numpy as np
import dolfin as dl

# local utility methods
util_path = "../../utilities/"
sys.path.append(util_path)
from priorSampler import PriorSampler

class LinearElasticityModel:
    
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
        self.b = dl.Constant((0, 0))
        self.t = dl.Constant((0, 10))
        
        self.bc = [dl.DirichletBC(self.Vu, dl.Constant((0,0)), self.boundaryU)]

        facets = dl.MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        dl.AutoSubDomain(self.boundaryQ).mark(facets, 1)
        self.ds = dl.Measure("ds", domain=self.mesh, subdomain_data=facets)

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

        self.nu = 0.25
        self.lam_fact = dl.Constant(self.nu / (1+self.nu)*(1-2*self.nu))
        self.mu_fact = dl.Constant(1/(2*(1+self.nu)))

        self.spatial_dim = self.u_fn.geometric_dimension()
        self.a_form = self.m_fn*dl.inner(self.lam_fact*dl.tr(dl.grad(self.u_trial))*dl.Identity(self.spatial_dim) \
                                        + 2*self.mu_fact * dl.sym(dl.grad(self.u_trial)), \
                                    dl.sym(dl.grad(self.u_test)))*dl.dx
        
        self.L_form = dl.inner(self.b, self.u_test)*dl.dx + dl.inner(self.t, self.u_test)*self.ds

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
        return on_boundary and dl.near(x[0], 0.)
    
    @staticmethod
    def boundaryQ(x, on_boundary):
        return on_boundary and dl.near(x[0], 1.)
    
    @staticmethod
    def is_point_on_dirichlet_boundary(x):
        # locate boundary nodes
        tol = 1.e-10
        if np.abs(x[0]) < tol \
            or np.abs(x[1]) < tol \
            or np.abs(x[0] - 1.) < tol \
            or np.abs(x[1] - 1.) < tol:
            # select left boundary
            if x[0] < tol:
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
            u = self.u_fn.compute_vertex_values().copy()
            return u
        else:
            return self.u_fn.compute_vertex_values().copy()

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
        
        