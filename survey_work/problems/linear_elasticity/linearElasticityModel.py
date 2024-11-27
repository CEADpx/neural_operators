import sys
import numpy as np
import dolfin as dl

# local utility methods
src_path = "../../../src/"
sys.path.append(src_path + 'pde/')
from pdeModel import PDEModel

class LinearElasticityModel(PDEModel):
    
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
        self.b = dl.Constant((0, 0))
        self.t = dl.Constant((0, 10))
        
        self.bc = [dl.DirichletBC(self.Vu, dl.Constant((0,0)), self.boundaryU)]

        facets = dl.MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        dl.AutoSubDomain(self.boundaryQ).mark(facets, 1)
        self.ds = dl.Measure("ds", domain=self.mesh, subdomain_data=facets)

        # store transformed m where input is from Gaussian prior
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
        self.assemble()

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
        
        