import numpy as np
import dolfin as dl

class PriorSampler:

    def __init__(self, V, a, c, seed = 0):
        
        # Delta and gamma
        self.a = dl.Constant(a)
        self.c = dl.Constant(c)

        self.seed = seed
        
        # function space
        self.V = V

        self.V_v2d = dl.vertex_to_dof_map(self.V)
        self.V_d2v = dl.dof_to_vertex_map(self.V)

        # Source function
        self.s_fn = dl.Function(self.V)
        self.s_dim = self.s_fn.vector().size()

        # variational form
        self.u_fn = dl.Function(self.V)
        self.u = None
        
        self.u_trial = dl.TrialFunction(self.V)
        self.u_test = dl.TestFunction(self.V)

        self.b_fn = dl.Function(self.V)
        self.b_fn.vector().set_local(np.ones(self.V.dim()))
        
        self.a_form = self.a*self.b_fn\
                        *dl.inner(dl.nabla_grad(self.u_trial), \
                                  dl.nabla_grad(self.u_test))*dl.dx \
                    + self.c*self.u_trial*self.u_test*dl.dx
        self.L_form = self.s_fn*self.u_test*dl.dx
        
        # assemble matrix and vector
        self.lhs = None
        self.rhs = None 
        self.assemble()

        # assemble mass matrix for log-prior
        self.M_mat = dl.assemble(self.u_trial*self.u_test*dl.dx)

        # compute mean
        self.mean = None
        self.mean_fn = dl.Function(self.V)
        self.mean = self.compute_mean(self.mean)

    def empty_sample(self):
        return np.zeros(self.V.dim())
    
    def assemble(self):
        self.lhs = dl.assemble(self.a_form)
        self.rhs = dl.assemble(self.L_form)

    def compute_mean(self, m):
        self.s_fn.vector().zero()
        self.mean_fn.vector().zero()
        
        # reassemble
        self.assemble()
        
        # solve
        dl.solve(self.lhs, self.mean_fn.vector(), self.rhs)

        # vertex_dof ordered
        m = self.mean_fn.vector().get_local()[self.V_v2d]
        return m

    def set_diffusivity(self, diffusion):

        # assume diffusion is vertex_dof ordered
        self.b_fn.vector().set_local(diffusion[self.V_d2v])
        
        # need to recompute quantities including the mean
        self.mean = self.compute_mean(self.mean)

    def __call__(self, m = None):

        # forcing term
        self.s_fn.vector().zero()
        self.s_fn.vector().set_local(np.random.normal(0.,1.,self.s_dim))

        # assemble (no need to reassemble A) --> if diffusion is changed, then A would have been assembled at that time
        self.rhs = dl.assemble(self.L_form)
        
        # solve
        self.u_fn.vector().zero()
        dl.solve(self.lhs, self.u_fn.vector(), self.rhs)

        # add mean
        self.u_fn.vector().axpy(1., self.mean_fn.vector())

        # vertex_dof ordered
        self.u = self.u_fn.vector().get_local()[self.V_v2d]
        
        # compute log-prior
        log_prior = -np.sqrt(self.s_fn.vector().inner(self.M_mat * self.s_fn.vector()))

        if m is not None:
            m = self.u.copy()
            return m, log_prior
        else:
            return self.u.copy(), log_prior
    
    def logPrior(self, m):
        # CHECK!
        # If the forcing term f is known, then log-prior is straightforward but for general m, we need to compute <C^{-1}(m - mean), C^{-1}(m - mean)>_L2, C = A^{-2}, A being a differential operator
        self.s_fn.vector().zero()

        self.u_fn.vector().zero()
        self.u_fn.vector().set_local(m[self.V_d2v])
        
        self.s_fn.vector().axpy(1., self.lhs * (self.u_fn.vector() - self.mean_fn.vector()))
        log_prior = -np.sqrt(self.s_fn.vector().inner(self.M_mat * self.s_fn.vector()))

        return log_prior