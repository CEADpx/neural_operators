import sys
import numpy as np
import dolfin as dl

src_path = "../"
sys.path.append(src_path + 'pde/')
from fenicsUtilities import build_vector_vertex_maps

class PriorSampler:

    def __init__(self, V, a, c, seed = 0):
        
        # Delta and gamma
        self.a = dl.Constant(a)
        self.c = dl.Constant(c)

        self.seed = seed
        
        # function space
        self.V = V

        # vertex to dof vector and dof vector to vertex maps
        self.V_vec2vv, self.V_vv2vec = build_vector_vertex_maps(self.V)

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

    def function_to_vertex(self, u_fn, u_vv = None):
        if u_vv is None:
            return u_fn.vector().get_local()[self.V_vec2vv].copy()
        else:
            u_vv = u_fn.vector().get_local()[self.V_vec2vv].copy()
            return u_vv
            
    def vertex_to_function(self, u_vv, u_fn = None):
        if u_fn is None:
            u_fn = dl.Function(self.V)
            u_fn.vector().set_local(u_vv[self.V_vv2vec])
            return u_fn
        else:
            u_fn.vector().set_local(u_vv[self.V_vv2vec])
            return u_fn
            
    def function_to_vector(self, u_fn, u_vec = None):
        if u_vec is None:
            return u_fn.vector().get_local().copy()
        else:
            u_vec = u_fn.vector().get_local().copy()
            return u_vec
            
    def vector_to_function(self, u_vec, u_fn = None):
        if u_fn is None:
            u_fn = dl.Function(self.V)
            u_fn.vector().set_local(u_vec)
            return u_fn
        else:
            u_fn.vector().set_local(u_vec)
            return u_fn

    def compute_mean(self, m):
        self.s_fn.vector().zero()
        self.mean_fn.vector().zero()
        
        # reassemble
        self.assemble()
        
        # solve
        dl.solve(self.lhs, self.mean_fn.vector(), self.rhs)

        # vertex_dof ordered
        m = self.mean_fn.vector().get_local()[self.V_vec2vv]
        return m

    def set_diffusivity(self, diffusion):

        # assume diffusion is vertex_dof ordered
        self.b_fn.vector().set_local(diffusion[self.V_vv2vec])
        
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
        self.u = self.u_fn.vector().get_local()[self.V_vec2vv]
        
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
        self.u_fn.vector().set_local(m[self.V_vv2vec])
        
        self.s_fn.vector().axpy(1., self.lhs * (self.u_fn.vector() - self.mean_fn.vector()))
        log_prior = -np.sqrt(self.s_fn.vector().inner(self.M_mat * self.s_fn.vector()))

        return log_prior