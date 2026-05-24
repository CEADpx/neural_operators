import sys
from pathlib import Path

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from petsc4py import PETSc

_pde_path = Path(__file__).resolve().parent.parent / "pde"
if str(_pde_path) not in sys.path:
    sys.path.insert(0, str(_pde_path))
from fenicsUtilities import build_vector_vertex_maps


def _num_dofs(V):
    return len(fem.Function(V).x.array)


class PriorSampler:

    def __init__(self, V, a, c, seed=0):

        self.a = fem.Constant(V.mesh, default_scalar_type(a))
        self.c = fem.Constant(V.mesh, default_scalar_type(c))

        self.seed = seed

        # function space
        self.V = V
        self._a_compiled = None
        self._L_compiled = None
        self._M_compiled = None
        self._ksp = None

        # vertex to dof vector and dof vector to vertex maps
        self.V_vec2vv, self.V_vv2vec = build_vector_vertex_maps(self.V)

        # Source function
        self.s_fn = fem.Function(self.V)
        self.s_dim = _num_dofs(self.V)

        # variational form
        self.u_fn = fem.Function(self.V)
        self.u = None

        self.u_trial = ufl.TrialFunction(self.V)
        self.u_test = ufl.TestFunction(self.V)

        self.b_fn = fem.Function(self.V)
        self.b_fn.x.array[:] = 1.0
        self._update_function_ghosts(self.b_fn)

        self.a_form = (
            self.a
            * self.b_fn
            * ufl.inner(ufl.grad(self.u_trial), ufl.grad(self.u_test))
            * ufl.dx
            + self.c * self.u_trial * self.u_test * ufl.dx
        )
        self.L_form = self.s_fn * self.u_test * ufl.dx

        # assemble matrix and vector
        self.lhs = None
        self.rhs = None
        self.assemble()

        # assemble mass matrix for log-prior
        self.M_mat = assemble_matrix(
            fem.form(self.u_trial * self.u_test * ufl.dx)
        )
        self.M_mat.assemble()

        # compute mean
        self.mean = None
        self.mean_fn = fem.Function(self.V)
        self.mean = self.compute_mean(self.mean)

    def empty_sample(self):
        return np.zeros(self.s_dim)

    def _compile_forms(self):
        if self._a_compiled is None:
            self._a_compiled = fem.form(self.a_form)
            self._L_compiled = fem.form(self.L_form)
            self._M_compiled = fem.form(self.u_trial * self.u_test * ufl.dx)

    def assemble(self):
        self._compile_forms()
        self.lhs = assemble_matrix(self._a_compiled)
        self.lhs.assemble()
        self.rhs = assemble_vector(self._L_compiled)
        self._ksp = None

    def _assemble_rhs(self):
        self._compile_forms()
        self.rhs = assemble_vector(self._L_compiled)

    def _setup_solver(self):
        if self._ksp is None:
            self._ksp = PETSc.KSP().create(self.V.mesh.comm)
            self._ksp.setType(PETSc.KSP.Type.PREONLY)
            self._ksp.getPC().setType(PETSc.PC.Type.LU)
        self._ksp.setOperators(self.lhs)

    def _solve(self, u_fn, rhs=None):
        if rhs is None:
            rhs = self.rhs
        u_fn.x.petsc_vec.set(0.0)
        self._setup_solver()
        self._ksp.solve(rhs, u_fn.x.petsc_vec)

    def _update_function_ghosts(self, fn):
        fn.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    def _petsc_vec_to_array(self, vec):
        vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        return vec.array.copy()

    def _log_prior_from_source(self):
        mass_action = create_vector(self.V)
        self.s_fn.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        self.M_mat.mult(self.s_fn.x.petsc_vec, mass_action)
        return -float(np.sqrt(self.s_fn.x.petsc_vec.dot(mass_action)))

    def function_to_vertex(self, u_fn, u_vv=None):
        values = u_fn.x.array[self.V_vec2vv].copy()
        if u_vv is None:
            return values
        u_vv[:] = values
        return u_vv

    def vertex_to_function(self, u_vv, u_fn=None):
        if u_fn is None:
            u_fn = fem.Function(self.V)
        u_fn.x.array[:] = u_vv[self.V_vv2vec]
        return u_fn

    def function_to_vector(self, u_fn, u_vec=None):
        values = u_fn.x.array.copy()
        if u_vec is None:
            return values
        u_vec[:] = values
        return u_vec

    def vector_to_function(self, u_vec, u_fn=None):
        if u_fn is None:
            u_fn = fem.Function(self.V)
        u_fn.x.array[:] = u_vec
        return u_fn

    def compute_mean(self, m):
        self.s_fn.x.array[:] = 0.0
        self.mean_fn.x.array[:] = 0.0

        # reassemble
        self.assemble()

        # solve
        self._solve(self.mean_fn)

        # vertex_dof ordered
        m = self.mean_fn.x.array[self.V_vec2vv].copy()
        return m

    def set_diffusivity(self, diffusion):
        # assume diffusion is vertex_dof ordered
        self.b_fn.x.array[:] = diffusion[self.V_vv2vec]
        self._update_function_ghosts(self.b_fn)

        # need to recompute quantities including the mean
        self.mean = self.compute_mean(self.mean)

    def __call__(self, m=None):
        # forcing term
        self.s_fn.x.array[:] = np.random.normal(0.0, 1.0, self.s_dim)
        self._update_function_ghosts(self.s_fn)

        # assemble rhs only
        self._assemble_rhs()

        # solve
        self.u_fn.x.array[:] = 0.0
        self._solve(self.u_fn)

        # add mean
        self.u_fn.x.petsc_vec.axpy(1.0, self.mean_fn.x.petsc_vec)

        # vertex_dof ordered
        self.u = self.u_fn.x.array[self.V_vec2vv].copy()

        log_prior = self._log_prior_from_source()

        if m is not None:
            m = self.u.copy()
            return m, log_prior
        return self.u.copy(), log_prior

    def logPrior(self, m):
        self.s_fn.x.array[:] = 0.0

        self.u_fn.x.array[:] = 0.0
        self.u_fn.x.array[:] = m[self.V_vv2vec]
        self._update_function_ghosts(self.u_fn)

        diff = create_vector(self.V)
        diff.array[:] = self.u_fn.x.array - self.mean_fn.x.array
        diff.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        lhs_action = create_vector(self.V)
        self.lhs.mult(diff, lhs_action)
        self.s_fn.x.array[:] = self._petsc_vec_to_array(lhs_action)

        return self._log_prior_from_source()
