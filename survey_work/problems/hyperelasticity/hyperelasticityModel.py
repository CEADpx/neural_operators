import os
import sys

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, log
from dolfinx.fem.petsc import NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_root, "src", "pde"))
from pdeModel import PDEModel


class HyperelasticityModel(PDEModel):
    """
    2D compressible hyperelasticity on the unit square (Neo-Hookean energy).

    Same setup as ``LinearElasticityModel``: uncertain scalar Young's modulus
    field ``m(x)``, left-edge Dirichlet clamp ``u = 0`` on ``Γ_{u_d}``, body force zero,
    and uniform traction on ``Γ_{u_q} = ∂D_u \\ Γ_{u_d}``. Strain energy is the stable
    compressible Neo-Hookean form ``(μ/2)(I₁ - 3 - 2 ln J) + (λ/2)(ln J)²``.
    Uses a Newton solve each forward evaluation. When ``reset_u`` is True,
    traction is ramped in ``n_load_steps`` increments for robust convergence.
    """

    def __init__(
        self,
        Vm,
        Vu,
        prior_sampler,
        logn_scale=1.0,
        logn_translate=0.0,
        seed=0
    ):
        super().__init__(Vm, Vu, prior_sampler, seed)

        self.logn_scale = logn_scale
        self.logn_translate = logn_translate
        self.nu = 0.45
        self.lam_fact = self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu_fact = 1.0 / (2 * (1 + self.nu))

        # Newton solver parameters
        self.newton_rtol = 1e-8
        self.newton_atol = 1e-8
        self.newton_max_it = 50
        self.reset_u = True
        self.n_load_steps = 20

        qd = {"quadrature_degree": 4}
        dx = ufl.Measure("dx", domain=self.mesh, metadata=qd)
        ds = ufl.Measure("ds", domain=self.mesh, metadata=qd)

        # External body force and boundary traction
        self.b = ufl.as_vector((0.0, 0.0))
        self._traction_x = fem.Constant(self.mesh, default_scalar_type(20.0))
        self._traction_y = fem.Constant(self.mesh, default_scalar_type(50.0))
        self.t = ufl.as_vector((self._traction_x, self._traction_y))

        dofs = fem.locate_dofs_geometrical(Vu, self._dirichlet_boundary)
        zero = np.zeros(2, dtype=default_scalar_type)
        self.bc = [fem.dirichletbc(zero, dofs, Vu)]

        self.m_mean = self.compute_mean(self.m_mean)

        self.m_fn = fem.Function(self.Vm)
        self.vertex_to_function(self.m_mean, self.m_fn, is_m=True)
        self._update_ghosts(self.m_fn)

        self.u_fn = fem.Function(self.Vu)
        self.u_test = ufl.TestFunction(self.Vu)

        # variational form
        spatial_dim = self.mesh.geometry.dim
        I = ufl.variable(ufl.Identity(spatial_dim))
        F = ufl.variable(I + ufl.grad(self.u_fn))
        C = ufl.variable(F.T * F)
        Ic = ufl.variable(ufl.tr(C))
        J = ufl.variable(ufl.det(F))

        mu = self.m_fn * self.mu_fact
        lam = self.m_fn * self.lam_fact
        # Stable compressible Neo-Hookean
        W = (mu / 2.0) * (Ic - 3.0 - 2.0 * ufl.ln(J)) + (lam / 2.0) * (ufl.ln(J)) ** 2
        P = ufl.diff(W, F)

        self._residual_form = (
            ufl.inner(self.b, self.u_test) * dx
            + ufl.inner(self.t, self.u_test) * ds
            - ufl.inner(ufl.grad(self.u_test), P) * dx
        )

        self._nonlinear_problem = None
        self._newton_solver = None

    @staticmethod
    def _dirichlet_boundary(x):
        return np.isclose(x[0], 0.0, atol=1e-10)

    @staticmethod
    def is_point_on_dirichlet_boundary(x):
        tol = 1e-10
        if (
            np.abs(x[0]) < tol
            or np.abs(x[1]) < tol
            or np.abs(x[0] - 1.0) < tol
            or np.abs(x[1] - 1.0) < tol
        ):
            if x[0] < tol:
                return True
        return False

    def _update_ghosts(self, fn):
        fn.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    def _setup_solver(self):
        if self._nonlinear_problem is None:
            log.set_log_level(log.LogLevel.WARNING)
            self._nonlinear_problem = NewtonSolverNonlinearProblem(
                self._residual_form,
                self.u_fn,
                bcs=self.bc,
            )
            self._newton_solver = NewtonSolver(self.mesh.comm, self._nonlinear_problem)
            self._newton_solver.rtol = self.newton_rtol
            self._newton_solver.atol = self.newton_atol
            self._newton_solver.max_it = self.newton_max_it
            self._newton_solver.convergence_criterion = "incremental"

    def _run_newton(self):
        self._setup_solver()
        n_iter, converged = self._newton_solver.solve(self.u_fn)
        if not converged:
            raise RuntimeError(
                f"Hyperelasticity Newton solver did not converge ({n_iter} iterations)."
            )
        return n_iter

    def set_traction(self, tx, ty):
        self._traction_x.value = tx
        self._traction_y.value = ty

    def assemble(self, assemble_lhs=True, assemble_rhs=True):
        """No-op for API compatibility with linear models."""

    def transform_gaussian_pointwise(self, w, m_local=None):
        if m_local is None:
            self.m_transformed = self.logn_scale * np.exp(w) + self.logn_translate
            return self.m_transformed.copy()
        return self.logn_scale * np.exp(w) + self.logn_translate

    def compute_mean(self, m):
        return self.transform_gaussian_pointwise(self.prior_sampler.mean, m)

    def solveFwd(self, u=None, m=None, transform_m=False):
        if m is None:
            m = self.samplePrior()

        if transform_m:
            self.m_transformed = self.transform_gaussian_pointwise(
                m, self.m_transformed
            )
        else:
            self.m_transformed = m

        self.vertex_to_function(self.m_transformed, self.m_fn, is_m=True)
        self._update_ghosts(self.m_fn)

        if (
            abs(float(self._traction_x.value)) < 1e-14
            and abs(float(self._traction_y.value)) < 1e-14
        ):
            self.u_fn.x.array[:] = 0.0
            self._update_ghosts(self.u_fn)
            return self.function_to_vertex(self.u_fn, u, is_m=False)

        if self.reset_u:
            self.u_fn.x.array[:] = 0.0

        target_tx = float(self._traction_x.value)
        target_ty = float(self._traction_y.value)
        n_steps = self.n_load_steps if self.reset_u else 1

        for step in range(1, n_steps + 1):
            if n_steps > 1:
                load_frac = step / n_steps
                self._traction_x.value = target_tx * load_frac
                self._traction_y.value = target_ty * load_frac
            self._run_newton()

        self._traction_x.value = target_tx
        self._traction_y.value = target_ty
        self._update_ghosts(self.u_fn)

        return self.function_to_vertex(self.u_fn, u, is_m=False)

    def samplePrior(self, m=None, transform_m=False):
        if transform_m:
            w, _ = self.prior_sampler()
            self.m_transformed = self.transform_gaussian_pointwise(
                w, self.m_transformed
            )
        else:
            self.m_transformed = self.prior_sampler()[0]

        if m is None:
            return self.m_transformed.copy()
        m = self.m_transformed.copy()
        return m
