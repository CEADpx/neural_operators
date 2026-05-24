import sys
from pathlib import Path

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    set_bc,
)
from petsc4py import PETSc

_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_root / "src" / "pde"))
from pdeModel import PDEModel


class PoissonModel(PDEModel):

    def __init__(
        self,
        Vm,
        Vu,
        prior_sampler,
        logn_scale=1.0,
        logn_translate=0.0,
        seed=0,
    ):
        super().__init__(Vm, Vu, prior_sampler, seed)

        self.logn_scale = logn_scale
        self.logn_translate = logn_translate

        domain = self.mesh
        x = ufl.SpatialCoordinate(domain)
        qd = {"quadrature_degree": 4}
        dx = ufl.Measure("dx", domain=domain, metadata=qd)
        ds = ufl.Measure("ds", domain=domain, metadata=qd)
        self.f_expr = 1000 * (1 - x[1]) * x[1] * (1 - x[0]) * (1 - x[0])
        self.q_expr = 50 * ufl.sin(5 * ufl.pi * x[1])

        self.m_mean = self.compute_mean(self.m_mean)

        self.m_fn = fem.Function(self.Vm)
        self.vertex_to_function(self.m_mean, self.m_fn, is_m=True)
        self._update_ghosts(self.m_fn)

        self.u_fn = fem.Function(self.Vu)

        self.u_trial = ufl.TrialFunction(self.Vu)
        self.u_test = ufl.TestFunction(self.Vu)

        self.a_form = (
            self.m_fn
            * ufl.inner(ufl.grad(self.u_trial), ufl.grad(self.u_test))
            * dx
        )
        self.L_form = self.f_expr * self.u_test * dx + self.q_expr * self.u_test * ds

        fdim = domain.topology.dim - 1
        domain.topology.create_connectivity(fdim, domain.topology.dim)
        dofs = fem.locate_dofs_geometrical(self.Vu, self._dirichlet_boundary)
        self.bc = [fem.dirichletbc(default_scalar_type(0.0), dofs, self.Vu)]

        self._a_compiled = None
        self._L_compiled = None
        self._ksp = None

        self.assemble()

    @staticmethod
    def _dirichlet_boundary(x):
        """Left, bottom, and top boundaries — exclude the right edge (legacy ``boundaryU``)."""
        tol = 1e-10
        not_right = x[0] < 1.0 - tol
        return not_right & (
            np.isclose(x[0], 0.0, atol=tol)
            | np.isclose(x[1], 0.0, atol=tol)
            | np.isclose(x[1], 1.0, atol=tol)
        )

    @staticmethod
    def boundaryU(x, on_boundary):
        return on_boundary and x[0] < 1.0 - 1e-10

    @staticmethod
    def is_point_on_dirichlet_boundary(x):
        tol = 1e-10
        if (
            np.abs(x[0]) < tol
            or np.abs(x[1]) < tol
            or np.abs(x[0] - 1.0) < tol
            or np.abs(x[1] - 1.0) < tol
        ):
            if x[0] < 1.0 - tol:
                return True
        return False

    def _compile_forms(self):
        if self._a_compiled is None:
            self._a_compiled = fem.form(self.a_form)
            self._L_compiled = fem.form(self.L_form)

    def _update_ghosts(self, fn):
        fn.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    def _setup_solver(self):
        if self._ksp is None:
            self._ksp = PETSc.KSP().create(self.mesh.comm)
            self._ksp.setType(PETSc.KSP.Type.PREONLY)
            self._ksp.getPC().setType(PETSc.PC.Type.LU)
        self._ksp.setOperators(self.lhs)

    def assemble(self, assemble_lhs=True, assemble_rhs=True):
        self._compile_forms()

        if assemble_lhs or self.lhs is None:
            self.lhs = assemble_matrix(self._a_compiled, bcs=self.bc)
            self.lhs.assemble()
            self._ksp = None

        if assemble_rhs or self.rhs is None:
            self.rhs = assemble_vector(self._L_compiled)
            apply_lifting(self.rhs, [self._a_compiled], bcs=[self.bc])
            self.rhs.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            set_bc(self.rhs, self.bc)

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

        self.assemble(assemble_lhs=True, assemble_rhs=False)

        self.u_fn.x.petsc_vec.set(0.0)
        self._setup_solver()
        self._ksp.solve(self.rhs, self.u_fn.x.petsc_vec)
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
        m[:] = self.m_transformed
        return m
