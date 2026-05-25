import os
import sys

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

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_root, "src", "pde"))
from pdeModel import PDEModel


class LinearElasticityModel(PDEModel):

    def __init__(
        self,
        Vm,
        Vu,
        prior_sampler,
        logn_scale=1.0,
        logn_translate=0.0,
        nu=0.45,
        seed=0
    ):
        super().__init__(Vm, Vu, prior_sampler, seed)

        self.logn_scale = logn_scale
        self.logn_translate = logn_translate
        self.nu = nu

        domain = self.mesh
        qd = {"quadrature_degree": 4}
        dx = ufl.Measure("dx", domain=domain, metadata=qd)

        fdim = domain.topology.dim - 1
        domain.topology.create_connectivity(fdim, domain.topology.dim)

        # Legacy FEniCS used Measure("ds") without a subdomain index (traction on
        # the full exterior boundary), not ds(1) on the marked right edge only.
        ds = ufl.Measure("ds", domain=domain, metadata=qd)

        self.b = ufl.as_vector((0.0, 0.0))
        self._traction_x = fem.Constant(domain, default_scalar_type(20.0))
        self._traction_y = fem.Constant(domain, default_scalar_type(50.0))
        self.t = ufl.as_vector((self._traction_x, self._traction_y))

        dofs = fem.locate_dofs_geometrical(Vu, self._dirichlet_boundary)
        zero = np.zeros(2, dtype=default_scalar_type)
        self.bc = [fem.dirichletbc(zero, dofs, Vu)]

        self.m_mean = self.compute_mean(self.m_mean)

        self.m_fn = fem.Function(self.Vm)
        self.vertex_to_function(self.m_mean, self.m_fn, is_m=True)
        self._update_ghosts(self.m_fn)

        self.u_fn = fem.Function(self.Vu)

        self.u_trial = ufl.TrialFunction(self.Vu)
        self.u_test = ufl.TestFunction(self.Vu)

        self.lam_fact = self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu_fact = 1.0 / (2 * (1 + self.nu))

        spatial_dim = domain.geometry.dim
        I = ufl.Identity(spatial_dim)
        self.a_form = (
            self.m_fn
            * ufl.inner(
                self.lam_fact * ufl.tr(ufl.grad(self.u_trial)) * I
                + 2 * self.mu_fact * ufl.sym(ufl.grad(self.u_trial)),
                ufl.sym(ufl.grad(self.u_test)),
            )
            * dx
        )
        self.L_form = ufl.inner(self.b, self.u_test) * dx + ufl.inner(
            self.t, self.u_test
        ) * ds

        self._a_compiled = None
        self._L_compiled = None
        self._ksp = None

        self.assemble()

    @staticmethod
    def _dirichlet_boundary(x):
        return np.isclose(x[0], 0.0, atol=1e-10)

    @staticmethod
    def _traction_boundary(x):
        return np.isclose(x[0], 1.0, atol=1e-10)

    @staticmethod
    def boundaryU(x, on_boundary):
        return on_boundary and x[0] < 1e-10

    @staticmethod
    def boundaryQ(x, on_boundary):
        return on_boundary and abs(x[0] - 1.0) < 1e-10

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

    def set_traction(self, tx, ty):
        self._traction_x.value = tx
        self._traction_y.value = ty
        self.assemble(assemble_lhs=False, assemble_rhs=True)

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
        m = self.m_transformed.copy()
        return m
