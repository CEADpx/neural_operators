from pathlib import Path

import numpy as np

MESH_DIR = Path(__file__).resolve().parents[2] / "setup/fenicsx_test/meshes"


def legacy_unit_square_mesh_path(nx: int = 50, ny: int = 50) -> Path:
    """Path to XDMF mesh exported from legacy ``UnitSquareMesh(nx, ny)``."""
    return MESH_DIR / f"unit_square_{nx}x{ny}.xdmf"


def load_legacy_unit_square_mesh(comm, nx: int = 50, ny: int = 50):
    """
    Load the unit-square mesh exported from legacy FEniCS ``UnitSquareMesh``.

    Vertex numbering may differ from legacy DOF order; use
    ``legacy_dof_permutation`` when comparing index-wise with legacy outputs.
    """
    from dolfinx import io

    path = legacy_unit_square_mesh_path(nx, ny)
    if not path.exists():
        raise FileNotFoundError(
            f"Legacy mesh not found at {path}. "
            "Run `conda activate neuralop && "
            "python setup/fenicsx_test/export_legacy_mesh.py` first."
        )
    with io.XDMFFile(comm, path, "r") as xdmf:
        return xdmf.read_mesh()


def legacy_dof_permutation(legacy_coords, fenicsx_coords):
    """
    Map legacy vertex indices to FEniCSx DOF indices by matching coordinates.

    ``fenicsx_array[perm[i]] = legacy_array[i]`` for nodal data aligned with
    legacy ``UnitSquareMesh`` vertex numbering.
    """
    from scipy.spatial import cKDTree

    _, fenicsx_idx_for_legacy = cKDTree(fenicsx_coords[:, :2]).query(legacy_coords)
    return fenicsx_idx_for_legacy.astype(int)


def mesh_cells(domain):
    """Return (n_cells, n_vertices_per_cell) connectivity in geometry vertex indices."""
    topology = domain.topology
    tdim = topology.dim
    topology.create_connectivity(tdim, 0)
    c_to_v = topology.connectivity(tdim, 0)
    n_cells = topology.index_map(tdim).size_local
    return np.array([c_to_v.links(i) for i in range(n_cells)])


def write_mesh_xdmf(domain, path, comm):
    """Write a dolfinx mesh to XDMF (``path`` should end in ``.xdmf``)."""
    from dolfinx import io

    with io.XDMFFile(comm, path, "w") as xdmf:
        xdmf.write_mesh(domain)


def get_dirichlet_bc(bdry_fn, x):
    boundary_nodes = []

    for i in range(x.shape[0]):
        if bdry_fn(x[i,:]):
            boundary_nodes.append(i)
        
    return np.array(boundary_nodes)

def get_grid_dirichlet_bc(bdry_fn, x, y):
    boundary_nodes = []

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if bdry_fn(np.array([x[i,j], y[i,j]])):
                boundary_nodes.append((i, j))
        
    return np.array(boundary_nodes)

def test_dirichlet_bc_functions():

    # define example boundary function
    def boundary(x):
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
    # test
    nx, ny = 6, 11
    a, b = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing='ij')
    x = np.vstack((a.flatten(), b.flatten())).T
    bc_ids = get_dirichlet_bc(boundary, x)
    bc_vals = x[bc_ids,:]
    print('test mesh')
    for i in range(bc_vals.shape[0]):
        print('bc id: {}, bc val: {}'.format(bc_ids[i], bc_vals[i,:]))

    print('\ntest grid')
    bc_grid_ids = get_grid_dirichlet_bc(boundary, a, b)
    bc_grid_vals = [a[bc_grid_ids[:,0], bc_grid_ids[:,1]], b[bc_grid_ids[:,0], bc_grid_ids[:,1]]]
    bc_grid_vals = np.array(bc_grid_vals).T
    for i in range(bc_grid_ids.shape[0]):
        print('bc id: ({}, {}), bc val: ({}, {})'.format(bc_grid_ids[i,0], bc_grid_ids[i,1], bc_grid_vals[i,0], bc_grid_vals[i,1]))