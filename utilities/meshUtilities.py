import numpy as np

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