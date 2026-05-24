import numpy as np
from dolfinx import fem, geometry


def _function_space(u_or_V):
    if hasattr(u_or_V, "function_space"):
        return u_or_V.function_space
    return u_or_V


def _num_array_dofs(V):
    return len(fem.Function(V).x.array)


def eval_at_vertices(u):
    """Evaluate a Lagrange function at mesh geometry vertices."""
    V = _function_space(u)
    domain = V.mesh
    points = domain.geometry.x
    block_size = V.dofmap.index_map_bs

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(
        domain, cell_candidates, points
    )

    if block_size == 1:
        values = np.zeros(points.shape[0], dtype=u.x.array.dtype)
        for i in range(points.shape[0]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                values[i] = float(
                    u.eval(
                        points[i],
                        np.array([links[0]], dtype=np.int32),
                    ).reshape(-1)[0]
                )
        return values

    values = np.zeros((points.shape[0], block_size), dtype=u.x.array.dtype)
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            values[i] = np.asarray(
                u.eval(points[i], np.array([links[0]], dtype=np.int32))
            ).reshape(-1)[:block_size]
    return values.reshape(-1)


def build_vector_vertex_maps(V, debug=False):
    """
    Build maps between native FEniCSx coefficient ordering and vertex ordering.

    Returns
    -------
    map_vec_to_vertex, map_vertex_to_vec :
        Arrays such that ``u_vv = u_vec[map_vec_to_vertex]`` and
        ``u_vec = u_vv[map_vertex_to_vec]``.
    """
    V = _function_space(V)
    u = fem.Function(V)
    num_dofs = len(u.x.array)

    map_vec_to_vertex = np.arange(num_dofs, dtype=int)
    map_vertex_to_vec = np.arange(num_dofs, dtype=int)

    # Fast path: for Lagrange spaces on matching meshes this is usually identity.
    u.x.array[:] = np.random.default_rng(0).random(num_dofs)
    vertex_values = eval_at_vertices(u)
    if len(vertex_values) == num_dofs and np.allclose(u.x.array, vertex_values):
        return map_vec_to_vertex, map_vertex_to_vec

    map_vec_to_vertex = np.zeros(num_dofs, dtype=int)
    map_vertex_to_vec = np.zeros(num_dofs, dtype=int)

    for ii in range(num_dofs):
        u.x.array[:] = 0.0
        u.x.array[ii] = 1.0
        u_vv = eval_at_vertices(u)
        nz = np.where(np.abs(u_vv) > 0.5)[0]
        if len(nz) == 0:
            raise RuntimeError(f"Could not locate vertex value for dof {ii}.")
        idx = int(nz[0])

        if debug:
            print(f"ii={ii}, vertex index={idx}, vertex value={u_vv[idx]}")

        map_vertex_to_vec[ii] = idx
        map_vec_to_vertex[idx] = ii

    return map_vec_to_vertex, map_vertex_to_vec


def function_to_vector(u, u_vec=None):
    values = u.x.array.copy()
    if u_vec is None:
        return values
    u_vec[:] = values
    return u_vec


def vector_to_function(u_vec, u):
    u.x.array[:] = u_vec
    return u


def function_to_vertex(u, u_vv=None, V=None, map_vec_to_vertex=None):
    if map_vec_to_vertex is None:
        map_vec_to_vertex, _ = build_vector_vertex_maps(
            V if V is not None else u
        )
    values = u.x.array[map_vec_to_vertex].copy()
    if u_vv is None:
        return values
    u_vv[:] = values
    return u_vv


def vertex_to_function(u_vv, u=None, V=None, map_vertex_to_vec=None):
    if u is None and V is None:
        raise ValueError("Need to provide either V or u.")

    if map_vertex_to_vec is None:
        _, map_vertex_to_vec = build_vector_vertex_maps(V if V is not None else u)

    if u is None:
        u = fem.Function(V)

    u.x.array[:] = u_vv[map_vertex_to_vec]
    return u


def vector_to_vertex(u_vec, u_vv=None, V=None, map_vec_to_vertex=None):
    if V is None:
        raise ValueError("Need to provide V.")

    if map_vec_to_vertex is None:
        map_vec_to_vertex, _ = build_vector_vertex_maps(V)

    values = u_vec[map_vec_to_vertex]
    if u_vv is None:
        return values
    u_vv[:] = values
    return u_vv


def vertex_to_vector(u_vv, u_vec=None, V=None, map_vertex_to_vec=None):
    if V is None:
        raise ValueError("Need to provide V.")

    if map_vertex_to_vec is None:
        _, map_vertex_to_vec = build_vector_vertex_maps(V)

    values = u_vv[map_vertex_to_vec]
    if u_vec is None:
        return values
    u_vec[:] = values
    return u_vec
