import numpy as np
import dolfin as dl

def function_to_vector(u, u_vec = None):
    if u_vec is not None:
        u_vec = u.vector().get_local()
        return u_vec
    else:
        return u.vector().get_local()

def vector_to_function(u_vec, u):
    if u is not None:
        u.vector()[:] = u_vec
        return u
    else:
        return dl.Function(u.function_space(), u_vec)

def function_to_vertex(u, u_vv = None, V = None):
    # compute_vertex_values() does not work as intended when dealing with vector functions. Best to use vertex_to_dof_map
    # if u_vv is not None:
    #     u_vv = u.compute_vertex_values()
    #     return u_vv
    # else:
    #     return u.compute_vertex_values()

    if V is None:
        V = u.function_space()

    V_v2d = dl.vertex_to_dof_map(V)
    if u_vv is not None:
        u_vv = u.vector().get_local()[V_v2d]
        return u_vv
    else:
        return u.vector().get_local()[V_v2d]

def vertex_to_function(u_vv, u = None, V = None):
    if V is None and u is None:
        raise ValueError('Need to provide either V or u')
    
    if u is not None:
        V = u.function_space()
        V_d2v = dl.dof_to_vertex_map(V)
        u.vector().set_local(u_vv[V_d2v])
        return u
    else:
        u = dl.Function(V)
        V_d2v = dl.dof_to_vertex_map(V)
        u.vector().set_local(u_vv[V_d2v])
        return u
    
def vector_to_vertex(u_vec, u_vv = None, V = None):
    if V is None:
        raise ValueError('Need to provide V')
    
    V_v2d = dl.vertex_to_dof_map(V)

    if u_vv is not None:        
        u_vv = u_vec[V_v2d]
        return u_vec
    else:
        return u_vec[V_v2d]
    
def vertex_to_vector(u_vv, u_vec = None, V = None):
    if V is None:
        raise ValueError('Need to provide V')
    
    V_d2v = dl.dof_to_vertex_map(V)

    if u_vec is not None:        
        u_vec[V_d2v] = u_vv
        return u_vec
    else:
        return u_vv[V_d2v]
    

def test_fenics_conversions():
    # local utility methods
    from plotUtilities import get_default_plot_mix_collection_data, plot_mix_collection

    # create mesh
    mesh = dl.UnitSquareMesh(50, 50)
    V = dl.FunctionSpace(mesh, "Lagrange", 1)

    # test functions
    u = dl.Function(V)
    u_assign = np.random.rand(V.dim())
    u.vector().set_local(u_assign)
    u2 = dl.Function(V)

    u_vv = function_to_vertex(u, None, V=V)
    u_vec = function_to_vector(u)

    u_vv_fn = vertex_to_function(u_vv, V=V)
    u_vec_fn = vector_to_function(u_vec, u2)

    u_vv_to_vec = vertex_to_vector(u_vv, V=V)
    u_vec_to_vv = vector_to_vertex(u_vec, V=V)

    # 
    # diff_u_vv_u_vv_fn = u_vv - u_vv_fn.compute_vertex_values() # works for scalar functions but not for vector functions
    diff_u_vv_u_vv_fn = u_vv - function_to_vertex(u_vv_fn, None, V)
    diff_u_vv_u_vec_to_vv = u_vv - u_vec_to_vv

    print('diff_u_vv_u_vv_fn:', np.linalg.norm(diff_u_vv_u_vv_fn))
    print('diff_u_vv_u_vec_to_vv:', np.linalg.norm(diff_u_vv_u_vec_to_vv))

    diff_u_vec_u_vec_fn = u_vec - u_vec_fn.vector().get_local()
    diff_u_vec_u_vv_to_vec = u_vec - u_vv_to_vec

    print('diff_u_vec_u_vec_fn:', np.linalg.norm(diff_u_vec_u_vec_fn))
    print('diff_u_vec_u_vv_to_vec:', np.linalg.norm(diff_u_vec_u_vv_to_vec))

    # create data for plot
    data = get_default_plot_mix_collection_data()
    data['figsize'] = (30, 10)
    data['fs'] = 30
    data['rows'] = 2
    data['cols'] = 5
    data['nodes'] = mesh.coordinates()
    data['sup_title'] = 'u_vv, u_vec, and their conversions'

    uvec = [[u_vv, function_to_vertex(u_vv_fn, None, V), u_vec_to_vv, diff_u_vv_u_vv_fn, diff_u_vv_u_vec_to_vv], \
            [u_vec, function_to_vector(u_vec_fn), u_vv_to_vec, diff_u_vec_u_vec_fn, diff_u_vec_u_vv_to_vec]]

    title_vec = np.array([['u_vv', 'u_vv_fn', 'u_vec_to_vv', \
                        'diff(u_vv, u_vv_fn)', 'diff(u_vv, u_vec_to_vv)'], \
                        ['u_vec', 'u_vec_fn', 'u_vv_to_vec', \
                        'diff(u_vec, u_vec_fn)', 'diff(u_vec, u_vv_to_vec)']])

    data['u']= uvec
    data['title'] = title_vec
    data['cmap'] = np.array([['jet' for _ in range(5)], ['viridis' for _ in range(5)]])
    data['axis_off'] = [[True for _ in range(5)], [True for _ in range(5)]]
    data['is_vec'] = [[False for _ in range(5)], [False for _ in range(5)]]
    data['add_disp'] = [[False for _ in range(5)], [False for _ in range(5)]]



    plot_mix_collection(data)
