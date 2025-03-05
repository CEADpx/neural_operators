import sys
import numpy as np
import dolfin as dl

src_path = "../"
sys.path.append(src_path + 'plotting/')
from field_plot import *
from plot_mix_collection import *

import matplotlib.pyplot as plt
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def build_vector_vertex_maps(V, debug = False):

    nodes = V.mesh().coordinates()
    num_nodes = nodes.shape[0]

    n_comps = V.dim()//num_nodes

    map_vec_to_vertex = np.zeros(V.dim(), dtype=int)
    map_vertex_to_vec = np.zeros(V.dim(), dtype=int)

    u_fn = dl.Function(V)

    for i in range(num_nodes):

        for j in range(n_comps):
            ii = num_nodes*j + i
            # at ith node, set displacement at ith dof and find the its location in the vector data
            u_fn.vector().zero()
            u_fn.vector()[ii] = 1.
            u_vv = u_fn.compute_vertex_values()

            # quick_field_plot(u_fn.compute_vertex_values(), V.mesh().coordinates(), title = 'u_fn')


            # find index where u_vec is nonzero
            idx = np.where(u_vv != 0)[0][0]

            if debug:
                u_vec = u_fn.vector().get_local()
                xi = nodes[idx%num_nodes]
                u_x = u_fn(xi)

                print('ii = {}, idx = {}, x = {}, u(x) = {}, u_ve[{}] = {}, u_vv[{}] = {}'.format(ii, \
                                idx, xi, u_x, \
                                    ii, u_vec[ii], idx, u_vv[idx]))
            
            map_vertex_to_vec[ii] = idx
            map_vec_to_vertex[idx] = ii

    return map_vec_to_vertex, map_vertex_to_vec

def test_vector_vertex_maps():
    # create mesh
    mesh = dl.UnitSquareMesh(10, 10)
    V = dl.VectorFunctionSpace(mesh, "Lagrange", 1)
    nodes = mesh.coordinates()
    num_nodes = nodes.shape[0]

    map_vec_to_vertex, map_vertex_to_vec = build_vector_vertex_maps(V)

    u_fn = dl.Function(V)
    u_fn.interpolate(dl.Expression(("0.1*x[0]*x[1]", "-0.1*x[0]*x[1]"), degree=2))

    u2_fn = dl.Function(V)

    u_vec = u_fn.vector().get_local()
    u_vv = u_vec[map_vec_to_vertex]

    u_cvv = u_fn.compute_vertex_values()

    u_vec_vv = u_vec[map_vec_to_vertex]
    u_fn.vector().set_local(u_vec_vv[map_vertex_to_vec])

    u_vv_vec = u_vv[map_vertex_to_vec]
    u2_fn.vector().set_local(u_vv_vec)

    u_cvv_vec = u_cvv[map_vertex_to_vec]
    u3_fn = dl.Function(V)
    u3_fn.vector().set_local(u_cvv_vec)

    # create data for plot
    ncols = 4
    data = get_default_plot_mix_collection_data()
    data['figsize'] = (20, 20)
    data['fs'] = 20
    data['rows'] = 2
    data['cols'] = ncols
    data['nodes'] = mesh.coordinates()
    data['sup_title'] = 'u_vv, u_vec, and their conversions'

    uvec = [[u_vv, u_vec_vv, u_cvv, u3_fn.vector().get_local()[map_vec_to_vertex]], \
            [u_vec, u_vv_vec, u2_fn.vector().get_local(), u3_fn.vector().get_local()]]

    title_vec = np.array([['u_vv', 'u_vec_vv', 'u_cvv', 'u_cvv_fn'], \
                        ['u_vec', 'u_vv_vec', 'u_vv_vec_fn', 'u_cvv_vec_fn']])


    data['u']= uvec
    data['title'] = title_vec
    data['cmap'] = np.array([['jet' for _ in range(ncols)], ['viridis' for _ in range(ncols)]])
    data['axis_off'] = [[True for _ in range(ncols)], [True for _ in range(ncols)]]
    data['is_vec'] = [[True for _ in range(ncols)], [True for _ in range(ncols)]]
    data['add_disp'] = [[True for _ in range(ncols)], [True for _ in range(ncols)]]
    data['plot_type'] = [['field' for _ in range(ncols)], ['field' for _ in range(ncols)]]

    plot_mix_collection(data)

    

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
