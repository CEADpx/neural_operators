import numpy as np
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["KMP_WARNINGS"] = "FALSE" 
import pickle

import dolfin as dl

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import Subplot
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_cmap(fn, cmap_name = 'my_colormap'):
    # fn is '.pkl' file
    cdict = pickle.load(open(fn,'rb'))
    mycmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return plt.get_cmap(mycmap)

# mycmap_fn = util_path + '/erdc_cyan2orange.pkl'
# cyan2orange = load_cmap(mycmap_fn, cmap_name = 'cyan2orange')

def field_plot_fenics(ax, f, Vh, add_displacement_to_nodes = False, \
                      is_displacement = False, \
                      is_fn = False, dbg_log = False, **kwargs):
    
    if is_fn:
        f_fn = f
    else:
        f_fn = dl.Function(Vh)
        f_fn.vector().zero()
        f_fn.vector().axpy(1.0, f)
    
    mesh = Vh.mesh()
    gdim = mesh.geometry().dim()

    if gdim != 2:
        raise ValueError("Only 2D plots are supported")

    w0 = f_fn.compute_vertex_values(mesh)
    nv = mesh.num_vertices()

    U = [w0[i * nv: (i + 1) * nv] for i in range(gdim)]
    
    if gdim == 2:
        if len(U[gdim - 1]) == 0:
            U = np.array(U[0]).T
        else:
            U = np.array(U).T
    else:
        U = np.array(U).T

    n1, n2 = U.shape[0], 1 if len(U.shape) == 1 else U.shape[1]
    if dbg_log:
        print('n1, n2 = {}, {}'.format(n1, n2))

    nodes = mesh.coordinates()
    elements = mesh.cells()

    # Compute magnitude of the field
    C = None
    if len(U.shape) == 1:
        C = U[:]**2
    else:
        for i in range(n2):
            if i == 0:
                C = U[:, i]**2
            else:
                C += U[:, i]**2

    C = np.sqrt(C)

    # manipulate the configuration of the plot
    nodes_def = nodes
    if is_displacement:
        if n2 != 2:
            raise ValueError("Displacement should be a 2D array for dim = 2")

        if add_displacement_to_nodes:
            nodes_def = nodes + U

    if dbg_log:
        print('nodes_def.shape = {}'.format(nodes_def.shape))
    
    triang = tri.Triangulation(nodes_def[:, 0], nodes_def[:, 1], elements)
    shading = kwargs.pop("shading", "gouraud") # or 'shading', 'flat'
    cbar = ax.tripcolor(triang, C, shading=shading, **kwargs)

    return cbar

def field_plot(ax, fn_nodal_values, nodes, elements = None, dim = 2, \
                        add_displacement_to_nodes = False, \
                        is_displacement = False, \
                        dbg_log = False, **kwargs):
    
    if dim != 2:
        raise ValueError("Only 2D plots are supported")
    
    if dbg_log:
        print('fn_nodal_values.shape = {}, nodes.shape = {}'.format(fn_nodal_values.shape, \
                                                                nodes.shape))
    
    n1, n2 = len(fn_nodal_values), 1
    if fn_nodal_values.ndim == 2:
        n2 = fn_nodal_values.shape[1]
    elif fn_nodal_values.ndim > 2: 
        raise ValueError("fn_nodal_values should be a 1D or 2D array")

    if n1 != nodes.shape[0]:
        raise ValueError("Number of nodes in the mesh and the number of dofs do not match")
    
    # Compute magnitude of the field
    C = None
    if fn_nodal_values.ndim == 1:
        C = fn_nodal_values[:]**2
    else:
        for i in range(n2):
            if i == 0:
                C = fn_nodal_values[:, i]**2
            else:
                C += fn_nodal_values[:, i]**2

    C = np.sqrt(C)

    # manipulate the configuration of the plot
    nodes_def = nodes
    if is_displacement:
        if n2 != 2:
            raise ValueError("Displacement should be a 2D array for dim = 2")

        if add_displacement_to_nodes:
            nodes_def = nodes + fn_nodal_values

    if dbg_log:
        print('nodes_def.shape = {}'.format(nodes_def.shape))
    
    triang = None
    if elements is not None:
        triang = tri.Triangulation(nodes_def[:, 0], nodes_def[:, 1], elements)
    else:
        triang = tri.Triangulation(nodes_def[:, 0], nodes_def[:, 1])

    shading = kwargs.pop("shading", "gouraud") # or 'shading', 'flat'

    cbar = ax.tripcolor(triang, C, shading=shading, **kwargs)

    return cbar