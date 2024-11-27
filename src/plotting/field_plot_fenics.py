###
# This file contains the utility method to plot the FEM solution on 2D domain.
# The function is:
# 1. field_plot_fenics: Plots the FEM solution on 2D domain. It needs fenics library.

import numpy as np

import dolfin as dl

import matplotlib.pyplot as plt
import matplotlib.tri as tri
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# local utility methods
from fenicsUtilities import function_to_vertex

# Plots the FEM solution on 2D domain. It needs fenics library.
def field_plot_fenics(ax, f, Vh, \
                      plot_absolute = False, \
                      add_displacement_to_nodes = False, \
                      is_displacement = False, \
                      is_fn = False, dbg_log = False, **kwargs):
    
    if is_fn:
        f_fn = f
    else:
        f_fn = dl.Function(Vh)
        f_fn.vector().zero()
        if isinstance(f, np.ndarray):
            f_fn.vector().set_local(f)
        else:
            f_fn.vector().axpy(1.0, f)
    
    mesh = Vh.mesh()
    gdim = mesh.geometry().dim()

    if gdim != 2:
        raise ValueError("Only 2D plots are supported")

    w0 = function_to_vertex(f_fn, None, V=Vh)
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
    plot_C = None
    if len(U.shape) == 1:
        plot_C = np.sqrt(U[:]**2) if plot_absolute else U[:]
    else:
        for i in range(n2):
            if i == 0:
                plot_C = U[:, i]**2
            else:
                plot_C += U[:, i]**2

        plot_C = np.sqrt(plot_C)

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
    cbar = ax.tripcolor(triang, plot_C, shading=shading, **kwargs)

    return cbar
