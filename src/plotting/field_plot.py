###
# This file contains functions to plot the field on 2D domain using matplotlib
# The functions are:
# 1. field_plot: Plots the FEM solution on 2D domain without using any external library. Just needs a FE nodal solution and nodes.
# 2. field_plot_grid: Plots the field on 2D grid (for FNO method)
# 3. quick_field_plot: A quick function to plot the field on 2D domain
# 4. quick_field_plot_grid: A quick function to plot the field on 2D grid

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def load_cmap(fn, cmap_name = 'my_colormap'):
    import pickle
    
    # fn is '.pkl' file
    cdict = pickle.load(open(fn,'rb'))
    mycmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return plt.get_cmap(mycmap)

## to load the colormap, use the following command
## mycmap_fn = util_path + '/erdc_cyan2orange.pkl'
## cyan2orange = load_cmap(mycmap_fn, cmap_name = 'cyan2orange')


# Plots the FEM solution on 2D domain without using any external library. Just needs a FE nodal solution and nodes. 
def field_plot(ax, fn_nodal_values, nodes, elements = None, dim = 2, \
                        plot_absolute = False, \
                        add_displacement_to_nodes = False, \
                        is_displacement = False, \
                        dbg_log = False, **kwargs):
    
    if dim != 2:
        raise ValueError("Only 2D plots are supported")
    
    if dbg_log:
        print('fn_nodal_values.shape = {}, nodes.shape = {}'.format(fn_nodal_values.shape, \
                                                                nodes.shape))
    
    num_nodes = nodes.shape[0]
    num_fn_values = fn_nodal_values.shape[0]

    dof_per_node = num_fn_values // num_nodes
    if dof_per_node == 0:
        raise ValueError("Number of dofs per node is zero")

    # Compute magnitude of the field
    plot_C = None
    if dof_per_node == 1:
        plot_C = np.sqrt(fn_nodal_values[:]**2) if plot_absolute else fn_nodal_values[:]
    else:
        for i in range(dof_per_node):
            if i == 0:
                plot_C = fn_nodal_values[i*num_nodes:(i+1)*num_nodes]**2
            else:
                plot_C += fn_nodal_values[i*num_nodes:(i+1)*num_nodes]**2

        plot_C = np.sqrt(plot_C)

    # do we warp the configuration of domain (i.e., displace the nodal coordinates)?
    nodes_def = nodes.copy()
    if is_displacement:
        if dof_per_node != 2:
            raise ValueError("Expected a vector function")

        if add_displacement_to_nodes:
            nodes_def[:, 0] = nodes[:, 0] + fn_nodal_values[0:num_nodes]
            nodes_def[:, 1] = nodes[:, 1] + fn_nodal_values[num_nodes:2*num_nodes]

    if dbg_log:
        print('nodes_def.shape = {}'.format(nodes_def.shape))
    
    triang = None
    if elements is not None:
        triang = tri.Triangulation(nodes_def[:, 0], nodes_def[:, 1], elements)
    else:
        triang = tri.Triangulation(nodes_def[:, 0], nodes_def[:, 1])

    shading = kwargs.pop("shading", "gouraud") # or 'shading', 'flat'

    cbar = ax.tripcolor(triang, plot_C, shading=shading, **kwargs)

    return cbar

# Plots the field on 2D grid (for FNO method)
def field_plot_grid(ax, fn_nodal_values, grid_x, grid_y, dim = 2, \
                        plot_absolute = False, \
                        add_displacement_to_nodes = False, \
                        is_displacement = False, \
                        dbg_log = False, **kwargs):
    if dim != 2:
        raise ValueError("Only 2D plots are supported")
    
    # grid_x and grid_y are of shape (nx, ny)
    # fn_nodal_values is of shape (nx, ny) if scalar and (nx, ny, 2) if vector
    nx, ny = grid_x.shape[0], grid_x.shape[1]
    n_comps = 1 if len(fn_nodal_values.shape) == 2 else fn_nodal_values.shape[2]
    if dbg_log:
        print('nx = {}, ny = {}, n_comps = {}'.format(nx, ny, n_comps))

    # we reduce the grid_x and grid_y to 1D arrays and then stack them together
    nodes = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    if dbg_log:
        print('nodes.shape = {}'.format(nodes.shape))

    # also reduce the fn_nodal_values to 1D array
    if n_comps == 1:
        fn_nodal_values = fn_nodal_values.flatten()
    else:
        fn_nodal_values = fn_nodal_values.reshape((nx*ny, n_comps))
    
    # Compute magnitude of the field
    plot_C = None
    if n_comps == 1:
        plot_C = np.sqrt(fn_nodal_values[:]**2) if plot_absolute else fn_nodal_values[:]
    else:
        for i in range(n_comps):
            if i == 0:
                plot_C = fn_nodal_values[:, i]**2
            else:
                plot_C += fn_nodal_values[:, i]**2

        plot_C = np.sqrt(plot_C)

    # manipulate the configuration of the plot
    nodes_def = nodes
    if is_displacement and add_displacement_to_nodes:
        if n_comps != 2:
            raise ValueError("Displacement should be a 2D array for dim = 2")

        nodes_def = nodes + fn_nodal_values

    if dbg_log:
        print('nodes_def.shape = {}'.format(nodes_def.shape))
    
    triang = tri.Triangulation(nodes_def[:, 0], nodes_def[:, 1])

    shading = kwargs.pop("shading", "gouraud") # or 'shading', 'flat'

    cbar = ax.tripcolor(triang, plot_C, shading=shading, **kwargs)

    return cbar


def quick_field_plot(fn_nodal_values, nodes, \
                        title = None, \
                        cmap = None, \
                        add_displacement_to_nodes = False, \
                        is_displacement = False, \
                        figsize = (6,6), \
                        fs = 20, \
                        savefilename = None, \
                        **kwargs):
    
    fig, ax = plt.subplots(figsize=figsize)
    cmap = 'jet' if cmap is None else cmap
    if is_displacement:
        cbar = field_plot(ax, fn_nodal_values, \
            nodes, cmap = cmap, \
                add_displacement_to_nodes = add_displacement_to_nodes, \
                    is_displacement = is_displacement)
    else:
        cbar = field_plot(ax, fn_nodal_values, nodes, cmap = cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='8%', pad=0.03)
    cax.tick_params(labelsize=fs)
    cbar = fig.colorbar(cbar, cax=cax, orientation='vertical')
    ax.axis('off')
    if title is not None:
        ax.set_title(title, fontsize=fs)
    if savefilename is not None:
        plt.savefig(savefilename,  bbox_inches='tight')
    plt.show()

def quick_field_plot_grid(fn_nodal_values, grid_x, grid_y, \
                        title = None, \
                        cmap = None, \
                        add_displacement_to_nodes = False, \
                        is_displacement = False, \
                        figsize = (6,6), \
                        fs = 20, \
                        savefilename = None, \
                        **kwargs):
    
    fig, ax = plt.subplots(figsize=figsize)
    cmap = 'jet' if cmap is None else cmap
    if is_displacement:
        cbar = field_plot_grid(ax, fn_nodal_values, grid_x, grid_y, \
                               cmap = cmap, \
                add_displacement_to_nodes = add_displacement_to_nodes, \
                    is_displacement = is_displacement)
    else:
        cbar = field_plot_grid(ax, fn_nodal_values, grid_x, grid_y, cmap = cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='8%', pad=0.03)
    cax.tick_params(labelsize=fs)
    cbar = fig.colorbar(cbar, cax=cax, orientation='vertical')
    ax.axis('off')
    if title is not None:
        ax.set_title(title, fontsize=fs)
    if savefilename is not None:
        plt.savefig(savefilename,  bbox_inches='tight')
    plt.show()