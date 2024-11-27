### 
# This file contains functions to plot the field on 2D domain using matplotlib
# The functions are:
# 1. get_default_plot_mix_collection_data: Get the default data for plot_mix_collection
# 2. plot_mix_collection: Needs data in the required format. It can plot array of data either on nodes or grid. Use 'get_default_plot_mix_collection_data' to get the default data.
# 3. plot_collection: Plots the nodal field on 2D domain.
# 4. plot_collection_grid: Plots the field on 2D grid (for FNO method)


import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# local utility methods
from field_plot import field_plot, field_plot_grid
from point_plot import point_plot


def get_default_plot_mix_collection_data(rows = 1, cols = 1, \
    nodes = None, grid_x = None, grid_y = None, nodes_point_plot = None, \
    figsize = (20, 20), fs = 20, sup_title = None, y_sup_title = 1.025, \
    savefilename = None, fig_pad = 1.08, cax_size = '8%', cax_pad = 0.03, \
    u = None, cmap = None, title = None, \
    plot_type = None, \
    axis_off = None, is_vec = None, add_disp = None):

    if u is None:
        u = [[None for _ in range(cols)] for _ in range(rows)]
    if cmap is None:
        cmap = [['jet' for _ in range(cols)] for _ in range(rows)]
    if title is None:
        title = [[None for _ in range(cols)] for _ in range(rows)]
    if axis_off is None:
        axis_off = [[True for _ in range(cols)] for _ in range(rows)]
    if is_vec is None:
        is_vec = [[False for _ in range(cols)] for _ in range(rows)]
    if add_disp is None:
        add_disp = [[False for _ in range(cols)] for _ in range(rows)]
    if plot_type is None:
        plot_type = [[None for _ in range(cols)] for _ in range(rows)]

    return {
        'rows': rows,
        'cols': cols,
        'nodes': nodes,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'nodes_point_plot': nodes_point_plot,
        'figsize': figsize,
        'fs': fs,
        'sup_title': sup_title,
        'y_sup_title': y_sup_title,
        'savefilename': savefilename,
        'fig_pad': fig_pad,
        'cax_size': cax_size,
        'cax_pad': cax_pad,
        'u': u,
        'cmap': cmap,
        'title': title,
        'axis_off': axis_off,
        'is_vec': is_vec,
        'add_disp': add_disp,
        'plot_type': plot_type
    }


def plot_mix_collection(data):
    
    rows = data['rows']
    cols = data['cols']
    
    figsize = data['figsize']
    fs = data['fs']
    sup_title = data['sup_title']
    y_sup_title = data['y_sup_title']
    fs_sup_title = data['fs_sup_title'] if 'fs_sup_title' in data else 1.25*fs
    savefilename = data['savefilename'] if 'savefilename' in data else None
    fig_pad = data['fig_pad'] if 'fig_pad' in data else 1.08
    cax_size = data['cax_size'] if 'cax_size' in data else '8%'
    cax_pad = data['cax_pad'] if 'cax_pad' in data else 0.03

    nodes = data['nodes']
    grid_x = data['grid_x'] if 'grid_x' in data else None
    grid_y = data['grid_y'] if 'grid_y' in data else None
    nodes_point_plot = data['nodes_point_plot'] if 'nodes_point_plot' in data else None
    
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = np.array([axs]) if rows == 1 else axs

    for i in range(rows):
        for j in range(cols):

            
            u = data['u'][i][j]
            cmap = data['cmap'][i][j] if 'cmap' in data else 'jet'
            ttl = data['title'][i][j] if 'title' in data else None
            axis_off = data['axis_off'][i][j] if 'axis_off' in data else True
            is_vec = data['is_vec'][i][j] if 'is_vec' in data else False
            add_disp = data['add_disp'][i][j] if 'add_disp' in data else False
            
            ptype = data['plot_type'][i][j] if 'plot_type' in data else 'field'
            if ptype not in ['grid', 'field', 'point']:
                raise ValueError("Invalid plot type")
            if ptype == 'grid' and (grid_x is None or grid_y is None):
                raise ValueError("Grid data is missing")
            if ptype == 'point' and nodes_point_plot is None:
                raise ValueError("Nodes for point plot is missing")
            if ptype == 'field' and nodes is None:
                raise ValueError("Nodes for field plot is missing")
            
            if is_vec == False:
                if ptype == 'grid':
                    cbar = field_plot_grid(axs[i,j], \
                        u, grid_x, grid_y, cmap = cmap)
                elif ptype == 'field':
                    cbar = field_plot(axs[i,j], \
                        u, nodes, cmap = cmap)
                elif ptype == 'point':
                    cbar = point_plot(axs[i,j], \
                        u, nodes_point_plot, cmap = cmap)
            else:
                if ptype == 'grid':
                    cbar = field_plot_grid(axs[i,j], \
                        u, grid_x, grid_y, cmap = cmap, \
                        is_displacement = True, \
                        add_displacement_to_nodes = add_disp)
                elif ptype == 'field':
                    cbar = field_plot(axs[i,j], \
                        u, nodes, cmap = cmap, \
                        is_displacement = True, \
                        add_displacement_to_nodes = add_disp)
                elif ptype == 'point':
                    cbar = point_plot(axs[i,j], \
                        u, nodes, cmap = cmap, \
                        is_displacement = True, \
                        add_displacement_to_nodes = add_disp)

            divider = make_axes_locatable(axs[i,j])
            cax = divider.append_axes('right', size=cax_size, pad=cax_pad)
            cax.tick_params(labelsize=fs)
            cbar = fig.colorbar(cbar, cax=cax, orientation='vertical')
            if axis_off:
                axs[i,j].axis('off')
            if ttl is not None:
                axs[i,j].set_title(ttl, fontsize=fs)

    fig.tight_layout(pad = fig_pad)
    if sup_title is not None:
        fig.suptitle(sup_title, fontsize=fs_sup_title, y = y_sup_title)
    if savefilename is not None:
        plt.savefig(savefilename,  bbox_inches='tight')
    plt.show()


def plot_collection(uvec, rows, cols, nodes, \
                    title_vec = None, sup_title = None, \
                    cmapvec = None, fs = 20, \
                    figsize = (20, 20), \
                    y_sup_title = 1.025, \
                    savefilename = None):

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = np.array([axs]) if rows == 1 else axs

    for i in range(rows):
        for j in range(cols):
            
            cbar = field_plot(axs[i,j], \
                    uvec[i][j], \
                    nodes, cmap = cmapvec[i][j] if cmapvec is not None else 'jet')
            divider = make_axes_locatable(axs[i,j])
            cax = divider.append_axes('right', size='8%', pad=0.03)
            cax.tick_params(labelsize=fs)
            cbar = fig.colorbar(cbar, cax=cax, orientation='vertical')
            axs[i,j].axis('off')
            if title_vec[i][j] is not None:
                axs[i,j].set_title(title_vec[i][j], fontsize=fs)

    fig.tight_layout()
    if sup_title is not None:
        fig.suptitle(sup_title, fontsize=1.25*fs, y = y_sup_title)
    if savefilename is not None:
        plt.savefig(savefilename,  bbox_inches='tight')
    plt.show()



def plot_collection_grid(uvec, rows, cols, grid_x, grid_y, \
                    title_vec = None, sup_title = None, \
                    cmapvec = None, fs = 20, \
                    figsize = (20, 20), \
                    y_sup_title = 1.025, \
                    savefilename = None):

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = np.array([axs]) if rows == 1 else axs

    for i in range(rows):
        for j in range(cols):
            
            cbar = field_plot_grid(axs[i,j], \
                    uvec[i][j], \
                    grid_x, grid_y, \
                    cmap = cmapvec[i][j] if cmapvec is not None else 'jet')
            divider = make_axes_locatable(axs[i,j])
            cax = divider.append_axes('right', size='8%', pad=0.03)
            cax.tick_params(labelsize=fs)
            cbar = fig.colorbar(cbar, cax=cax, orientation='vertical')
            axs[i,j].axis('off')
            if title_vec[i][j] is not None:
                axs[i,j].set_title(title_vec[i][j], fontsize=fs)

    fig.tight_layout()
    if sup_title is not None:
        fig.suptitle(sup_title, fontsize=1.25*fs, y = y_sup_title)
    if savefilename is not None:
        plt.savefig(savefilename,  bbox_inches='tight')
    plt.show()