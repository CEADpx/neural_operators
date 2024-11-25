import numpy as np
import pickle

import dolfin as dl

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# local utility methods
from fenicsUtilities import function_to_vertex

def get_default_plot_mix_collection_data(rows = 1, cols = 1, \
    nodes = None, grid_x = None, grid_y = None, use_grid = False, \
    figsize = (20, 20), fs = 20, sup_title = None, y_sup_title = 1.025, \
    savefilename = None, fig_pad = 1.08, cax_size = '8%', cax_pad = 0.03, \
    u = None, cmap = None, title = None, \
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

    return {
        'rows': rows,
        'cols': cols,
        'nodes': nodes,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'use_grid': use_grid,
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
        'add_disp': add_disp
    }

def load_cmap(fn, cmap_name = 'my_colormap'):
    # fn is '.pkl' file
    cdict = pickle.load(open(fn,'rb'))
    mycmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return plt.get_cmap(mycmap)

## to load the colormap, use the following command
## mycmap_fn = util_path + '/erdc_cyan2orange.pkl'
## cyan2orange = load_cmap(mycmap_fn, cmap_name = 'cyan2orange')

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


def point_plot(ax, nodal_values, nodes, \
                    cmap = None, \
                    fs = 20):
    
    cbar = ax.scatter(nodes[:,0], nodes[:,1], c = nodal_values, cmap = cmap)
    
    return cbar

def quick_point_plot(nodal_values, nodes, \
                    title = None, \
                    cmap = None, \
                    fs = 20, \
                    figsize = (8, 8),
                    axis_off = False, \
                    ax_lim = None):
    

    fig, ax = plt.subplots(figsize=figsize)
    if cmap is not None:
        cbar = point_plot(ax, nodal_values, nodes, cmap = cmap)
    else:
        cbar = point_plot(ax, nodal_values, nodes)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='8%', pad=0.03)
    cax.tick_params(labelsize=20)
    cbar = fig.colorbar(cbar, cax=cax, orientation='vertical')
    if axis_off: 
        ax.axis('off')
    if ax_lim is not None:
        ax.set_xlim(ax_lim[:,0])
        ax.set_ylim(ax_lim[:,1])

    if title is not None:
        ax.set_title(title, fontsize=20)
    plt.show()


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
    use_grid = data['use_grid'] if 'use_grid' in data else False
    grid_x = data['grid_x'] if 'grid_x' in data else None
    grid_y = data['grid_y'] if 'grid_y' in data else None
    if use_grid:
        if grid_x is None or grid_y is None:
            raise ValueError("Grid is not provided")

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
            
            if is_vec == False:
                if use_grid:
                    cbar = field_plot_grid(axs[i,j], \
                        u, grid_x, grid_y, cmap = cmap)
                else:
                    cbar = field_plot(axs[i,j], \
                        u, nodes, cmap = cmap)
            else:
                if use_grid:
                    cbar = field_plot_grid(axs[i,j], \
                        u, grid_x, grid_y, cmap = cmap, \
                        is_displacement = True, \
                        add_displacement_to_nodes = add_disp)
                else:
                    cbar = field_plot(axs[i,j], \
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


def locate_index(data, val = 0.1):
    idx = (np.abs(data - val)).argmin()
    return idx, data[idx]

def plot_s_vec_values(s_vec, r_vec, tag_vec, l_style_vec, xy_text_vec, plot_annot_xy, plot_annot_xy_region, savefilename = None):

    clr_choices = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    mkr_choices = ['o' for i in range(5)] 
    # mkr_choices = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'P', '*', 'X']
    
    n_min_max = np.min([len(s) for s in s_vec])
    N = 400 if n_min_max >= 400 else n_min_max

    p_vec = [s_vec[i][:N] / s_vec[i][0] for i in range(len(s_vec))]
    p_at_r = [p_vec[i][r_vec[i]-1] for i in range(len(s_vec))]

    lf = 20
    lw = 2
    mkr = 10

    plt.style.use('seaborn-v0_8-whitegrid') # checking by running command 'plt.style.available'

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    # plot sigular values
    s_clrs = np.random.choice(clr_choices, len(s_vec), replace = False)
    s_mkrs = np.random.choice(mkr_choices, len(s_vec), replace = False)

    for i in range(len(s_vec)):
        ax.plot(np.arange(1, N+1), p_vec[i], \
                label = r'$\sigma^{'+str(tag_vec[i])+'}$', \
                lw = lw, \
                linestyle = l_style_vec[i], \
                color = s_clrs[i])

    # get small region for inset
    annot_flag = True
    if annot_flag:
        x1, x2, y1, y2 = plot_annot_xy_region  # subregion of the original image
        axins = ax.inset_axes(
            plot_annot_xy,
            xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
        
        for i in range(len(s_vec)):
            axins.plot(np.arange(1, N+1), p_vec[i], \
                label = r'$\sigma^{'+str(tag_vec[i])+'}$', \
                lw = lw, \
                linestyle = l_style_vec[i], \
                color = s_clrs[i])

        ax.indicate_inset_zoom(axins, edgecolor="black")

    # input annot
    val_vec = [None, 0.1, 0.01]
    val_color = ['grey', 'tab:brown', 'cadetblue']
    rot_vec = [0, 0, 0]

    for j in range(3):
        for i in range(len(s_vec)):
            val = val_vec[j] if val_vec[j] is not None else p_at_r[i]
            vclr = val_color[j]
            s_mkr = s_mkrs[i]
            if j == 0:
                index = r_vec[i] - 1
                index_val = p_at_r[i]
            else:
                index, index_val = locate_index(p_vec[i], val = val)
            
            print('j = {}, i = {}, index = {}, index_val = {}'.format(j, i, index+1, index_val))

            ax_annot = axins if annot_flag else ax

            annot_lbl = None
            if i == 0:
                # plot the marker legend on top
                if j == 0:
                    annot_lbl = r'$\sigma_r$ (r = red. dim.)'
                else:
                    index_val_str = '{:.2f}'.format(index_val)
                    annot_lbl = r'$\sigma = {{{}}}$'.format(index_val_str)

            if annot_lbl is None:
                axins.plot(index+1, index_val, linestyle = '', marker = s_mkr, \
                          markersize = mkr, \
                          markerfacecolor=  vclr, markeredgecolor = vclr)
                
                ax.plot(index+1, index_val, linestyle = '', marker = s_mkr, \
                          markersize = mkr,  \
                          markerfacecolor=  vclr, markeredgecolor = vclr)
            else:
                axins.plot(index+1, index_val, linestyle = '', marker = s_mkr, \
                          markersize = mkr, \
                          markerfacecolor=  vclr, markeredgecolor = vclr, \
                          label = annot_lbl)
                
                ax.plot(index+1, index_val, linestyle = '', marker = s_mkr, \
                          markersize = mkr, \
                          markerfacecolor=  vclr, markeredgecolor = vclr, \
                          label = annot_lbl)
            
            # if annot_flag:
            #     axins.plot(index+1, index_val, marker = s_mkr, \
            #               markersize = mkr, \
            #               markerfacecolor=  vclr, markeredgecolor = vclr)
            
            val = '{:.3f}'.format(val)
            annot_text = r'$\sigma^{}_{{{}}}$'.format(tag_vec[i], index + 1)
            if j == 0:
                annot_text += ' = ' + val

            # xy is difficult and requires trial and error
            xy_text = xy_text_vec[j]

            ax_annot.annotate(annot_text, \
                        xy=(index+1, index_val),
                        xytext=xy_text[i], xycoords = 'data', \
                        textcoords='offset points', color = vclr, \
                        size = lf , rotation = rot_vec[j])
            
            
                # ax.plot(0.2 + j*0.2, 1.05, linestyle = '', marker = s_mkr, markersize = mkr, \
                #           markerfacecolor=  vclr, markeredgecolor = vclr, label = annot_text)


    ax.legend(bbox_to_anchor=(0.5, 1.25), fontsize = lf, fancybox = True, frameon = True, ncol=3, loc = 'upper center')#, facecolor="gray")
    ax.set_xlabel(r'Index', fontsize = lf)
    ax.set_ylabel(r'Normalized Singular Values ($\sigma_i = \frac{\lambda_i}{\lambda_1})$', fontsize = lf)

    ax.set_ylim([-0.1, 1.1])
    ax.set_xlim([-20, N+20])

    

    plt.tight_layout(pad=0.4)

    if savefilename is not None:
        plt.savefig(savefilename + '.png')

    plt.show()


