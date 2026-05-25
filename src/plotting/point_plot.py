import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

from field_plot import vector_values_at_nodes


def point_plot(ax, nodal_values, nodes, \
                cmap = None, \
                plot_absolute = False, \
                add_displacement_to_nodes = False, \
                is_displacement = False, \
                vector_layout = 'mixed'):

    num_nodes = nodes.shape[0]
    values_at_nodes, dof_per_node = vector_values_at_nodes(
        nodal_values, num_nodes, vector_layout=vector_layout
    )

    # Compute magnitude of the field
    plot_C = None
    if dof_per_node == 1:
        plot_C = np.sqrt(nodal_values[:]**2) if plot_absolute else nodal_values[:]
    else:
        plot_C = np.sqrt(np.sum(values_at_nodes**2, axis=1))

    # do we warp the configuration of domain (i.e., displace the nodal coordinates)?
    nodes_def = nodes.copy()
    if is_displacement:
        if dof_per_node != 2:
            raise ValueError("Expected a vector function")

        if add_displacement_to_nodes:
            nodes_def[:, 0] = nodes[:, 0] + values_at_nodes[:, 0]
            nodes_def[:, 1] = nodes[:, 1] + values_at_nodes[:, 1]
    
    cbar = ax.scatter(nodes_def[:,0], nodes_def[:,1], c = plot_C, cmap = cmap)

    return cbar

def quick_point_plot(nodal_values, nodes, \
                    title = None, \
                    cmap = None, \
                        add_displacement_to_nodes = False, \
                        is_displacement = False, \
                        vector_layout = 'mixed', \
                    fs = 20, \
                    figsize = (8, 8),
                    axis_off = False, \
                    ax_lim = None):
    

    fig, ax = plt.subplots(figsize=figsize)
    cmap = 'jet' if cmap is None else cmap

    if is_displacement:
        cbar = point_plot(ax, nodal_values, nodes, cmap = cmap, 
                          is_displacement = is_displacement,
                          add_displacement_to_nodes = add_displacement_to_nodes,
                          vector_layout = vector_layout)
    else:
        cbar = point_plot(ax, nodal_values, nodes, cmap = cmap,
                          vector_layout = vector_layout)
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
