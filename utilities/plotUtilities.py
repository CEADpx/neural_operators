import numpy as np
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["KMP_WARNINGS"] = "FALSE" 
import pickle

import hippylib as hp

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


# Taken from https://github.com/live-clones/dolfin/\
#            blob/master/python/dolfin/common/plotting.py

def field_plot(ax, f, Vh, reference_config = True, \
                      is_displacement = False, show_cbar = False, \
                      is_fn = False, **kwargs):
    
    if is_fn:
        f_fn = f
    else:
        f_fn = hp.vector2Function(f, Vh)
    
    mesh = Vh.mesh()
    gdim = mesh.geometry().dim()
    tdim = mesh.topology().dim()
    

    w0 = f_fn.compute_vertex_values(mesh)
    nv = mesh.num_vertices()


    X = mesh.coordinates()
    X = [X[:, i] for i in range(gdim)]
    U = [w0[i * nv: (i + 1) * nv] for i in range(gdim)]
    
    # Compute magnitude
    C = U[0]**2
    if is_displacement:
        for i in range(1, gdim):
            C += U[i]**2
    C = np.sqrt(C)

    Xdef = X if reference_config else [X[i] + U[i] for i in range(gdim)]
    
    cbar = None
    if gdim == 2 and tdim == 2:
        # FIXME: Not tested
        triang = tri.Triangulation(Xdef[0], Xdef[1], mesh.cells())
        shading = kwargs.pop("shading", "gouraud") # or 'shading', 'flat'
        cbar = ax.tripcolor(triang, C, shading=shading, **kwargs)

    return cbar