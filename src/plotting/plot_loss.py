###
# This file contains functions to plot the field on 2D domain using matplotlib
# The functions are:
# 1. field_plot: Plots the FEM solution on 2D domain without using any external library. Just needs a FE nodal solution and nodes.
# 2. field_plot_grid: Plots the field on 2D grid (for FNO method)
# 3. quick_field_plot: A quick function to plot the field on 2D domain
# 4. quick_field_plot_grid: A quick function to plot the field on 2D grid

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def plot_loss( train_loss, test_loss, \
              xl = None, yl = None, \
               fs = 20, lw = 2, \
               train_clr = 'tab:blue', \
               test_clr = 'tab:red', \
               title = None, \
               show_grid = True, \
               savefile = None, \
               figsize = [10,10]):
    
    fig, ax = plt.subplots(figsize=figsize)

    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    if xl is None:
        xl = r'Epochs'
    if yl is None:
        yl = r'Loss (l2 squared)'

    ax.plot(x, train_loss, lw = lw, color = train_clr, label = r'Training Loss')
    ax.plot(x, test_loss, lw = lw, color = test_clr, label = r'Testing Loss')

    ax.set_yscale('log')
    ax.set_xlabel(xl, fontsize = fs)
    ax.set_ylabel(yl, fontsize = fs)
    ax.legend(fontsize = fs, fancybox = True, frameon = True)

    if title is not None:
        ax.set_title(title, fontsize = fs)
        
    plt.tight_layout(pad=0.4)
    if savefile is not None:
        plt.savefig(savefile)
    
    if show_grid:
        plt.grid()
    plt.show()