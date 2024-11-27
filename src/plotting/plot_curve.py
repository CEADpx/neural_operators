import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def plot_curve(data, x = None, xl = None, yl = None, \
               fs = 20, lw = 2, lclr = 'tab:blue', \
               l_lbl = None, \
               title = None, \
               savefile = None, \
               figsize = [10,10]):
    
    fig, ax = plt.subplots(figsize=figsize)

    if x is not None:
        if l_lbl is not None:
            ax.plot(x, data, lw = lw, color = lclr, label = l_lbl)
        else:
            ax.plot(x, data, lw = lw, color = lclr)
    else:
        ax.plot(data)
    if xl is not None:
        ax.set_xlabel(xl, fontsize = fs)
    if yl is not None:
        ax.set_ylabel(yl, fontsize = fs)

    if l_lbl is not None:
        ax.legend(fontsize = fs, fancybox = True, frameon = True)

    if title is not None:
        ax.set_title(title, fontsize = fs)
        
    plt.tight_layout(pad=0.4)
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()
