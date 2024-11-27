import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

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


