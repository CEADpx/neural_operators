import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

src_path = "../"
sys.path.append(src_path + 'plotting')
from field_plot import field_plot
from point_plot import point_plot

def mcmc_plot_fields_base(w_mean, w_sample, w_sample_i, mcmc, savefilename = None):

    model, x_obs = mcmc.model, mcmc.x_obs

    m_mean = mcmc.model.transform_gaussian_pointwise(w_mean)
    u_mean = mcmc.model.solveFwd(u = None, m = m_mean, transform_m = False)
    
    
    m_sample = mcmc.model.transform_gaussian_pointwise(w_sample)
    u_sample = mcmc.model.solveFwd(u = None, m = m_sample, transform_m = False)

    u_mean_obs = mcmc.state_to_obs(u_mean)
    u_sample_obs = mcmc.state_to_obs(u_sample)

    w_true, m_true, u_true = mcmc.data['w_true'], mcmc.data['m_true'], mcmc.data['u_true'] 
    u_obs = mcmc.data['u_obs']

    rows, cols = 3, 4
    fig, axs = plt.subplots(rows, cols, figsize=(20, 12))
    axs = np.array([axs]) if rows == 1 else axs

    uvec = [[w_true, m_true, u_true, u_obs], \
            [w_sample, m_sample, u_sample, u_sample_obs], \
            [w_mean, m_mean, u_mean, u_mean_obs]]
        
    title_vec = [ [ r'$w_{true} \sim N(0, C)$', \
                    r'$m_{true} = a\, \exp(w_{true}) + b$', \
                    r'$u_{true} = F(m_{true})$', \
                    r'$u_{obs}$' \
                    ], \
                    [ r'$w_{sample} \sim N(0, C)$', \
                    r'$m_{sample} = a\, \exp(w_{sample}) + b$', \
                    r'$u_{sample} = F(m_{sample})$', \
                    r'$u_{sample, obs}$' \
                    ], \
                    [ r'$w_{mean} \sim N(0, C)$', \
                    r'$m_{mean} = a\, \exp(w_{mean}) + b$', \
                    r'$u_{mean} = F(m_{mean})$', \
                    r'$u_{mean, obs}$' \
                ]]
        
    cmap_vec = [['magma', 'jet', 'viridis', 'copper'] \
                    for i in range(rows)]
    
    sup_title = r'Ground truth, $i^{th}$ sample' \
            + r', and posterior mean $(w, m, u(m), u_{obs})$' \
            + r', i = {}'.format(w_sample_i)

    fs = 25
    y_sup_title = 1.075

    nodes = model.m_nodes


    for i in range(rows):
        for j in range(cols):

            # add grid points
            if j == cols - 1:
                axs[i,j].set_xlim([-0.1, 1.1])
                axs[i,j].set_ylim([-0.1, 1.1])
            
            if j < cols - 1:
                cbar = field_plot(axs[i,j], \
                    uvec[i][j], \
                    nodes, cmap = cmap_vec[i][j])
            else:
                uob = uvec[i][j]
                cbar = point_plot(axs[i,j], uob, x_obs, cmap = cmap_vec[i][j])
            divider = make_axes_locatable(axs[i,j])
            cax = divider.append_axes('right', size='8%', pad=0.03)
            cax.tick_params(labelsize=fs)
            cbar = fig.colorbar(cbar, cax=cax, orientation='vertical')
            if j < cols - 1:
                axs[i,j].axis('off')
            if title_vec is not None:
                tt = title_vec[i][j]
                if i > 0:
                    u1, u2 = uvec[0][j], uvec[i][j]
                    err = np.linalg.norm(u1 - u2)/np.linalg.norm(u1)
                    tt += '\n' + r'err (l2 rel) = {:.2f}%'.format(err)
                axs[i,j].set_title(tt, fontsize=fs, pad=10)

    fig.tight_layout()
    if sup_title is not None:
        fig.suptitle(sup_title, fontsize=1.25*fs, y = y_sup_title)
    if savefilename is not None:
        plt.savefig(savefilename,  bbox_inches='tight')
    plt.show()

def mcmc_plot_fields(mcmc, savefilename = None):

    w_mean = mcmc.tracer.accepted_samples_mean_m
    w_sample_i = len(mcmc.tracer.accepted_samples_m) - 1
    w_sample = mcmc.tracer.accepted_samples_m[w_sample_i]

    mcmc_plot_fields_base(w_mean, w_sample, w_sample_i, mcmc, savefilename)
    