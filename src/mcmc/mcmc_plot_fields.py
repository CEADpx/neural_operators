import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

src_path = "../"
sys.path.append(src_path + 'plotting')
from field_plot import field_plot
from point_plot import point_plot

def mcmc_plot_fields_base(w_mean, w_sample, w_sample_i, mcmc, savefilename = None, params = None):

    fs = 25 if params is None else params['fs']
    figsize = (20, 12) if params is None else params['figsize']
    y_sup_title = 1.075 if params is None else params['y_sup_title']
    ttl_pad = 10 if params is None else params['ttl_pad']

    model, x_obs = mcmc.model, mcmc.x_obs

    m_mean = mcmc.model.transform_gaussian_pointwise(w_mean)
    u_mean = mcmc.model.solveFwd(u = None, m = m_mean, transform_m = False)
    
    
    m_sample = mcmc.model.transform_gaussian_pointwise(w_sample)
    u_sample = mcmc.model.solveFwd(u = None, m = m_sample, transform_m = False)

    u_mean_obs = mcmc.state_to_obs(u_mean)
    u_sample_obs = mcmc.state_to_obs(u_sample)

    w_true, m_true, u_true = mcmc.data['w_true'], mcmc.data['m_true'], mcmc.data['u_true'] 
    u_obs = mcmc.data['u_obs']

    # is u vector or a scalar?
    nodes = model.m_nodes
    u_is_vec = False
    if len(u_mean) == len(nodes):
        u_is_vec = False
    else:
        u_is_vec = True

    u_vec_plot = params['u_vec_plot'] if params is not None and 'u_vec_plot' in params else False
    if u_is_vec == False:
        u_vec_plot = False
    
    rows, cols = 3, 4
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = np.array([axs]) if rows == 1 else axs

    uvec = [[w_true, m_true, u_true, u_obs], \
            [w_sample, m_sample, u_sample, u_sample_obs], \
            [w_mean, m_mean, u_mean, u_mean_obs]]
        
    title_vec = [ [ r'$w_{true} \sim N(0, C)$', \
                    r'$m_{true} = \alpha_m\, \exp(w_{true}) + \beta_m$', \
                    r'$u_{true} = F(m_{true})$', \
                    r'$u_{obs}$' \
                    ], \
                    [ r'$w_{sample} \sim N(0, C)$', \
                    r'$m_{sample} = \alpha_m\, \exp(w_{sample}) + \beta_m$', \
                    r'$u_{sample} = F(m_{sample})$', \
                    r'$u_{sample, obs}$' \
                    ], \
                    [ r'$w_{mean} \sim N(0, C)$', \
                    r'$m_{mean} = \alpha_m\, \exp(w_{mean}) + \beta_m$', \
                    r'$u_{mean} = F(m_{mean})$', \
                    r'$u_{mean, obs}$' \
                ]]

    # get cmap from params if it exists
    cmap_w = 'magma' if params is None or 'cmap_w' not in params else params['cmap_w']
    cmap_m = 'jet' if params is None or 'cmap_m' not in params else params['cmap_m']
    cmap_u = 'jet' if params is None or 'cmap_u' not in params else params['cmap_u']
    cmap_uobs = 'copper' if params is None or 'cmap_uobs' not in params else params['cmap_uobs']

    cmap_vec = [[cmap_w, cmap_m, cmap_u, cmap_uobs] \
                    for i in range(rows)]
    
    sup_title = r'Ground truth, $i^{th}$ sample' \
            + r', and posterior mean $(w, m, u(m), u_{obs})$' \
            + r', i = {}'.format(w_sample_i)
    
    if params is not None and 'sup_title' in params:
        sup_title = params['sup_title']

    # fs = 25
    # y_sup_title = 1.075

    


    for i in range(rows):
        for j in range(cols):

            # add grid points
            if j == 3:
                axs[i,j].set_xlim([-0.1, 1.1])
                axs[i,j].set_ylim([-0.1, 1.1])
            
            if j < 3:
                if j < 2:
                    cbar = field_plot(axs[i,j], \
                            uvec[i][j], \
                            nodes, cmap = cmap_vec[i][j])
                else:
                    if u_vec_plot == False:
                        cbar = field_plot(axs[i,j], \
                            uvec[i][j], \
                            nodes, cmap = cmap_vec[i][j])
                    else:
                        cbar = field_plot(axs[i,j], \
                            uvec[i][j], \
                            nodes, cmap = cmap_vec[i][j], is_displacement = True, add_displacement_to_nodes = True)

            else:
                uob = uvec[i][j]
                if u_vec_plot == False:
                    cbar = point_plot(axs[i,j], uob, x_obs, cmap = cmap_vec[i]
                    [j])
                else:
                    cbar = point_plot(axs[i,j], uob, x_obs, cmap = cmap_vec[i]
                    [j], is_displacement = True, add_displacement_to_nodes = True)

            divider = make_axes_locatable(axs[i,j])
            cax = divider.append_axes('right', size='8%', pad=0.03)
            cax.tick_params(labelsize=fs)
            cbar = fig.colorbar(cbar, cax=cax, orientation='vertical')
            if j < cols - 1:
                axs[i,j].axis('off')
            else:
                if u_vec_plot == False:
                    axs[i,j].set_xticks([0, 0.5, 1])
                    axs[i,j].set_yticks([0, 0.5, 1])
                else:
                    axs[i,j].axis('off')
            if title_vec is not None:
                tt = title_vec[i][j]
                if i > 0:
                    u1, u2 = uvec[0][j], uvec[i][j]
                    err = np.linalg.norm(u1 - u2)/np.linalg.norm(u1)
                    tt += '\n' + r'err (l2 rel) = {:.2f}%'.format(err*100)
                axs[i,j].set_title(tt, fontsize=fs, pad=ttl_pad)

    fig.tight_layout()
    if sup_title is not None:
        fig.suptitle(sup_title, fontsize=1.25*fs, y = y_sup_title)
    if savefilename is not None:
        plt.savefig(savefilename,  bbox_inches='tight')
    plt.show()

def mcmc_plot_fields(mcmc, savefilename = None, params = None):

    w_mean = mcmc.tracer.accepted_samples_mean_m
    w_sample_i = len(mcmc.tracer.accepted_samples_m) - 1
    w_sample = mcmc.tracer.accepted_samples_m[w_sample_i]

    mcmc_plot_fields_base(w_mean, w_sample, w_sample_i, mcmc, savefilename = savefilename, params = params)