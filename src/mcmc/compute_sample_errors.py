import numpy as np

def compute_sample_errors(mcmc):
    n = 4
    true_vec = [mcmc.data['w_true'], mcmc.data['m_true'], mcmc.data['u_true'], mcmc.data['u_obs']]
    true_vec_norm = [np.linalg.norm(true_vec[i]) for i in range(4)] 

    w_mean = mcmc.tracer.accepted_samples_mean_m
    m_mean = mcmc.model.transform_gaussian_pointwise(w_mean)
    u_mean = mcmc.model.solveFwd(u = None, m = m_mean, transform_m = False)
    u_obs_mean = mcmc.state_to_obs(u_mean)
    mean_vec = [w_mean, m_mean, u_mean, u_obs_mean]
    
    mean_err_vec_norm = [100*np.linalg.norm(mean_vec[i] \
            - true_vec[i])/true_vec_norm[i] for i in range(4)]
    
    w_sample = mcmc.tracer.accepted_samples_m[-1]
    m_sample = mcmc.model.transform_gaussian_pointwise(w_sample)
    u_sample = mcmc.model.solveFwd(u = None, m = m_sample, transform_m = False)
    u_obs_sample = mcmc.state_to_obs(u_sample)
    sample_vec = [w_sample, m_sample, u_sample, u_obs_sample]
    
    sample_err_vec_norm = [100*np.linalg.norm(sample_vec[i] \
            - true_vec[i])/true_vec_norm[i] for i in range(4)]

    err_tag = ['w', 'm', 'u', 'u_obs']
    rs = []
    for i in range(4):
        me = '||{} - {}_mean|| = {:.3e}'.format(err_tag[i], \
            err_tag[i], mean_err_vec_norm[i])
        se = '||{} - {}_sample|| = {:.3e}'.format(err_tag[i], \
            err_tag[i], sample_err_vec_norm[i])
        
        rs.append('Error (%) in {}: {}, {}'.format(err_tag[i], \
            me, se))
        
    s = rs[0]
    for i in range(1, len(rs)):
        s += '\n' + rs[i]
    return s