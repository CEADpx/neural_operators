import numpy as np
import pickle

class Tracer:

    def __init__(self, mcmc):

        self.mcmc_params = mcmc.get_mcmc_params_for_tracer()

        self.init_m = np.zeros(mcmc.m_dim)
        self.init_u = np.zeros(mcmc.u_dim)
        self.init_u_obs = np.zeros(mcmc.u_obs_dim)

        self.accepted_samples_m = np.zeros((1, mcmc.m_dim))
        self.accepted_samples_u = np.zeros((1, mcmc.u_dim))

        self.accepted_samples_mean_m = np.zeros(mcmc.m_dim)
        self.accepted_samples_std_m = np.zeros(mcmc.m_dim)
        self.accepted_samples_mean_u = np.zeros(mcmc.u_dim)
        self.accepted_samples_std_u = np.zeros(mcmc.u_dim)
        
        self.accepted_samples_obs_u = np.zeros((1, mcmc.u_obs_dim))
        self.accepted_samples_obs_mean_u = np.zeros(mcmc.u_obs_dim)
        self.accepted_samples_obs_err = np.zeros((1, mcmc.u_obs_dim))
        self.accepted_samples_obs_err_mean = np.zeros(mcmc.u_obs_dim)

        self.accepted_samples_log_likelihood = np.zeros(1)
        self.accepted_samples_log_posterior = np.zeros(1)
        self.accepted_samples_log_prior = np.zeros(1)
        self.accepted_samples_cost = np.zeros(1)

        self.accepted_samples_log_likelihood_mean = 0
        self.accepted_samples_log_posterior_mean = 0
        self.accepted_samples_log_prior_mean = 0
        self.accepted_samples_cost_mean = 0
        
        self.current_acceptance_rate = 0
        self.acceptances = 0
        self.acceptance_rate = np.zeros(1)
        self.accep_reject_record = np.zeros(1)

        self.append_sample_i = 0

        # save related
        self.savepath = './'
        self.savefilename = self.savepath + 'mcmc_results'
        self.save_every = 10
        self.statistics_every = 10

        # logging
        self.print_every = 10
        self.print_lvl = 1
        self.display_plot_every = self.print_every

    def init(self, current, init_sample):
        self.init_m = init_sample.m.copy()
        self.init_u = init_sample.u.copy()
        self.init_u_obs = init_sample.u_obs.copy()

        self.accepted_samples_m[0] = current.m
        self.accepted_samples_u[0] = current.u
        self.accepted_samples_obs_u[0] = current.u_obs
        self.accepted_samples_obs_err[0] = current.err_obs
        self.accepted_samples_log_likelihood[0] = current.log_likelihood
        self.accepted_samples_log_posterior[0] = current.log_posterior
        self.accepted_samples_log_prior[0] = current.log_prior
        self.accepted_samples_cost[0] = current.cost

    def stats(self):
        # compute statistics
        self.accepted_samples_mean_m = np.mean(self.accepted_samples_m, axis = 0)
        self.accepted_samples_std_m = np.std(self.accepted_samples_m, axis = 0)
        self.accepted_samples_mean_u = np.mean(self.accepted_samples_u, axis = 0)
        self.accepted_samples_std_u = np.std(self.accepted_samples_u, axis = 0)
        self.accepted_samples_obs_mean_u = np.mean(self.accepted_samples_obs_u, axis = 0)
        self.accepted_samples_obs_err_mean = np.mean(self.accepted_samples_obs_err, axis = 0)
        
        self.accepted_samples_log_likelihood_mean = np.mean(self.accepted_samples_log_likelihood)
        self.accepted_samples_log_posterior_mean = np.mean(self.accepted_samples_log_posterior)
        self.accepted_samples_log_prior_mean = np.mean(self.accepted_samples_log_prior)
        self.accepted_samples_cost_mean = np.mean(self.accepted_samples_cost)


    def append(self, i, current, accept, mcmc_params = None, force_save = False):

        # if we have seen i before, then we will not modify the data but we may need to save if force_save is True
        if i < self.append_sample_i:
            if force_save:
                # update stats
                self.stats()
                # save
                self.save()
            return
        
        self.append_sample_i = i        
        
        # update acceptances
        if accept == 1:
            self.acceptances += 1

        # append new data
        self.current_acceptance_rate = self.acceptances / (i + 1)
        self.acceptance_rate = np.append(self.acceptance_rate, self.current_acceptance_rate)
        self.accep_reject_record = np.append(self.accep_reject_record, accept)

        self.accepted_samples_m = np.vstack((self.accepted_samples_m, current.m))
        self.accepted_samples_u = np.vstack((self.accepted_samples_u, current.u))
        self.accepted_samples_obs_u = np.vstack((self.accepted_samples_obs_u, current.u_obs))
        self.accepted_samples_obs_err = np.vstack((self.accepted_samples_obs_err, current.err_obs))

        self.accepted_samples_log_likelihood = np.append(self.accepted_samples_log_likelihood, current.log_likelihood)
        self.accepted_samples_log_posterior = np.append(self.accepted_samples_log_posterior, current.log_posterior)
        self.accepted_samples_log_prior = np.append(self.accepted_samples_log_prior, current.log_prior)
        self.accepted_samples_cost = np.append(self.accepted_samples_cost, current.cost)
        
        # stats?
        if i % self.statistics_every == 0: 
            self.stats()

        # save?
        if i % self.save_every == 0:
            self.save(mcmc_params)

    def save(self, mcmc_params):

        if mcmc_params is not None:
            self.mcmc_params = mcmc_params
            
        # save using pickle
        with open(self.savefilename + '.pkl', 'wb') as f:
            pickle.dump(self, f)