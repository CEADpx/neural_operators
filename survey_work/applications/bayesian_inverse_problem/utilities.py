import sys
import time
import os
import numpy as np
from scipy.interpolate import griddata

src_path = "../../../src/"
sys.path.append(src_path + 'mcmc')
from state import State
from tracer import Tracer
from compute_sample_errors import compute_sample_errors

sys.path.append(src_path + 'plotting')
from plot_curve import plot_curve
from mcmc_plot_fields import mcmc_plot_fields

class MCMC:
    def __init__(self, model, prior, data, sigma_noise, pcn_beta = 0.2, \
                 surrogate_to_use = None, surrogate_models = None, seed = 0):
        
        # model class that provides solveFwd()
        self.model = model
        self.m_dim = self.model.m_dim
        self.u_dim = self.model.u_dim

        self.u_nodes = self.model.u_nodes
        
        # prior class that provides () and logPrior()
        self.prior = prior

        # data (dict) that provides x_obs, u_obs, m_true, u_true, etc.
        self.data = data

        # noise (std-dev) in the observations
        self.sigma_noise = sigma_noise

        # preconditioned Crank-Nicolson beta --> proposal = beta * new_sample + (1-beta^2)^(1/2) * current 
        self.pcn_beta = pcn_beta

        # seed
        self.seed = seed
        
        # unpack data
        self.x_obs = data['x_obs']
        self.grid_x_obs = data['grid_x']
        self.grid_y_obs = data['grid_y']
        self.u_obs = data['u_obs']
        self.u_obs_dim = len(self.x_obs)
        self.w_true = data['w_true']
        self.m_true = data['m_true']
        self.u_true = data['u_true']

        # current and proposed input parameter and state variables
        self.current = State(self.m_dim, self.u_dim, self.u_obs_dim)
        self.proposed = State(self.m_dim, self.u_dim, self.u_obs_dim)
        self.init_sample = State(self.m_dim, self.u_dim, self.u_obs_dim)

        # store mcmc parameters
        self.n_samples = 0
        self.n_burnin = 0

        # surrogate models (dictionary of SurrogateModel objects)
        # models are accessed by the key, .e.g., self.surrogate_models['DeepONet']
        self.surrogate_to_use = surrogate_to_use
        self.surrogate_models = surrogate_models

        # tracer
        self.tracer = Tracer(self) 
        self.log_file = None

    def get_mcmc_params_for_tracer(self):
        return {'pcn_beta': self.pcn_beta, \
                'sigma_noise': self.sigma_noise, \
                'seed': self.seed, \
                'n_samples': self.n_samples, \
                'n_burnin': self.n_burnin, \
                'surrogate_to_use': self.surrogate_to_use}

    def solveFwd(self, current):
        if self.surrogate_to_use is not None:
            current.u = self.surrogate_models[self.surrogate_to_use].solveFwd(current.m)
        else:
            current.u = self.model.solveFwd(u = current.u, m = current.m, transform_m = True)
        
        return current.u
    
    def state_to_obs(self, u):
        return griddata(self.u_nodes, u, self.x_obs, method='linear')
    
    def logLikelihood(self, current):
        current.u = self.solveFwd(current)
        current.u_obs = self.state_to_obs(current.u)
        current.err_obs = current.u_obs - self.u_obs
        current.cost = 0.5 * np.linalg.norm(current.err_obs)**2 / self.sigma_noise**2
        current.log_likelihood = -current.cost

        return current.log_likelihood
    
    def logPosterior(self, current):
        current.log_prior = self.prior.logPrior(current.m)
        current.log_likelihood = self.logLikelihood(current)
        current.log_posterior = current.log_prior + current.log_likelihood
        return current.log_posterior
    
    def proposal(self, current, proposed):
        # preconditioned Crank-Nicolson
        # self.prior.get() returns the new sample
        proposed.m, proposed.log_prior = self.prior(proposed.m)
        return self.pcn_beta * proposed.m + np.sqrt(1 - self.pcn_beta**2) * current.m
    
    def sample(self, current):
        # compute the proposed state
        self.proposed.m = self.proposal(current, self.proposed)
        self.proposed.log_likelihood= self.logPosterior(self.proposed)
        
        # accept or reject (based on log-likelihood, i.e., -cost, for preconditioned Crank Nicholson following Stuart 2010 and HippyLib)
        alpha = current.cost - self.proposed.cost # or -current.log_likelihood + self.proposed.log_likelihood
        # alpha = self.proposed.log_prior + self.proposed.log_likelihood - current.log_prior - current.log_likelihood
        
        if alpha > np.log(np.random.uniform()):
            current.set(self.proposed)
            return 1

        return 0
    
    def run(self, init_m = None, n_samples = 1000, \
            n_burnin = 100, pcn_beta = 0.2, sigma_noise = 0.01, \
            savepath = './', savefilename = 'tracer', \
            save_every = 100, print_every = 100, \
            print_lvl = 1, display_plot_every = 100):
        
        # set the parameters
        self.n_samples = n_samples
        self.n_burnin = n_burnin
        self.pcn_beta = pcn_beta
        self.sigma_noise = sigma_noise
        
        self.tracer.savepath = savepath
        self.tracer.savefilename = self.tracer.savepath + savefilename
        self.tracer.save_every = save_every
        self.tracer.print_every = print_every
        self.tracer.display_plot_every = display_plot_every
        self.tracer.print_lvl = print_lvl
        self.tracer.mcmc_params = self.get_mcmc_params_for_tracer()

        # sampling time
        start_time = time.perf_counter()
        self.t = time.perf_counter()
        
        # print the initial message
        self.init_logger()

        # initialize the current state
        if init_m is not None:
            self.current.m = init_m
        else:
            self.current.m, self.current.log_prior = self.prior(self.current.m)

        self.current.log_posterior = self.logPosterior(self.current)
        if self.tracer.print_lvl > 1:
            self.logger('initializing the current state. Initial cost: {:.3e}'.format(self.current.cost)) 
        
        self.init_sample.set(self.current)

        # initial state
        # self.tracer.init(self.current)
        
        # run the MCMC
        init_done = False
        for i in range(n_samples + n_burnin):

            # sample
            accept = self.sample(self.current)

            # postprocess/print
            self.process_and_print(i)

            if i < n_burnin:
                continue
            
            # append
            if not init_done:
                self.tracer.init(self.current, self.init_sample)
                init_done = True
            
            self.save(i, self.current, accept)

        # save the final state
        self.tracer.append(i, self.current, accept, force_save=True)

        # print final message
        end_time = time.perf_counter()
        self.logger('-'*50)
        self.logger('MCMC finished in {:.3e}s. \nTotal samples: {:4d}, Accepted samples: {:4d}, Acceptance Rate: {:.3e}, Cost mean: {:.3e}'.format(end_time - start_time, n_samples + n_burnin, self.tracer.acceptances, self.tracer.current_acceptance_rate, self.tracer.accepted_samples_cost_mean))
        self.logger('-'*50)

        self.log_file.close()

    def init_logger(self):

        self.logger('-'*50)
        self.logger('MCMC started with {} samples and {} burnin'.format(self.n_samples, self.n_burnin))
        self.logger('PCN beta: {:.3e}, sigma_noise: {:.3e}'.format(self.pcn_beta, self.sigma_noise))
        if self.surrogate_to_use is not None:
            self.logger('Model used for solving the forward problem: {}'.format(self.surrogate_to_use))
        else:
            self.logger('Model used for solving the forward problem: {}'.format('True Model'))
        x = self.tracer.savefilename.split(os.path.sep)
        self.logger('Tracer save path: {}'.format(x[-2]))
        self.logger('Tracer save filename: {}'.format(x[-1]))
        
        self.logger('-'*50)
    
    def logger(self, s):
        if self.log_file is None or self.log_file.closed:
            self.log_file = open(self.tracer.savefilename + '_mcmc.log', 'w')
        
        self.log_file.write(s + '\n')
        self.log_file.flush()
        print(s)


    def process_and_print(self, i):
        
        # print msg
        if i % self.tracer.print_every == 0 and self.tracer.print_lvl > 0:
            dt = time.perf_counter() - self.t
            self.t = time.perf_counter()
            self.logger('-'*50)
            if i < self.n_burnin:
                self.logger('Burnin: {:4d}, Cost: {:.3e}'.format(i, self.current.cost))
            else:
                self.logger('Sample: {:4d}, Accepted samples: {:4d}, Acceptance Rate: {:.3e}, Cost mean: {:.3e}. \nTracing {} samples took {:.3e}s'.format(i, \
                        self.tracer.acceptances, \
                        self.tracer.current_acceptance_rate, \
                        self.tracer.accepted_samples_cost_mean, self.tracer.print_every, dt))
                
                # print errors
                s = compute_sample_errors(self)
                self.logger(s)

            self.logger('-'*50)
        
        # plotting
        if i > self.n_burnin and i % self.tracer.display_plot_every == 0:
            self.plot(i)

    def plot(self, i):
        # plot the results
        if self.tracer.acceptances < 5:
            return 

        # cost
        plot_curve(self.tracer.accepted_samples_cost, xl=r'Samples', \
            yl= r'Cost = $-\log(\pi_{like}(u_{obs} | w))$', \
            fs = 20, lw = 3, \
            savefile = self.tracer.savepath + 'cost_iter_{}.png'.format(i), \
            figsize=[6,4])
        
        # acceptance rate
        plot_curve(self.tracer.acceptance_rate, xl=r'Samples', \
            yl= r'Acceptance rate', \
            fs = 20, lw = 3, \
            savefile = self.tracer.savepath + 'acceptance_rate_iter_{}.png'.format(i), \
            figsize=[6,4])

        # compare true and posterior mean fields
        mcmc_plot_fields(self, savefilename = self.tracer.savepath + 'true_and_posterior_mean_w_m_u_iter_{}.png'.format(i))

    def save(self, i, current, accept):
        # save tracer
        self.tracer.append(i, current, accept)

