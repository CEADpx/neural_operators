import numpy as np

class State:

    def __init__(self, m_dim, u_dim, u_obs_dim):
        self.m = np.zeros(m_dim)
        self.u = np.zeros(u_dim)
        self.u_obs_dim = u_obs_dim
        self.u_obs = np.zeros(u_obs_dim)
        self.err_obs = np.zeros(u_obs_dim)
        self.log_likelihood = 0
        self.log_posterior = 0
        self.log_prior = 0
        self.cost = 0

    def set(self, a):
        self.m = a.m.copy()
        self.u = a.u.copy()
        self.u_obs = a.u_obs.copy()
        self.err_obs = a.err_obs.copy()
        self.log_likelihood = a.log_likelihood
        self.log_posterior = a.log_posterior
        self.log_prior = a.log_prior
        self.cost = a.cost