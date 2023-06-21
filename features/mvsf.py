import warnings

import numpy as np


class SF:

    def __init__(self, learning_rate_w, *args, use_true_reward=False, **kwargs):
        self.alpha_w = learning_rate_w
        self.use_true_reward = use_true_reward
        if len(args) != 0 or len(kwargs) != 0:
            print(self.__class__.__name__ + ' ignoring parameters ' + str(args) + ' and ' + str(kwargs))
            
    def build_successor(self, task, source=None):
        raise NotImplementedError
        
    def get_successor(self, state, policy_index):
        raise NotImplementedError
    
    def get_successors(self, state):
        raise NotImplementedError
    
    def update_successor(self, transitions, policy_index):
        raise NotImplementedError
        
    def reset(self):
        self.n_tasks = 0
        self.psi = []
        self.true_w = []
        self.fit_w = []
        self.gpi_counters = []

    def add_training_task(self, task, source=None, n_samples=None):
        self.psi.append(self.build_successor(task, source))
        self.n_tasks = len(self.psi)

        true_w = task.get_w()
        self.true_w.append(true_w)
        if self.use_true_reward:
            fit_w = true_w
        else:
            n_features = task.feature_dim()
            fit_w = np.random.uniform(low=-0.01, high=0.01, size=(n_features, 1))
        self.fit_w.append(fit_w)
        
        for i in range(len(self.gpi_counters)):
            self.gpi_counters[i] = np.append(self.gpi_counters[i], 0)
        self.gpi_counters.append(np.zeros((self.n_tasks,), dtype=int))
        
    def update_reward(self, phi, r, task_index):
        w = self.fit_w[task_index]
        phi = phi.reshape(w.shape)
        if not self.use_true_reward:
            r_fit = np.sum(phi * w)
            self.fit_w[task_index] = w + self.alpha_w * (r - r_fit) * phi

        r_true = np.sum(phi * self.true_w[task_index])
        if not np.allclose(r, r_true):
            raise Exception('sampled reward {} != linear reward {} - please check task {}!'.format(
                r, r_true, task_index))
    
    def GPE_w(self, state, policy_index, w):
        psi = self.get_successor(state, policy_index)
        q = psi @ w
        return q[:,:, 0]
        
    def GPE(self, state, policy_index, task_index):
        return self.GPE_w(state, policy_index, self.fit_w[task_index])
    
    def GPI_w(self, state, w):
        psi = self.get_successors(state)
        q = (psi @ w)[:,:,:, 0]
        task = np.squeeze(np.argmax(np.max(q, axis=2), axis=1))
        return q, task
    
    def GPI(self, state, task_index, update_counters=False):
        q, task = self.GPI_w(state, self.fit_w[task_index])
        if update_counters:
            self.gpi_counters[task_index][task] += 1
        return q, task
    
    def GPI_usage_percent(self, task_index):
        counts = self.gpi_counters[task_index]        
        return 1. - (float(counts[task_index]) / np.sum(counts))


class MVSF:
    
    def __init__(self, learning_rate_w, risk_aversion, *args, use_true_reward=False, rank='full', **kwargs):
        self.alpha_w = learning_rate_w
        self.risk_aversion = risk_aversion
        self.use_true_reward = use_true_reward
        self.rank = rank
    
    def build_successor(self, task, source=None):
        raise NotImplementedError
        
    def get_successor(self, states, index):
        raise NotImplementedError  
    
    def get_successors(self, states):
        raise NotImplementedError
    
    def update_successor(self, transitions, index):
        raise NotImplementedError

    def reset(self):
        self.n_tasks = 0
        self.psi = []
        self.Sigma = []
        self.true_w = []
        self.fit_w = []
        self.gpi_counters = []

    def add_training_task(self, task, source=None, n_samples=None):
        psi, Sigma = self.build_successor(task, source)
        self.psi.append(psi)
        self.Sigma.append(Sigma)
        self.n_tasks = len(self.psi)
        
        true_w = task.get_w()
        self.true_w.append(true_w)
        if self.use_true_reward:
            fit_w = true_w
        else:
            n_features = task.feature_dim()
            fit_w = np.random.uniform(low=-0.01, high=0.01, size=(n_features, 1))
        self.fit_w.append(fit_w)
        
        for i in range(len(self.gpi_counters)):
            self.gpi_counters[i] = np.append(self.gpi_counters[i], 0)
        self.gpi_counters.append(np.zeros((self.n_tasks,), dtype=int))
    
    def update_reward(self, phi, r, index):
        w = self.fit_w[index]
        phi = phi.reshape(w.shape)
        if not self.use_true_reward:
            error = r - np.sum(phi * w)
            self.fit_w[index] = w + self.alpha_w * error * phi

        w = self.true_w[index]
        r_true = np.sum(phi * w)
        if not np.allclose(r, r_true):
            raise Exception('sampled reward {} != linear reward {} - please check task {}!'.format(r, r_true, index))
    
    def GPE_w(self, states, policy_index, w):
        psi, Sigma = self.get_successor(states, policy_index)
        mean = (psi @ w)[:,:, 0]
        variance = (w.T @ Sigma @ w)[:,:, 0, 0]
        if np.min(variance) < 0.:
            warnings.warn('sigma not psd!')
            variance = np.maximum(variance, 0.)
        
        return mean - self.risk_aversion * variance
    
    def GPE(self, states, policy_index, task_index):
        w = self.fit_w[task_index]
        return self.GPE_w(states, policy_index, w)
    
    def GPI_w(self, states, w):
        psi, Sigma = self.get_successors(states)
        mean = (psi @ w)[:,:,:, 0]
        variance = (w.T @ Sigma @ w)[:,:,:, 0, 0]
        if np.min(variance) < 0.:
            warnings.warn('sigma not psd!')
            variance = np.maximum(variance, 0.)
        
        q = mean - self.risk_aversion * variance
        task = np.squeeze(np.argmax(np.max(q, axis=2), axis=1))
        return q, task
    
    def GPI(self, states, index, update_counters=False):
        w = self.fit_w[index]
        q, task = self.GPI_w(states, w)
        if update_counters:
            self.gpi_counters[index][task] += 1
        return q, task
    
    def GPI_usage_percent(self, index):
        counts = self.gpi_counters[index]        
        return 1. - (float(counts[index]) / np.sum(counts))
