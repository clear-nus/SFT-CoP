from copy import deepcopy
from collections import defaultdict

import numpy as np

from features.mvsf import SF, MVSF


class TabularSF(SF):

    def __init__(self, alpha, *args, 
                 noise_init=lambda size: np.random.uniform(-0.01, 0.01, size=size), **kwargs):
        super(TabularSF, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.noise_init = noise_init
    
    def build_successor(self, task, source=None):
        if source is None or len(self.psi) == 0:
            n_actions = task.action_count()
            n_features = task.feature_dim()
            return defaultdict(lambda: self.noise_init((n_actions, n_features)))
        else:
            return deepcopy(self.psi[source])
    
    def get_successor(self, state, policy_index):
        return np.expand_dims(self.psi[policy_index][state], axis=0)
    
    def get_successors(self, state):
        return np.expand_dims(np.array([psi[state] for psi in self.psi]), axis=0)
    
    def update_reward(self, phi, r, index):
        w = self.fit_w[index]
        phi = phi.reshape(w.shape)
        if not self.use_true_reward:
            error = r.sum(0) - np.sum(phi * w)
            self.fit_w[index] = w + self.alpha_w * error * phi

        w = self.true_w[index]
        r_true = np.sum(phi * w)
        if not np.allclose(r.sum(0), r_true):
            raise Exception('sampled reward {} != linear reward {} - please check task {}!'.format(r, r_true, index))

    def update_successor(self, transitions, policy_index):
        for state, action, phi, next_state, next_action, gamma in transitions:
            psi = self.psi[policy_index]
            targets = phi.flatten() + gamma * psi[next_state][next_action,:] 
            errors = targets - psi[state][action,:]
            psi[state][action,:] = psi[state][action,:] + self.alpha * errors


class TabularMVSF(MVSF):
    
    def __init__(self, alpha, alpha_var, *args,
                 noise_init=lambda size: np.random.uniform(-0.01, 0.01, size=size), **kwargs):
        super(TabularMVSF, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.alpha_var = alpha_var
        self.noise_init = noise_init
    
    def build_successor(self, task, source=None):
        if source is None or len(self.psi) == 0:
            n_actions = task.action_count()
            n_features = task.feature_dim()
            psi = defaultdict(lambda: self.noise_init((n_actions, n_features)))
            Sigma = defaultdict(lambda: np.zeros((n_actions, n_features, n_features)))
        else:
            psi = deepcopy(self.psi[source])
            Sigma = deepcopy(self.Sigma[source])
        return psi, Sigma
                
    def get_successor(self, states, index):
        psi = np.expand_dims(self.psi[index][states], axis=0)
        Sigma = np.expand_dims(self.Sigma[index][states], axis=0)
        return psi, Sigma
    
    def get_successors(self, states):
        psi = np.expand_dims(np.array([psi[states] for psi in self.psi]), axis=0)
        Sigma = np.expand_dims(np.array([sigma[states] for sigma in self.Sigma]), axis=0)
        return psi, Sigma
    
    def update_reward(self, phi, r, index):
        w = self.fit_w[index]
        phi = phi.reshape(w.shape)
        if not self.use_true_reward:
            error = r.sum(0) - np.sum(phi * w)
            self.fit_w[index] = w + self.alpha_w * error * phi

        w = self.true_w[index]
        r_true = np.sum(phi * w)
        if not np.allclose(r.sum(0), r_true):
            raise Exception('sampled reward {} != linear reward {} - please check task {}!'.format(r, r_true, index))
    
    def update_successor(self, transitions, index):
        for state, action, phi, next_state, next_action, gamma in transitions:
            psi = self.psi[index]
            targets = phi.flatten() + gamma * psi[next_state][next_action,:] 
            error = targets - psi[state][action,:]
            psi[state][action,:] = psi[state][action,:] + self.alpha * error
             
            Sigma = self.Sigma[index]
            if self.rank == 'full':
                error2 = np.outer(error, error)
            elif self.rank == 'diag':
                error2 = np.diag(error ** 2)
            else:
                raise Exception('Invalid rank {}'.format(self.rank))
            targets2 = error2 + (gamma ** 2) * Sigma[next_state][next_action,:,:]
            error2 = targets2 - Sigma[state][action,:,:]
            Sigma[state][action,:,:] = Sigma[state][action,:,:] + self.alpha_var * error2
        

class TabularMOSF(SF):
    
    def __init__(self, alpha, eta, threshold, *args,
                 noise_init=lambda size: np.random.uniform(0.0, 0.01, size=size), **kwargs):
        super(TabularMOSF, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.eta = eta
        self.threshold = threshold
        self.noise_init = noise_init

    def reset(self):
        self.n_tasks = 0
        self.psi = []
        self.lambda_dual = []
        self.qc_iters = []
        self.true_wr = []
        self.true_wc = []
        self.fit_wr = []
        self.fit_wc = []
        self.gpi_counters = []

    def add_training_task(self, task, source=None, n_samples=None):
        psi = self.build_successor(task, source)
        self.psi.append(psi)
        
        dual_init = np.array([[0.0]])
        self.lambda_dual.append(dual_init)
        self.qc_iters.append(np.zeros([n_samples,1]))
        self.n_tasks = len(self.psi)
        
        true_wr = task.get_wr()
        true_wc = task.get_wc()
        self.true_wr.append(true_wr)
        self.true_wc.append(true_wc)
        if self.use_true_reward:
            fit_wr = true_wr
            fit_wc = true_wc
        else:
            n_features = task.feature_dim()
            fit_wr = np.random.uniform(low=-0.01, high=0.01, size=(n_features, 1))
            fit_wc = np.random.uniform(low=-0.01, high=0.01, size=(n_features, 1))
        self.fit_wr.append(fit_wr)
        self.fit_wc.append(fit_wc)
        
        for i in range(len(self.gpi_counters)):
            self.gpi_counters[i] = np.append(self.gpi_counters[i], 0)
        self.gpi_counters.append(np.zeros((self.n_tasks,), dtype=int))
    
    def build_successor(self, task, source=None):
        if source is None or len(self.psi) == 0:
            n_actions = task.action_count()
            n_features = task.feature_dim()
            psi = defaultdict(lambda: self.noise_init((n_actions, n_features)))
        else:
            psi = deepcopy(self.psi[source])
        return psi

    def get_successor(self, states, index):
        psi = np.expand_dims(self.psi[index][states], axis=0)
        return psi
    
    def get_successors(self, states):
        psi = np.expand_dims(np.array([psi[states] for psi in self.psi]), axis=0)
        return psi

    def init_lambda(self, s_init, task_index, logger):
        if self.n_tasks > 1:
            wr = self.fit_wr[task_index]
            wc = self.fit_wc[task_index]
            psi = self.get_successors(s_init)
            psi_pre = psi[:,:-1]
            qr = (psi_pre @ wr)[0,:,:,:]
            qc = (psi_pre @ wc)[0]
            threshold = np.array([[[self.threshold]]])
            qc_max = np.max(qc-threshold)
            if qc_max < 0:
                logger.info("Max Qc = {}, problem is infeasible".format(qc_max))
            else:
                lambda_dual = self.lambda_dual[task_index]
                for t in range(200000):
                    q = qr + (qc-threshold)@lambda_dual
                    q = q[:,:,0]
                    task = np.squeeze(np.argmax(np.max(q, axis=1), axis=0))
                    a = np.argmax(q[task])
                    grad = qc[task, a] - threshold.squeeze()
                    eta = 1000 if t < 10000 else 10000000/(t+1)
                    lambda_dual_next = np.maximum(lambda_dual - eta * grad[:,np.newaxis], 0.0)
                    if abs(lambda_dual_next-lambda_dual).mean() < 0.002:
                        break
                    lambda_dual = lambda_dual_next
                self.lambda_dual[task_index] = lambda_dual

    def update_reward(self, phi, r, index):
        wr = self.fit_wr[index]
        wc = self.fit_wc[index]
        phi = phi.reshape(wr.shape)
        if not self.use_true_reward:
            error_r = r[0] - np.sum(phi * wr)
            error_c = r[1] - np.sum(phi * wc)
            self.fit_wr[index] = wr + self.alpha_w * error_r * phi
            self.fit_wc[index] = wc + self.alpha_w * error_c * phi
        
        wr = self.true_wr[index]
        wc = self.true_wc[index]
        r_true = np.sum(phi * wr)
        c_true = np.sum(phi * wc)
        if not np.allclose(r[0], r_true):
            raise Exception('sampled reward {} != linear reward {} - please check task {}!'.format(r[0], r_true, index))
        if not np.allclose(r[1], c_true):
            raise Exception('sampled reward {} != linear reward {} - please check task {}!'.format(r[1], c_true, index))

    def update_successor(self, transitions, index):
        for state, action, phi, next_state, next_action, gamma in transitions:
            psi = self.psi[index]
            targets = phi.flatten() + gamma * psi[next_state][next_action,:] 
            error = targets - psi[state][action,:]
            psi[state][action,:] = psi[state][action,:] + self.alpha * error
        
    def updata_lambda(self, s_init, s, task_index, steps, steps_epis, gamma, GPI=True):
        wr = self.fit_wr[task_index]
        wc = self.fit_wc[task_index]
        psi = self.get_successors(s_init)
        qr = (psi @ wr)[0,:,:,:]
        qc = (psi @ wc)[0]
        threshold = np.array([[[self.threshold]]])
        for t in range(1):
            q = qr + (qc-threshold)@(self.lambda_dual[task_index])
            q = q[:,:,0]
            task = np.squeeze(np.argmax(np.max(q, axis=1), axis=0))
            a = np.argmax(q[task])
            if not GPI: a = np.argmax(q[task_index])
            grad = qc[task, a] - threshold.squeeze()
            self.qc_iters[task_index][steps] = grad
            eta = self.eta/(t+1)
            lambda_dual_next = np.maximum(self.lambda_dual[task_index] - eta * grad[:,np.newaxis], 0.0)
            self.lambda_dual[task_index] = lambda_dual_next

    def GPE_w(self, states, policy_index, wr, wc, lambda_dual):
        psi = self.get_successor(states, policy_index)
        qr = (psi @ wr)[0,:,:]
        qc = (psi @ wc)[0]
        threshold = np.array([[self.threshold]])
        q = qr + (qc-threshold)@lambda_dual
        return q[np.newaxis,:,0]
    
    def GPE(self, states, policy_index, task_index):
        wr = self.fit_wr[task_index]
        wc = self.fit_wc[task_index]
        lambda_dual = self.lambda_dual[task_index]
        return self.GPE_w(states, policy_index, wr, wc, lambda_dual)

    def GPI_w(self, states, wr, wc, lambda_dual):
        psi = self.get_successors(states)
        qr = (psi @ wr)[0,:,:,:]
        qc = (psi @ wc)[0]
        threshold = np.array([[[self.threshold]]])
        q = qr + (qc-threshold)@lambda_dual
        q = q[np.newaxis,:,:,0]
        task = np.squeeze(np.argmax(np.max(q, axis=2), axis=1))
        return q, task

    def GPI(self, states, index, update_counters=False):
        wr = self.fit_wr[index]
        wc = self.fit_wc[index]
        lambda_dual = self.lambda_dual[index]
        q, task = self.GPI_w(states, wr, wc, lambda_dual)
        if update_counters:
            self.gpi_counters[index][task] += 1
        return q, task
