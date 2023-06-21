import numpy as np

from agents.agent import Agent


class SFQL(Agent):
    
    def __init__(self, lookup_table, *args, use_gpi=True, **kwargs):
        super(SFQL, self).__init__(*args, **kwargs)
        self.sf = lookup_table
        self.use_gpi = use_gpi

    def get_Q_values(self, s, s_enc):
        q, self.c = self.sf.GPI(s_enc, self.task_index, update_counters=self.use_gpi)
        if not self.use_gpi:
            self.c = self.task_index
        return q[:, self.c,:]
    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma, noise):
        t = self.task_index
        phi = self.phi(s, a, s1, noise)
        self.sf.update_reward(phi, r, t)
        
        if self.use_gpi:
            q1, _ = self.sf.GPI(s1_enc, t)
            q1 = np.max(q1[0,:,:], axis=0)
        else:
            q1 = self.sf.GPE(s1_enc, t, t)[0,:]
        a1 = np.argmax(q1)
        transitions = [(s_enc, a, phi, s1_enc, a1, gamma)]
        self.sf.update_successor(transitions, t)
        
        if self.c != t:
            q1 = self.sf.GPE(s1_enc, self.c, self.c)
            next_action = np.argmax(q1)
            transitions = [(s_enc, a, phi, s1_enc, next_action, gamma)]
            self.sf.update_successor(transitions, self.c)
        
    def reset(self, n_tasks, n_samples):
        super(SFQL, self).reset(n_tasks, n_samples)
        self.sf.reset()
        
    def add_training_task(self, task, n_samples):
        super(SFQL, self).add_training_task(task, n_samples)
        self.sf.add_training_task(task, -1, n_samples)
        self.visits = np.zeros((13, 13))
    
    def get_progress_strings(self):
        sample_str = 'TASK {:>3}  STEP {:>6}  EPIS {:>4}  \u03B5 {:.4f}'.format(
            self.task_index+1, self.steps, self.episode, self.epsilon)
        reward_str = 'EPR {:>8.4f}  F {:>3}  R {:.4f}  Rg {:.4f}  Rb {:.4f}'.format(
            self.episode_reward[0], self.fails, self.reward[0], self.r_good[0], self.r_bad[0])
        gpi_percent = self.sf.GPI_usage_percent(self.task_index)
        w_error = np.linalg.norm(self.sf.fit_w[self.task_index] - self.sf.true_w[self.task_index])
        gpi_str = 'GPI% {:.4f}  WERR {:.4f}'.format(gpi_percent, w_error)

        return sample_str, reward_str, gpi_str


class SFTCoPQL(SFQL):

    def set_active_training_task(self, index):
        super(SFTCoPQL, self).set_active_training_task(index)
        self.sf.init_lambda(self.s_enc_init, self.task_index, self.logger)

    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma, noise):
        t = self.task_index
        phi = self.phi(s, a, s1, noise)
        self.sf.update_reward(phi, r, t)

        if self.use_gpi:
            q1, _ = self.sf.GPI(s1_enc, t)
            q1 = np.max(q1[0,:,:], axis=0)
        else:
            q1 = self.sf.GPE(s1_enc, t, t)[0,:]
        a1 = np.argmax(q1)
        transitions = [(s_enc, a, phi, s1_enc, a1, gamma)]
        self.sf.update_successor(transitions, t)
        
        if self.c != t:
            q1 = self.sf.GPE(s1_enc, self.c, self.c)
            next_action = np.argmax(q1)
            transitions = [(s_enc, a, phi, s1_enc, next_action, gamma)]
            self.sf.update_successor(transitions, self.c)
        
        self.sf.updata_lambda(self.s_enc_init, self.s_enc, t, self.steps, self.steps_since_last_episode, self.gamma)

    def get_progress_strings(self):
        sample_str = 'TASK {:>3}  STEP {:>5}  EPIS {:>4}  \u03B5 {:.4f}  Qc {:>20}'.format(
            self.task_index+1, self.steps, self.episode, self.epsilon, str(self.sf.qc_iters[self.task_index][self.steps-1]))
        reward_str = 'Lambda {:>26}  EPR {:>8.4f}  F {:>3}  R {:.4f}  Rg {:.4f}  Rb {:.4f}'.format(
            str(self.sf.lambda_dual[self.task_index].squeeze()), self.episode_reward[0], self.fails, self.reward[0], self.r_good[0], self.r_bad[0])
        gpi_percent = self.sf.GPI_usage_percent(self.task_index)
        wr_error = np.linalg.norm(self.sf.fit_wr[self.task_index] - self.sf.true_wr[self.task_index])
        wc_error = np.linalg.norm(self.sf.fit_wc[self.task_index] - self.sf.true_wc[self.task_index])
        gpi_str = 'GPI% {:.4f}  WRERR {:.4f}  WCERR {:.4f}'.format(gpi_percent, wr_error, wc_error)

        return sample_str, reward_str, gpi_str
