import time
import random

import numpy as np


class Agent:
    
    def __init__(self, agent_name, encoding, *args,
                 gamma, T, epsilon=0.1, epsilon_decay=1., epsilon_min=0., epsilon_iter=0,
                 freq_print=1000, freq_save=200, logger, **kwargs):
        self.key = agent_name
        if encoding is None:
            encoding = lambda s: s
        self.encoding = encoding

        self.gamma = gamma
        self.T = T
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_iter = epsilon_iter

        self.freq_print = freq_print
        self.freq_save = freq_save
        self.logger = logger

    def get_Q_values(self, s, s_enc):
        raise NotImplementedError
    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma, noise):
        raise NotImplementedError
    
    # ===========================================================================
    # TASK MANAGEMENT
    # ===========================================================================
    def reset(self, n_tasks, n_samples):
        self.tasks = []
        self.phis = []
        self.start_time = time.time()
        
        self.iters = 0
        self.cum_fails, self.cum_reward = 0, np.zeros(2)
        self.r_good, self.r_bad = np.zeros(2), np.zeros(2)
        len_save = n_tasks*n_samples//self.freq_save
        self.fails_hist, self.reward_hist = [0.0]*len_save, [0.0]*len_save
        self.episode_reward_hist_per_task, self.episode_fails_hist_per_task = [0.0]*len_save, [0.0]*len_save
        self.cum_fails_hist, self.cum_reward_hist = [0.0]*len_save, [0.0]*len_save
        self.r_good_hist, self.r_bad_hist = [0.0]*len_save, [0.0]*len_save
        
    def add_training_task(self, task, n_samples):
        self.tasks.append(task)
        self.n_tasks = len(self.tasks)
        self.phis.append(task.features)
        if self.n_tasks == 1:
            self.n_actions = task.action_count()
            self.n_features = task.feature_dim()
            if self.encoding == 'task':
                self.encoding = task.encode
    
    def set_active_training_task(self, index):
        # set the task
        self.task_index = index
        self.active_task = self.tasks[index]
        self.phi = self.phis[index]
        
        # reset task-dependent counters
        self.s_init = self.active_task.initialize()
        self.s_enc_init = self.encoding(self.s_init)
        r_init = self.active_task.get_initial_reward()

        self.s = self.s_enc = None
        self.new_episode = True
        self.episode, self.episode_fails, self.episode_reward = 0, 0, r_init
        self.steps_since_last_episode, self.fails_since_last_episode, self.reward_since_last_episode = 0, 0, r_init
        self.steps, self.fails, self.reward = 0, 0, r_init
        self.epsilon = self.epsilon_init
        self.episode_reward_hist = []
        
    # ===========================================================================
    # TRAINING
    # ===========================================================================
    def _epsilon_greedy(self, q):
        if q.size != self.n_actions:
            raise Exception('model actions {} != env actions {}'.format(q.size, self.n_actions))
        
        if random.random() <= self.epsilon:
            a = random.randrange(self.n_actions)
        else:
            a = np.argmax(q)

        if self.iters > self.epsilon_iter:
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        
        return a
    
    def get_progress_strings(self):
        raise NotImplementedError

    def next_sample(self, viewer=None, n_view_ev=None):
        if self.new_episode:
            self.s = self.active_task.initialize()
            self.s_enc = self.encoding(self.s)
            r_init = self.active_task.get_initial_reward()
            self.new_episode = False
            self.episode += 1
            self.steps_since_last_episode = 0
            self.episode_fails = self.fails_since_last_episode
            self.fails_since_last_episode = 0
            self.episode_reward = self.reward_since_last_episode
            self.reward_since_last_episode = r_init
            if self.episode > 1:
                self.episode_reward_hist.append(self.episode_reward)  
        
        q = self.get_Q_values(self.s, self.s_enc)
        
        a = self._epsilon_greedy(q)
        
        s1, r, terminal, noise = self.active_task.transition(a)
        s1_enc = self.encoding(s1)
        if terminal:
            gamma = 0.
            self.new_episode = True
        else:
            gamma = self.gamma
        self.noise = noise
        
        self.train_agent(self.s, self.s_enc, a, r, s1, s1_enc, gamma, noise)
        
        self.s, self.s_enc = s1, s1_enc
        self.iters += 1
        self.steps += 1
        self.reward = self.reward + r
        self.steps_since_last_episode += 1
        self.reward_since_last_episode = self.reward_since_last_episode + r
        self.cum_reward = self.cum_reward + r
        
        if self.steps_since_last_episode >= self.T:
            self.new_episode = True
            
        if noise[0]:
            self.fails += 1
            self.cum_fails += 1
            self.fails_since_last_episode += 1

        if noise[1]:
            self.r_bad = self.r_bad + r
        else:
            self.r_good = self.r_good + r

        if (self.steps-1) % self.freq_print == 0:
            self.logger.info('\t'.join(self.get_progress_strings()))
        if (self.steps-1) % self.freq_save == 0:
            idx_save = self.iters // self.freq_save
            self.reward_hist[idx_save] = self.reward
            self.fails_hist[idx_save] = self.fails
            self.episode_reward_hist_per_task[idx_save] = self.episode_reward
            self.episode_fails_hist_per_task[idx_save] = self.episode_fails
            self.cum_reward_hist[idx_save] = self.cum_reward
            self.cum_fails_hist[idx_save] = self.cum_fails
            self.r_good_hist[idx_save] = self.r_good
            self.r_bad_hist[idx_save] = self.r_bad

        # viewing
        if viewer is not None and self.episode % n_view_ev == 0:
            viewer.update()
    
    def train_on_task(self, train_task, n_samples, viewer=None, n_view_ev=None):
        self.add_training_task(train_task, n_samples)
        self.set_active_training_task(self.n_tasks - 1)
        for _ in range(n_samples):
            self.next_sample(viewer, n_view_ev)

    def train(self, train_tasks, n_samples, viewers=None, n_view_ev=None, **kwargs):
        if viewers is None: 
            viewers = [None] * len(train_tasks)
        self.reset()
        for train_task, viewer in zip(train_tasks, viewers):
            self.train_on_task(train_task, n_samples, viewer, n_view_ev)
    