import random

import numpy as np

from tasks.task import Task


class MOShapes(Task):
    """
        Gridworld with multiple objectives
    """
    LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3

    def __init__(self, maze, shape_rewards, fail_prob=1., fail_reward=-0.1, goal_reward=1.):

        self.height, self.width = maze.shape
        self.maze = maze
        self.shape_rewards = shape_rewards
        shape_types = sorted(list(shape_rewards.keys()))
        self.all_shapes = dict(zip(shape_types, range(len(shape_types))))
        self.fail_prob = fail_prob
        self.fail_reward = fail_reward
        self.goal_reward = goal_reward
        self.constrained = False
        
        self.goal = None
        self.initial = []
        self.occupied = set()
        self.shape_ids = dict()
        self.traps = set()
        for c in range(self.width):
            for r in range(self.height):
                if maze[r, c] == 'G':
                    self.goal = (r, c)
                elif maze[r, c] == '_':
                    self.initial.append((r, c))
                elif maze[r, c] == 'X':
                    self.occupied.add((r, c))
                elif maze[r, c] == 'T':
                    self.traps.add((r, c))
                elif maze[r, c] in {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}:
                    self.shape_ids[(r, c)] = len(self.shape_ids)
        self.unsafe = {(7,0),(7,5),(12,5),(0,7),(5,7),(5,12)}

    def clone(self):
        return MOShapes(self.maze, self.shape_rewards, self.fail_prob, self.fail_reward, self.goal_reward)

    def initialize(self):
        self.state = (random.choice(self.initial), tuple(0 for _ in range(len(self.shape_ids))))
        return self.state

    def encode(self, state):
        (y, x), coll = state
        n_state = self.width + self.height
        result = np.zeros((n_state + len(coll),))
        result[y] = 1
        result[self.height + x] = 1
        result[n_state:] = np.array(coll)
        result = result.reshape((1, -1))
        return result
    
    def encode_dim(self):
        return self.width + self.height + len(self.shape_ids)

    def action_count(self):
        return 4

    def transition(self, action):
        (row, col), collected = self.state
            
        # perform the movement
        if action == MOShapes.LEFT: 
            col -= 1
        elif action == MOShapes.UP: 
            row -= 1
        elif action == MOShapes.RIGHT: 
            col += 1
        elif action == MOShapes.DOWN: 
            row += 1
        else:
            raise Exception('bad action {}'.format(action))
        
        # out of bounds, cannot move
        if col < 0 or col >= self.width or row < 0 or row >= self.height:
            return self.state, np.array([0., 0.]), False, (False, False)

        # into a blocked cell, cannot move
        s1 = (row, col)
        if s1 in self.occupied:
            return self.state, np.array([0., 0.]), False, (False, False)
            
        # can now move
        self.state = (s1, collected)
        
        # into a goal cell
        if s1 == self.goal:
            return self.state, np.array([self.goal_reward, 0.]), True, (False, False)
        
        # into a shape cell
        if s1 in self.shape_ids:
            shape_id = self.shape_ids[s1]
            if collected[shape_id] == 1:
                return self.state, np.array([0., 0.]), False, (False, False)
            else:
                collected = list(collected)
                collected[shape_id] = 1
                collected = tuple(collected)
                self.state = (s1, collected)
                reward = self.shape_rewards[self.maze[row, col]]
                if s1 in self.unsafe:
                    return self.state, np.array([reward, 0.]), False, (False, True)
                else:
                    return self.state, np.array([reward, 0.]), False, (False, False)
        
        # trap
        if (row, col) in self.traps:
            if random.random() < self.fail_prob:
                return self.state, np.array([0., self.fail_reward]), False, (True, False)

        # into an empty cell
        return self.state, np.array([0., 0.]), False, (False, False)

    def features(self, state, action, next_state, noise):
        fail, _ = noise
        s1, _ = next_state
        _, collected = state
        nc = len(self.all_shapes)

        phi = np.zeros((nc + 2,))
        if s1 in self.shape_ids:
            if collected[self.shape_ids[s1]] != 1:
                y, x = s1
                shape_index = self.all_shapes[self.maze[y, x]]
                phi[shape_index] = 1.
        elif s1 == self.goal:
            phi[nc] = 1.
        if fail:
            phi[nc + 1] = 1.
        return phi
    
    def feature_dim(self):
        return len(self.all_shapes) + 2

    def get_wr(self):
        nc = len(self.all_shapes)
        w = np.zeros((nc + 2, 1))
        for shape, shape_index in self.all_shapes.items():
            w[shape_index, 0] = self.shape_rewards[shape]
        w[nc, 0] = self.goal_reward
        return w

    def get_wc(self):
        nc = len(self.all_shapes)
        w = np.zeros((nc + 2, 1))
        w[nc + 1, 0] = self.fail_reward
        return w

    def get_w(self):
        nc = len(self.all_shapes)
        w = np.zeros((nc + 2, 1))
        for shape, shape_index in self.all_shapes.items():
            w[shape_index, 0] = self.shape_rewards[shape]
        w[nc, 0] = self.goal_reward
        w[nc + 1, 0] = self.fail_reward
        return w

    def get_initial_reward(self):
        return np.zeros(2)
