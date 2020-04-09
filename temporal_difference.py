# -*- coding: utf-8 -*-


import numpy as np

import grid_world

np.random.seed(42)



class PolicyOptimizer:

    def __init__(self, environment, discount_factor=0.9, explore_scheme='log', exponential_decay=0.1):

        self.env = environment
        self.discount = discount_factor
        self.explore_scheme = explore_scheme
        self.explore_threshold = 1.0
        self.exponential_decay = exponential_decay
        self.state_values = {}
        self.policy = {}
        self.cur_episode = 0


    def setup(self):

        self.state_values = {s:0 for s in self.env.all_states()}
        # starting with random policy
        self.policy = {s: np.random.choice(self.env.actions[s], 1)[0]
                       for s in self.env.all_states() if not self.env.is_terminal(s)}
        # self.policy = {(2,0): 'U', (1,0): 'U', (2,3): 'U', (0,0): 'R', (0,1): 'R',
        #                (0,2): 'R', (1,2): 'R', (2,1): 'R', (2,2): 'R'}


    def play_episode(self, wind=None, wind_force=0.5):

        def pick_random():
            return np.random.choice(['U', 'D', 'L', 'R'], 1)[0]

        self.env.restart()
        self.cur_episode += 1
        while not self.env.is_terminal(self.env.current_state()):
            explore = np.random.random() < self.explore_threshold
            s = self.env.current_state()
            if explore and self.explore_scheme is not None:
                action = pick_random()
            else:
                action = self.policy[s]
            if wind is not None and np.random.random() > 1 - wind_force:
                if wind == 'random':
                    action = pick_random()
                if wind == 'right':
                    action = 'R'   # it doesn't matter if the action is possible or not
            reward = self.env.move(action)
            if self.env.current_state() != s:   # the intended move is valid
                old_value = self.update_state_value(s, self.env.current_state(), reward)
                self.update_policy(s, action, old_value)


    def update_state_value(self, cur_state, next_state, reward):

        if type(self.exponential_decay) == float:
            decay = self.exponential_decay
        elif self.exponential_decay == 'inverse':
            decay = 1 / self.cur_episode
        old = self.state_values[cur_state]
        self.state_values[cur_state] += decay * (reward + self.discount * self.state_values[next_state]
                                                 - self.state_values[cur_state])
        return old


    def update_policy(self, cur_state, chosen_action, old_value):

        if chosen_action != self.policy[cur_state] and self.state_values[cur_state] > old_value:
            self.policy[cur_state] = chosen_action


    def update_explore_threshold(self):

        t = self.cur_episode
        if self.explore_scheme == 'linear':
            self.explore_threshold = 1 / t
        elif self.explore_scheme == 'cubic':
            self.explore_threshold = 1 / t**3
        elif self.explore_scheme == 'log':
            if t != 1:
                self.explore_threshold = 1 / np.log(t)
        elif self.explore_scheme == 'loglog':
            if t > 3:
                self.explore_threshold = 1 / np.log(np.log(t))


    def run(self, nb_iter=1000, **kwargs):

        while self.cur_episode < nb_iter:
            self.play_episode(**kwargs)
            self.update_explore_threshold()
        grid_world.print_policy(self.policy, self.env)
        grid_world.print_values(self.state_values, self.env)



if __name__ == "__main__":

    optimizer = PolicyOptimizer(environment=grid_world.negative_grid())
    optimizer.setup()
    optimizer.run()

