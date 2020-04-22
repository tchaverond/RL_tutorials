# -*- coding: utf-8 -*-


import numpy as np

import grid_world

np.random.seed(42)


def pick_random(actions):
    return np.random.choice(actions, 1)[0]

class PolicyOptimizer:

    def __init__(self, environment, discount_factor=0.9, explore_scheme='log', exponential_decay=0.1):

        self.env = environment
        self.discount = discount_factor
        self.explore_scheme = explore_scheme
        self.explore_threshold = 1.0
        self.exponential_decay = exponential_decay
        self.action_values = {}
        self.visits = {}
        self.policy = {}
        self.cur_episode = 0
        self.episode_function = None


    def setup(self, method):

        for s in self.env.all_states():
            if self.env.is_terminal(s):
                self.action_values[(s,'X')] = 0
            else:
                for a in self.env.actions[s]:
                    self.action_values[(s,a)] = 0
        self.visits = {(s,a):0 for s in self.env.all_states()
                       if not self.env.is_terminal(s) for a in self.env.actions[s]}
        # starting with random policy
        self.policy = {s: np.random.choice(self.env.actions[s], 1)[0]
                       for s in self.env.all_states() if not self.env.is_terminal(s)}
        # self.policy = {(2,0): 'U', (1,0): 'U', (2,3): 'U', (0,0): 'R', (0,1): 'R',
        #                (0,2): 'R', (1,2): 'R', (2,1): 'R', (2,2): 'R'}
        if method == 'sarsa':
            self.episode_function = self.play_episode_sarsa
        elif method == 'qlearning':
            self.episode_function = self.play_episode_qlearning
        else:
            raise ValueError("Unknown method (expecting sarsa or qlearning).")


    def play_episode_sarsa(self, wind=None, wind_force=0.5):

        self.env.restart()
        self.cur_episode += 1
        s = self.env.current_state()
        action = self.epsilon_greedy(s)
        while not self.env.is_terminal(s):
            reward = self.env.move(action)
            s_prime = self.env.current_state()
            a_prime = self.windy_epsilon_greedy(s_prime, wind, wind_force)
            old_value = self.action_values[(s, action)]
            self.update_state_value(s, action, reward, s_prime, a_prime)
            self.update_policy(s, action, old_value)
            s = s_prime
            action = a_prime


    def play_episode_qlearning(self, wind=None, wind_force=0.5):

        self.env.restart()
        self.cur_episode += 1
        s = self.env.current_state()
        while not self.env.is_terminal(s):
            action = self.windy_epsilon_greedy(s, wind, wind_force)
            reward = self.env.move(action)
            s_prime = self.env.current_state()
            a_prime = self.policy.get(s_prime, 'X')
            old_value = self.action_values[(s, action)]
            self.update_state_value(s, action, reward, s_prime, a_prime)
            self.update_policy(s, action, old_value)
            s = s_prime


    def windy_epsilon_greedy(self, cur_state, wind, wind_force):

        action = self.epsilon_greedy(cur_state)
        if wind is not None and np.random.random() > 1 - wind_force:
            if wind == 'random':
                action = pick_random(self.env.actions.get(cur_state, ['X']))
            if wind == 'right' and 'R' in self.env.actions.get(cur_state, ['X']):
                action = 'R'
        return action


    def epsilon_greedy(self, cur_state):

        explore = np.random.random() < self.explore_threshold
        if explore and self.explore_scheme is not None:
            return pick_random(self.env.actions.get(cur_state, ['X']))
        else:
            return self.policy.get(cur_state, 'X')   # 'X': undefined move at terminal state


    def update_state_value(self, cur_state, cur_action, reward, next_state, next_action):

        self.visits[(cur_state, cur_action)] += 1
        decay = self.exponential_decay / self.visits[(cur_state, cur_action)]
        self.action_values[(cur_state, cur_action)] += \
            decay * (reward + self.discount * self.action_values[(next_state, next_action)]
                     - self.action_values[(cur_state, cur_action)])


    def update_policy(self, cur_state, chosen_action, old_value):

        if chosen_action != self.policy[cur_state] and self.action_values[(cur_state, chosen_action)] > old_value:
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


    def run(self, nb_iter=10000, **kwargs):

        while self.cur_episode < nb_iter:
            self.episode_function(**kwargs)
            self.update_explore_threshold()
        grid_world.print_policy(self.policy, self.env)
        state_values = self.compute_state_values_from_action_values()
        grid_world.print_values(state_values, self.env)


    def compute_state_values_from_action_values(self):
        state_values = {}
        for sa, value in self.action_values.items():
            state_values[sa[0]] = max(state_values.get(sa[0], -1), value)
        return state_values



if __name__ == "__main__":

    optimizer = PolicyOptimizer(environment=grid_world.negative_grid())
    optimizer.setup(method='qlearning')
    optimizer.run()

