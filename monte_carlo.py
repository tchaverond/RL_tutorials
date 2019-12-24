# -*- coding: utf-8 -*-


import numpy as np

import grid_world

np.random.seed(42)



class PolicyOptimizer:

    def __init__(self, environment, discount_factor=0.9, learning_scheme='log', epsilon=1e-4):

        self.env = environment
        self.discount = discount_factor
        self.learning_scheme = learning_scheme
        self.explore_threshold = 1.0
        self.epsilon = epsilon
        self.state_values = {}
        self.state_visits = {}
        # starting with no policy
        self.policy = {}
        # self.policy = {(2,0): 'U', (1,0): 'U', (2,3): 'U', (0,0): 'R', (0,1): 'R',
        #                (0,2): 'R', (1,2): 'R', (2,1): 'R', (2,2): 'R'}


    def perform_episode(self, wind=None, wind_force=0.5):

        def pick_random():
            return np.random.choice(['U', 'D', 'L', 'R'], 1)[0]

        self.env.restart()
        states = [self.env.current_state()]
        actions = []
        rewards = []
        while not self.env.is_terminal(self.env.current_state()):
            explore = np.random.random() < self.explore_threshold
            s = self.env.current_state()
            if explore and self.learning_scheme is not None:
                action = pick_random()
            elif s in self.policy:
                action = self.policy[s]
            else:
                action = pick_random()
            if wind is not None and np.random.random() > 1-wind_force:
                if wind == 'random':
                    action = pick_random()
                if wind == 'right':
                    action = 'R'   # it doesn't matter if the action is possible or not
            cur_reward = self.env.move(action)
            if self.env.current_state() != s:   # the intended move is valid
                actions.append(action)
                rewards.append(cur_reward)
                states.append(self.env.current_state())
        return states, actions, rewards


    def update_state_value_function(self, states, actions, rewards):

        for s,s_prime,a,r in zip(reversed(states[:-1]), reversed(states[1:]), reversed(actions), reversed(rewards)):
            # every-visit Monte Carlo
            # using 0 as default because terminal states are not known beforehand
            new_value = r + self.discount * self.state_values.get(s_prime, 0)
            if s in self.state_values:
                self.state_visits[s] += 1
                n = self.state_visits[s]
                if new_value > self.state_values[s]:
                    self.policy[s] = a
                self.state_values[s] = (1 - 1/np.log(n)) * self.state_values[s] + new_value/np.log(n)
            else:
                self.state_visits[s] = 1
                self.state_values[s] = new_value


    def update_policy(self):
        pass


    def update_explore_threshold(self, t):

        if self.learning_scheme == 'linear':
            self.explore_threshold = 1 / t
        elif self.learning_scheme == 'cubic':
            self.explore_threshold = 1 / t**3
        elif self.learning_scheme == 'log':
            if t != 1:
                self.explore_threshold = 1 / np.log(t)
        elif self.learning_scheme == 'loglog':
            if t > 3:
                self.explore_threshold = 1 / np.log(np.log(t))



    def improve_policy(self, nb_iter=1000, **kwargs):

        for t in range(nb_iter):
            states, actions, rewards = self.perform_episode(**kwargs)
            self.update_state_value_function(states, actions, rewards)
            self.update_policy()
            self.update_explore_threshold(t+1)
        grid_world.print_policy(self.policy, self.env)
        grid_world.print_values(self.state_values, self.env)



if __name__ == "__main__":

    optimizer = PolicyOptimizer(environment=grid_world.negative_grid(), learning_scheme='loglog')
    # optimizer.improve_policy()
    optimizer.improve_policy(wind='right', wind_force=0.2)
