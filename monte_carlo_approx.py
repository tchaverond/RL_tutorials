# -*- coding: utf-8 -*-


import numpy as np

import grid_world

np.random.seed(42)

# gridworld feature normalization
x1 = np.array([0,0,0,1,1,2,2,2,3])
x2 = np.array([0,1,2,0,2,0,1,2,2])
means = (np.mean(x1), np.mean(x2), np.mean(x1**2), np.mean(x2**2), np.mean(x1*x2))
stds = (np.std(x1), np.std(x2), np.std(x1**2), np.std(x2**2), np.std(x1*x2))



class PolicyOptimizer:

    def __init__(self, environment, discount_factor=0.9, learning_scheme='log'):

        self.env = environment
        self.discount = discount_factor
        self.learning_scheme = learning_scheme
        self.explore_threshold = 1.0
        self.theta = np.zeros(5)  # our feature vector is 5-dimensional: x1, x2, x1^2, x2^2 and x1x2
        # starting with no policy
        self.policy = {}
        # self.policy = {(2,0): 'U', (1,0): 'U', (2,3): 'U', (0,0): 'R', (0,1): 'R',
        #                (0,2): 'R', (1,2): 'R', (2,1): 'R', (2,2): 'R'}


    def perform_episode(self, wind=None, wind_force=0.5):

        def pick_random():
            return np.random.choice(['U', 'D', 'L', 'R'], 1)[0]

        self.env.restart()
        states = [self.env.current_state()]
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
                rewards.append(cur_reward)
                states.append(self.env.current_state())
        return states, rewards


    def compute_features_from_states(self, states):
        features = []
        for s in states[:-1]:   # discarding final state
            lin_comb = (s[0], s[1], s[0]**2, s[1]**2, s[0]*s[1])
            features.append(np.array([(f - mu) / sigma for f, mu, sigma in zip(lin_comb, means, stds)]))
        return features


    def update_state_value_function(self, state_features, rewards, t):
        return_value = 0
        for s,r in zip(reversed(state_features), reversed(rewards)):
            return_value = r + self.discount * return_value
            self.theta = self.theta + 1/t * (return_value - np.inner(self.theta,s)) * s


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
            states, rewards = self.perform_episode(**kwargs)
            state_features = self.compute_features_from_states(states)
            self.update_state_value_function(state_features, rewards, t+1)
            self.update_policy()
            self.update_explore_threshold(t+1)
            if t%100 == 0:
                print(self.theta)
        grid_world.print_policy(self.policy, self.env)




if __name__ == "__main__":

    optimizer = PolicyOptimizer(environment=grid_world.negative_grid(), learning_scheme='loglog')
    optimizer.improve_policy()
    # optimizer.improve_policy(wind='right', wind_force=0.2)
