# -*- coding: utf-8 -*-


import copy
import numpy as np
from tqdm import tqdm

import grid_world

np.random.seed(42)


class Agent:

    def __init__(self, environment, policy='random', discount_factor=0.9):

        self.env = environment
        self.state_values = {s:0 for s in environment.all_states()}
        self.possible_actions = ['U', 'D', 'R', 'L']
        self.prechosen_actions = None
        self.set_policy(policy)
        self.discount_factor = discount_factor


    def set_policy(self, policy):

        self.policy = policy
        if policy == 'win-from-start':
            self.prechosen_actions = {(0, 0): 'R', (0, 1): 'R', (0, 2): 'R', (1, 0): 'U',
                                      (1, 2): 'R', (2, 0): 'U', (2, 1): 'R', (2, 2): 'R',
                                      (2, 3): 'U'}


    def choose_action(self):

        if self.policy == 'random':
            return np.random.choice(self.possible_actions)
        if self.policy == 'win-from-start':
            return self.prechosen_actions[self.env.current_state()]


    def play_game(self):

        self.env.set_state(list(self.env.actions.keys())[np.random.choice(len(self.env.actions.keys()))])
        states = [self.env.current_state()]
        actions = []
        rewards = []

        while not self.env.game_over():
            while states[-1] == self.env.current_state():   # if our chosen action isn't possible, don't update state
                action = self.choose_action()
                reward = self.env.move(action)
            actions.append(action)
            rewards.append(reward)
            states.append(self.env.current_state())
        return states, actions, rewards


    def update_state_value_function(self, states, rewards, t, epsilon=10):

        deltas = [1]
        # while max(deltas) > epsilon:
        deltas = [None] * (len(states) - 1)
        for i, s, s_prime, r in zip(range(len(states) - 1),
                reversed(states[:-1]), reversed(states[1:]), reversed(rewards)):
            new_value = self.state_values[s] + 1/(np.log(t+2)) * \
                        (r + self.discount_factor * self.state_values[s_prime] - self.state_values[s])
            deltas[i] = np.abs(new_value - self.state_values[s])
            self.state_values[s] = new_value


    def solve_prediction_problem(self, max_iter=10000):

        state_values = {}
        for t in tqdm(range(max_iter)):
            states, actions, rewards = self.play_game()
            self.update_state_value_function(states, rewards, t)
            if t%1000 == 0:
                state_values[t] = copy.deepcopy(self.state_values)
        return state_values



if __name__ == "__main__":

    a = Agent(grid_world.standard_grid(), policy='random', discount_factor=1.0)
    state_values = a.solve_prediction_problem()
    for k, v in state_values.items():
        print(k)
        grid_world.print_values(v, a.env)
    a = Agent(grid_world.standard_grid(), policy='win-from-start')
    state_values = a.solve_prediction_problem()
    for k,v in state_values.items():
        print(k)
        grid_world.print_values(v, a.env)

