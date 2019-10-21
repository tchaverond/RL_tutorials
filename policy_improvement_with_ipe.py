# -*- coding: utf-8 -*-


import numpy as np

import grid_world
from iterative_policy_evaluation_try_again import print_values, print_policy

np.random.seed(42)



class PolicyOptimizer:

    def __init__(self, environment, discount_factor=0.9, epsilon=1e-4):

        self.env = environment
        self.discount = discount_factor
        self.epsilon = epsilon
        self.action_states = [s for s in self.env.all_states() if not self.env.is_terminal(s)]
        self.terminal_states = [s for s in self.env.all_states() if self.env.is_terminal(s)]
        self.state_values = None
        self.initialize_state_value_function()
        # starting with 'always-up' policy
        self.policy = {k: 'U' for k in self.action_states}


    def initialize_state_value_function(self):

        self.state_values = {s:0 for s in self.env.all_states()}


    def update_state_value_function(self, policy):

        deltas = [1]
        while max(deltas) > self.epsilon:
            deltas = [None] * len(self.action_states)
            for i, s in enumerate(self.action_states):
                self.env.set_state(s)
                old_value = self.state_values[s]
                self.state_values[s] = 0
                if policy == 'random':
                    p_a = 1 / len(self.env.actions[s])
                    for a in self.env.actions[s]:
                        reward = self.env.move(a)
                        next_state = self.env.current_state()
                        self.state_values[s] += p_a * (reward + self.discount * self.state_values[next_state])
                        self.env.undo_move(a)
                elif policy == 'self':
                    reward = self.env.move(self.policy[s])
                    next_state = self.env.current_state()
                    self.state_values[s] = reward + self.discount * self.state_values[next_state]
                    self.env.undo_move(self.policy[s])
                deltas[i] = np.abs(self.state_values[s] - old_value)


    def improve_policy(self):

        policy_changed = True
        i = 0
        while policy_changed:
            policy_changed = False
            for s in self.action_states:
                self.env.set_state(s)
                best_score = -1
                for a in self.env.actions[s]:
                    reward = self.env.move(a)
                    next_score = reward + self.discount * self.state_values[self.env.current_state()]
                    if next_score > best_score:
                        best_action = a
                        best_score = next_score
                    self.env.undo_move(a)
                if best_action != self.policy[s]:
                    self.policy[s] = best_action
                    policy_changed = True
            self.update_state_value_function(policy='self')

            i += 1
            if i%1 == 0:
                print("Number of iterations: ", i)
                print_policy(self.policy, self.env)
                print_values(self.state_values, self.env)



if __name__ == "__main__":

    optimizer = PolicyOptimizer(environment=grid_world.negative_grid())
    print_policy(optimizer.policy, optimizer.env)
    print_values(optimizer.state_values, optimizer.env)
    optimizer.improve_policy()

