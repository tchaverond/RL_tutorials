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


    def update_state_value_function(self, policy, wind, wind_force=0.5):

        deltas = [1]
        while max(deltas) > self.epsilon:
            deltas = [None] * len(self.action_states)
            for i, s in enumerate(self.action_states):
                self.env.set_state(s)
                old_value = self.state_values[s]
                self.state_values[s] = 0
                for a in self.env.actions[s]:
                    if policy == 'random':
                        p_a = 1 / len(self.env.actions[s])
                    elif policy == 'self':
                        if wind == 'random':
                            if self.policy[s] == a:
                                p_a = wind_force
                            else:
                                p_a = wind_force * (1 / (len(self.env.actions[s]))-1)
                        elif wind == 'right':
                            if a == 'R' and a in self.env.actions[s]:
                                p_a = wind_force
                            else:
                                p_a = 0
                            if self.policy[s] == a:
                                p_a += (1 - wind_force)
                        else:
                            p_a = int(self.policy[s] == a)
                    if p_a == 0 :
                        continue
                    reward = self.env.move(a)
                    next_state = self.env.current_state()
                    self.state_values[s] += p_a * (reward + self.discount * self.state_values[next_state])
                    self.env.undo_move(a)
                deltas[i] = np.abs(self.state_values[s] - old_value)


    def improve_policy(self, **kwargs):

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
            self.update_state_value_function(policy='self', **kwargs)

            i += 1
            if i%1 == 0:
                print("Number of iterations: ", i)
                print_policy(self.policy, self.env)
                print_values(self.state_values, self.env)



if __name__ == "__main__":

    optimizer = PolicyOptimizer(environment=grid_world.negative_grid())
    print_policy(optimizer.policy, optimizer.env)
    print_values(optimizer.state_values, optimizer.env)
    optimizer.improve_policy(wind=None)

    # Windy Gridworld: each action has a 50% chance to fail, another action (chosen at random) is performed instead
    optimizer = PolicyOptimizer(environment=grid_world.negative_grid())
    print_policy(optimizer.policy, optimizer.env)
    print_values(optimizer.state_values, optimizer.env)
    optimizer.improve_policy(wind='right', wind_force=0.26)   # .25 is the threshold to switch optimal agency

