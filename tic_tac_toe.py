# -*- coding: utf-8 -*-


import numpy as np
from tqdm import tqdm

np.random.seed(42)



def pick_random(choices):
    return np.random.choice(choices)


class Environment:

    def __init__(self):
        self.board = [0]*9   # player 1 uses 1, player 2 uses -1
        self.winning_positions = ((0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6))
        self.winner = 0   # 0: game still being played, 1: player 1 wins, 2: player 2 wins, -1: draw

    def draw_board(self):
        def convert_to_piece(x):
            if x == 1: return 'x'
            if x == -1: return 'o'
            return ' '
        board = '|' + '|'.join([convert_to_piece(value) + '\n' if (pos+1)%3 == 0 else convert_to_piece(value)
                                for pos,value in enumerate(self.board)]) + '\n --- \n'
        print(board)

    def get_state(self):
        return ''.join(['2' if i==-1 else str(i) for i in self.board])

    def get_possible_states(self, player):
        new_positions = [pos for pos in range(len(self.board)) if self.board[pos] == 0]
        states = []
        for pos in new_positions:
            self.board[pos] = 1 if player == 1 else -1
            states.append(self.get_state())
            self.board[pos] = 0
        return states

    def perform_action(self, resulting_state):
        self.board = [-1 if pos=='2' else int(pos) for pos in resulting_state]

    def check_winner(self):
        for w in self.winning_positions:
            if sum([self.board[pos] for pos in w]) == 3:
                self.winner = 1
                return
            elif sum([self.board[pos] for pos in w]) == -3:
                self.winner = 2
                return
        if 0 not in self.board:
            self.winner = -1

    def game_over(self):
        self.check_winner()
        return self.winner != 0



class Agent:

    def __init__(self, id, epsilon, learning_rate):
        self.id = id
        self.state_history = []
        self.state_scores = {}
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def take_action(self, env):
        next_states = env.get_possible_states(self.id)
        for state in next_states:
            if state not in self.state_scores.keys():
                self.state_scores[state] = 0.5
        explore = np.random.random() < self.epsilon
        if explore:
            action = pick_random(next_states)
        else:
            action = self.pick_best(next_states)
        env.perform_action(action)

    def pick_best(self, choices):
        values = {k:v for k,v in self.state_scores.items() if k in choices}
        return max(values, key=values.get)

    def identify_winning_state(self, state):
        self.state_scores[state] = 1

    def identify_losing_state(self, state):
        self.state_scores[state] = 0

    def identify_draw(self, state):
        self.state_scores[state] = 0.2

    def update_state_history(self, state):
        self.state_history.append(state)
        if state not in self.state_scores.keys():   # state we've never seen before
            self.state_scores[state] = 0

    def update(self):
        for state, next_state in zip(reversed(self.state_history[:-1]), reversed(self.state_history[1:])):
            self.state_scores[state] += self.learning_rate * (self.state_scores[next_state] - self.state_scores[state])
        self.state_history = []




def play_game(ag1, ag2, env, draw=False):

    current_player = None
    while not env.game_over():

        current_player = ag2 if current_player == ag1 else ag1
        if draw:
            env.draw_board()
        current_player.take_action(env)
        state = env.get_state()
        ag1.update_state_history(state)
        ag2.update_state_history(state)

    if draw:
        env.draw_board()
    # print(env.winner)
    if ag1.id == env.winner:
        ag1.identify_winning_state(state)
        ag2.identify_losing_state(state)
    elif ag2.id == env.winner:
        ag1.identify_losing_state(state)
        ag2.identify_winning_state(state)
    else:   # draw
        ag1.identify_draw(state)
        ag2.identify_draw(state)
    ag1.update()
    ag2.update()



if __name__ == "__main__":

    # TODO: optimize computation time (from ~8' for 100k episodes)
    ag1 = Agent(id=1, epsilon=.1, learning_rate=.1)
    ag2 = Agent(id=2, epsilon=.1, learning_rate=.1)
    for _ in tqdm(range(100000)):
        env = Environment()
        play_game(ag1, ag2, env, draw=False)
        # print(ag1.state_scores)
        # print(ag2.state_scores)
    env = Environment()
    play_game(ag1, ag2, env, draw=True)
