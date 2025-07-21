from collections import defaultdict
import numpy as np
import random

device = 'mps'

class TicTacToe():
    def __init__(self):
        self.board = [" "] * 9

    def show_board(self):
        print(f" {self.board[0]} | {self.board[1]} | {self.board[2]} ", end='\n')
        print("---+---+---")
        print(f" {self.board[3]} | {self.board[4]} | {self.board[5]} ", end='\n')
        print("---+---+---")
        print(f" {self.board[6]} | {self.board[7]} | {self.board[8]} ", end='\n')

    def clear(self):
        self.board = [" "] * 9

    def move(self, position, player):
        if self.board[position] == " ":
            self.board[position] = player
            return True
        return False
    
    def isWon(self):
        for i in range(3):
            if self.board[3*i] != " " and self.board[3*i] == self.board[3*i+1] and self.board[3*i] == self.board[3*i+2]:
                return self.board[3*i]
            if self.board[i] != " " and self.board[i] == self.board[i+3] and self.board[i] == self.board[i+6]:
                return self.board[i]
        
        if self.board[0] != " " and self.board[0] == self.board[4] and self.board[0] == self.board[8]:
            return self.board[0]
        if self.board[2] != " " and self.board[2] == self.board[4] and self.board[2] == self.board[6]:
            return self.board[2]
        return False
    
    def possibleMove(self):
        return [i for i in range(9) if self.board[i] == " "]
    
    def get_board(self):
        return self.board.copy()
    
    def draw(self):
        if self.isWon():
            return False
        return ' ' not in self.board

class Agent():
    def __init__(self):
        self.policy = defaultdict(lambda : random.choice([0,1,2,3,4,5,6,7,8]))
        self.state_action_values = defaultdict(float)
        self.gamma = 1.0
        self.alpha = 0.01
        self.epsilon = 1.0
        self.epsilon_decay_rate = 0.99
        self.min_epsilon = 0.15

    def move(self, state:TicTacToe, play_FR = False):
        if random.random() < self.epsilon and not play_FR:
            pos_moves = state.possibleMove()
            action = random.choice(pos_moves)
        else:
            action = self.policy[self.state_to_str(state.get_board())]
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.min_epsilon)
        return action
    
    def TD(self, prev_state, action, reward, next_state):
        prev_string = self.state_to_str(prev_state)
        if next_state is None:
            future_reward = 0
        else:
            next_string = self.state_to_str(next_state)
            pos_moves = [i for i in range(9) if next_state[i] == " "]
            future_reward = max(self.state_action_values[(next_string, a)] for a in pos_moves)
        self.state_action_values[(prev_string, action)] = self.state_action_values[(prev_string, action)] + self.alpha * (reward + self.gamma * future_reward - self.state_action_values[(prev_string, action)])
        valid_moves = [i for i in range(9) if prev_state[i] == " "]
        self.policy[prev_string] = max(valid_moves, key = lambda a: self.state_action_values[(prev_string,a)])

    def state_to_str(self,state):
        return ''.join(state)
    
def train_agent():
    agent_wins = 0
    opp_wins = 0
    agent = Agent()
    agentO = Agent()
    for i in range(1_000_000):
        board = TicTacToe()
        prev_state_O = None
        action_O = None
        next_state_O = None
        while True:
            prev_state = board.get_board()
            action = agent.move(board)
            board.move(action, "X")
            next_state = board.get_board()

            if board.isWon() == "X":
                agent.TD(prev_state, action, 1.0, None)
                agentO.TD(prev_state_O, action_O, -1.0, None)
                agent_wins += 1
                break
            elif board.draw():
                agent.TD(prev_state,action, 0.5, None)
                agentO.TD(prev_state_O, action_O, 0.5, None)
                break
            elif(next_state == prev_state):
                agent.TD(prev_state, action, -20.0, next_state)
            
            if prev_state_O is not None and action_O is not None and next_state_O is not None:
                agentO.TD(prev_state_O, action_O,-0.1, next_state_O)
            
            prev_state_O = board.get_board()
            action_O = agentO.move(board)
            board.move(action_O, "O")
            next_state_O = board.get_board()

            if board.isWon() == "O":
                agent.TD(prev_state, action, -1.0, None)
                agentO.TD(prev_state_O, action_O, 1.0, None)
                opp_wins += 1
                break
            elif board.draw():
                agent.TD(prev_state,action, 0.5, None)
                agentO.TD(prev_state_O, action_O, 0.5, None)
                break
            elif(next_state_O == prev_state_O):
                agentO.TD(prev_state_O, action_O, -20.0, next_state_O)
            agent.TD(prev_state, action, -0.1, next_state)
    return agent

def stats():
    agent = train_agent()
    agent.epsilon = 0
    num_wins = 0
    num_draws = 0
    num_losses = 0
    for _ in range(1_000_000):
        board = TicTacToe()
        while True:
            board.move(agent.move(board, True), "X")
            if board.isWon() == "X":
                num_wins += 1
                break
            elif board.draw():
                num_draws += 1
                break
            pos_moves = board.possibleMove()
            if pos_moves is None:
                num_draws += 1
                break
            board.move(random.choice(pos_moves), "O")
            
            if board.isWon() == "O":
                num_losses += 1
                break
            if board.draw():
                num_draws += 1
                break
    print(num_wins * 100/1000000)
    print(num_draws * 100/1000000)
    print(num_losses * 100/1000000)

stats()