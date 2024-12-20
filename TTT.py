import torch
from torch import nn
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
        for i in range(9):
            if self.board[i] == " ":
                return False
        return True            

class Agent():
    def __init__(self):
        self.Q_Table = defaultdict(lambda: np.zeros(9)-1.0)
        self.epsilon = 1.0
        self.epsilon_decay = 0.997
        self.min_epsilon = 0.01
        self.gamma = 0.99
        self.alpha = 0.05

    def move(self, state: TicTacToe, playFR = False):
        if random.uniform(0,1) < self.epsilon and not playFR:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            temp = state.possibleMove()
            if not temp:
                return None
            return random.choice(state.possibleMove())
        
        pos_str = self.state_to_str(state.get_board())
        q_values = self.Q_Table[pos_str]
        possible_values = [(action, q_values[action]) for action in state.possibleMove()]
        return max(possible_values, key = lambda x: x[1])[0]
        
    def state_to_str(self,state):
        return ''.join(state)
    
    def update_q_table(self, prev_state, next_state, action, reward, gameOver):
        prev_state_string = self.state_to_str(prev_state)
        next_state_string = self.state_to_str(next_state)
        prev_q = self.Q_Table[prev_state_string][action]
        next_q = self.Q_Table[next_state_string]
        if gameOver:
            future_reward = 0

        else:
            possible_moves = [i for i, cell in enumerate(next_state) if cell == " "]
            if not possible_moves:
                future_reward = 0
            else:
                future_reward = max(next_q[move] for move in possible_moves)
        self.Q_Table[prev_state_string][action] = prev_q + self.alpha * (reward + self.gamma * future_reward - prev_q)



def train_agent():
    t = TicTacToe()
    agentX = Agent()
    agentO = Agent()
    for _ in range(100000):
        t.clear()
        while True:
            prev_stateX = t.get_board()
            moveX = agentX.move(t)
            t.move(moveX, "X")

            if t.isWon() == "X":
                agentX.update_q_table(prev_stateX, t.get_board(), moveX, 2.0, True)
                agentO.update_q_table(prev_stateO, prev_stateX, moveO, -3.0, True)
                break
            elif t.draw():
                agentX.update_q_table(prev_stateX, t.get_board(), moveX, 0.5, True)
                agentO.update_q_table(prev_stateO, prev_stateX, moveO, 0.5, True)
                break
            else:
                agentX.update_q_table(prev_stateX, t.get_board(), moveX, -0.05, False)

            prev_stateO = t.get_board()
            moveO = agentO.move(t)
            t.move(moveO, "O")


            if t.isWon() == "O":
                agentO.update_q_table(prev_stateO, t.get_board(), moveO, 2.0, True)
                agentX.update_q_table(prev_stateX, prev_stateO, moveX, -3.0, True)
                break
            elif t.draw():
                agentO.update_q_table(prev_stateO, t.get_board(), moveO, 0.5, True)
                agentX.update_q_table(prev_stateX, prev_stateO, moveX, 0.5, True)
                break
            else:
                agentO.update_q_table(prev_stateO, t.get_board(), moveO, -0.05, False)
    return agentX

def get_stats():
    agent = train_agent()
    t=TicTacToe()
    won = 0
    draw = 0
    loss = 0
    for _ in range(100000):
        t.clear()
        while not t.isWon() and not t.draw():
            t.move(agent.move(t, True), "X")
            if(t.isWon() or t.draw()):
                break
            possibleMoves = t.possibleMove()
            move = random.choice(possibleMoves)
            t.move(move, "O")
        if t.draw():
            draw += 1
        elif t.isWon() == "X":
            won += 1
        else:
            loss += 1
    print(f"Win rate: {won/1000}%\nDraw rate: {draw/1000}%\nLoss rate: {loss/1000}%")

get_stats()