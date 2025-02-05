import torch
from torch import nn
from collections import deque
import random

class TicTacToe():
    def __init__(self):
        self.board = [" "] * 9
        self.actual_board = [-1] * 9

    def show_board(self):
        print(f" {self.board[0]} | {self.board[1]} | {self.board[2]} ", end='\n')
        print("---+---+---")
        print(f" {self.board[3]} | {self.board[4]} | {self.board[5]} ", end='\n')
        print("---+---+---")
        print(f" {self.board[6]} | {self.board[7]} | {self.board[8]} ", end='\n')

    def clear(self):
        self.board = [" "] * 9
        self.actual_board = [-1] * 9

    def move(self, position, player):
        if self.board[position] == " ":
            self.board[position] = player
            if(player == "O"):
                self.acutal_board[position] = 0
            else:
                self.actual_board[position] = 1
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
        return self.actual_board.copy()
    
    def draw(self):
        if self.isWon():
            return False
        for i in range(9):
            if self.board[i] == " ":
                return False
        return True   

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.l1 = nn.Linear(9, 16)
        self.l2 = nn.Linear(16,16)
        self.l3 = nn.Linear(16,9)

    def forward(self, x):
        x = nn.ReLU()(self.l1(x))
        x = nn.ReLU()(self.l2(x))
        return self.l3(x)
    
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

EPSILON_START = 0.99
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TAU = 0.05
GAMMA = 0.9
BATCH_SIZE = 1
LR = 0.0001

policyNet = NN()
targetNet = NN()
targetNet.load_state_dict(policyNet.state_dict())
optimizer = torch.optim.AdamW(policyNet.parameters(), lr = LR, amsgrad= True)
loss_fn = nn.SmoothL1Loss()

memory = ReplayMemory(10000)


def get_action(state : TicTacToe):
    global EPSILON_DECAY
    global EPSILON_END
    global EPSILON_START
    if random.random() < max(EPSILON_START, EPSILON_END):
        choice = random.choice(state.possibleMove())
        EPSILON_END = EPSILON_END * EPSILON_DECAY
        return choice
    with torch.no_grad():
        result = policyNet(state.get_board())
        _, index = torch.max(result, dim = 1)
        return index

def train_model():
    if len(memory) < BATCH_SIZE:
        return
    batch = memory.sample(BATCH_SIZE)
    state, actions, next_state, rewards = zip(*batch)
    non_terminal_states_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state)))
    non_terminal_states = [s for s in next_state if s is not None]

    state_action_values = policyNet(torch.tensor(state, dtype = torch.float32)).gather(1, actions)

    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_terminal_states_mask] = targetNet(torch.tensor(non_terminal_states, dtype = torch.float32)).max(1)[0]

    expected_state_action_values = rewards + GAMMA * next_state_values

    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(0))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for num in range(100):
    game = TicTacToe()
    while not game.isWon() and not game.draw():
        state = game.get_board()
        action = get_action(game)
        game.move(action, "X")
        next_state = game.get_board()
        if(game.isWon()):
            memory.push(state,action,None,10)
        elif(game.draw()):
            memory.push(state,action,None, 1)
        elif(state == game.get_board()):
            memory.push(state,action,next_state, -3)
        else:
            memory.push(state, action, next_state, 0)
        
        move = random.choice(game.get_board())
        
        game.move(action, "O")

        if(game.isWon()):
            memory.push(state,action,next_state,-10)
        elif(game.draw()):
            memory.push(state,action,next_state,1)
    if game.draw():
        print(f"Game {num+1} : Draw")
    elif game.isWon() == "X":
        print(f"Game {num+1}: AI Wins!!!")
    else:
        print(f"Game {num+1}: AI Loses")

    train_model()
    
    target_net_state_dict = targetNet.state_dict()
    policy_net_state_dict = policyNet.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
    targetNet.load_state_dict(target_net_state_dict)

torch.save(policyNet.state_dict(), "policyNet.pt")
torch.save(targetNet.state_dict(), "targetNet.pt")

        


        
        


