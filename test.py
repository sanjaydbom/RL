import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import torch
from torch import nn
import numpy as np
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, (8,8), 4)
        self.conv2 = nn.Conv2d(32, 64, (4,4), 2)
        self.conv3 = nn.Conv2d(64, 64, (3,3), 1)
        self.fc1 = nn.Linear(3136,256)
        self.output = nn.Linear(256,4)

    def forward(self,x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.output(x)
        return x

device = "mps"

agent = Agent().to(device=device)
agent.load_state_dict(torch.load("/Users/sanjay/Desktop/Python Projects/AI/RL/AtariAgent.pt", weights_only=True))

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", frameskip = 1)
env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False, screen_size=84)
env = FrameStackObservation(env, stack_size = 4)

def select_action(state):
    with torch.no_grad():
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
            state = state.to(device=device)
            action_values = agent(state)
            return torch.argmax(action_values)
        
tot_reward = 0
num_tests = 100
for i in range(num_tests):
    state, _ = env.reset()
    action = 1
    while True:
        next_state, reward, terminated, truncated, _ = env.step(action)
        tot_reward += reward
        if terminated or truncated:
            break
        state = next_state
        action = select_action(state)
print(tot_reward / num_tests)
