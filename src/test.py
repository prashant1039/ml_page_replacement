import torch
import numpy as np
from environment import PageReplacementEnv
from dqn_agent import DQNAgent

page_sequence = np.random.randint(0, 10, size=50)
num_frames = 3

env = PageReplacementEnv(num_frames, page_sequence)
agent = DQNAgent(state_size=num_frames, action_size=num_frames)
agent.model.load_state_dict(torch.load('../models/dqn_model.pth'))

state = env.reset()
done = False
while not done:
    action = agent.act(state)
    state, reward, done = env.step(action)

print(f"Total Page Faults (Test): {env.page_faults}")