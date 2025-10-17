import numpy as np
from environment import PageReplacementEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import torch

page_sequence = np.random.randint(0, 10, size=50)
num_frames = 3

env = PageReplacementEnv(num_frames, page_sequence)
agent = DQNAgent(state_size=num_frames, action_size=num_frames)

episodes = 20
batch_size = 32
history = []

for e in range(episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        agent.replay(batch_size)
    history.append(env.page_faults)
    print(f"Episode {e+1}/{episodes}, Page Faults: {env.page_faults}")

torch.save(agent.model.state_dict(), '../models/dqn_model.pth')

plt.plot(history)
plt.xlabel('Episode')
plt.ylabel('Page Faults')
plt.title('RL-based Page Replacement Learning Curve')
plt.show()