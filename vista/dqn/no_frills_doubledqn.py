import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from tree import SumTree
from utils import set_seed
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, device):
        # state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        self.device = device

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size

        sample_idxs = np.random.choice(self.real_size, batch_size, replace=False)

        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device)
        )
        return batch

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.silu(self.layer1(x))
        x = F.silu(self.layer2(x))
        return self.layer3(x)

def main():
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    steps_done = 0

    with open('no_frill_doubledqn.csv', 'w', newline='') as csvfile:
        fieldnames = ['episode', 'duration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        env = gym.make("CartPole-v1")    
        torch.manual_seed(0)
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
        # Initialize replay memory D to capacity N
        N = 50_000
        replay_mem = ReplayBuffer(4, 1, N, device)
        
        # Initialize action-value function Q with random weights
        policy_net = DQN(4, 2).to(device)
        target_net = DQN(4, 2).to(device)
        
        # target_net = DQN(4, 2).to(device)
        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        
        # For episode 1 --> M
        for i in range(5_000):
            
            # Initialize the environment and get it's state. Do any preprocessing here
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in range(500):
                
                # With probability epislon, select random action a_t
                sample = random.random()
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
                if sample <= eps_threshold:
                    a_t = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
                # otherwise select a_t = max_a Q*(state, action; theta)
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                else:
                    with torch.no_grad():
                        a_t = policy_net(state).max(1)[1].view(1, 1) 
    
                # execute action, a_t, in emulator aka env
                observation, reward, terminated, truncated, _ = env.step(a_t.item())
                
                # set s_{t+1} = s_t, a_t, x_{t+1}
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated
                next_state = observation 
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                
                # store transition (s_t, a_t, r_t, s_{t+1}) in D
                replay_mem.add((state, a_t, reward, next_state, int(done)))
    
                # sample minibatch (s_j, a_j, r_j, s_{j+1}) (b=64) of transitions from D
                if replay_mem.real_size > BATCH_SIZE:
                    state_b, action_b, reward_b, next_state_b, done_b = replay_mem.sample(BATCH_SIZE)
                    
                    # Q(s′,argmax a ′Q(s ′,a ′;θ i);θ i−)
                    Q_next = target_net(next_state_b).max(1)[0]
                    y_i = reward + (1-done_b) * GAMMA * Q_next

                    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                    Q = policy_net(state_b)[torch.arange(len(action_b)), action_b.to(torch.long).flatten()]
                    
                    loss = torch.mean((y_i - Q)**2)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    with torch.no_grad():
                        # Soft update of the target network's weights
                        # θ′ ← τ θ + (1 −τ )θ′
                        for tp, sp in zip(target_net.parameters(), policy_net.parameters()):
                            tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)

                state = next_state

                # increment global step counter for 
                steps_done += 1
                
                if done:
                    writer.writerow({"episode": i, "duration":t})
                    csvfile.flush()
                    break

if __name__=="__main__":
    main()