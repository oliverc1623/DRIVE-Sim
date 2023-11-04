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

class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size):
        # state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

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
            self.state[sample_idxs].to(device),
            self.action[sample_idxs].to(device),
            self.reward[sample_idxs].to(device),
            self.next_state[sample_idxs].to(device),
            self.done[sample_idxs].to(device)
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

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(step_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if memory.real_size < BATCH_SIZE:
        return
    weight = None
    batch = memory.sample(BATCH_SIZE)
    state, action, reward, next_state, done = batch

    Q_next = target_net(next_state).max(dim=1).values
    Q_target = reward + GAMMA * (1 - done) * Q_next
    Q = policy_net(state)
    Q = Q[torch.arange(len(action)), action.to(torch.long).flatten()]

    assert Q.shape == Q_target.shape, f"{Q.shape}, {Q_target.shape}"

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(Q, Q_target)
    
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    with torch.no_grad():
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        for tp, sp in zip(target_net.parameters(), policy_net.parameters()):
            tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)

    return loss.item()

def train(writer, trial_i, csvfile):
    timesteps = 50_000
    episodes = 0
    done = False
    steps = 0
    step_durations.append(steps)
    state, _ = env.reset()
    
    for i in range(1, timesteps+1):
        if done:
            done = False
            state, _ = env.reset()
            episodes += 1
            step_durations.append(steps)
            writer.writerow({"episode": episodes, "duration":steps, "trial": trial_i})
            csvfile.flush()
            steps = 0

        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = select_action(state).item()
        observation, reward, terminated, truncated, _ = env.step(action)
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        next_state = observation 

        # Store the transition in memory
        memory.add((state, action, reward, next_state, int(done)))
        
        # Move to the next state
        state = next_state
        
        # Perform one step of the optimization (on the policy network)
        optimize_model()
        steps += 1
    print("Finished timesteps")


if __name__=="__main__":
    env = gym.make("CartPole-v1")    
    # torch.manual_seed(0)
    # if GPU is to be used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    d = []
    with open('mem_dqn_carpolev1_2.csv', 'w', newline='') as csvfile:
        fieldnames = ['episode', 'duration', 'trial']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(4):
            policy_net = DQN(n_observations, n_actions).to(device)
            target_net = DQN(n_observations, n_actions).to(device)
            target_net.load_state_dict(policy_net.state_dict())
            
            optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
            memory = ReplayBuffer(n_observations, 1, 50_000)
            
            steps_done = 0
            step_durations = []
        
            print(f"Training trial: {i}")
            set_seed(env, i)
            train(writer, i, csvfile)
    print("Training complete")
