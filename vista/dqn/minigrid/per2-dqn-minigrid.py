import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper,RGBImgPartialObsWrapper,RGBImgObsWrapper
import collections
import random
import numpy as np
import sys, os
import csv
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
from tree import SumTree

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 1e-4
gamma         = 0.99
buffer_limit  = 50_000
batch_size    = 32

if torch.cuda.is_available():
    device= 'cuda:0'
else:
    device = 'cpu'
print(f'device:{device}')

class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size,device):
        # state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, state_size, 1, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, state_size, 1, dtype=torch.float)
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
            self.state[sample_idxs].permute(0,3,1,2).to(self.device)/255.,
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].permute(0,3,1,2).to(self.device)/255.,
            self.done[sample_idxs].to(self.device)
        )
        return batch


class Qnet(nn.Module):
    def __init__(self, n_input_channels, n_actions):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.mlp1 = nn.Linear(64, 256)
        self.mlp2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.mlp1(x))
        x = self.mlp2(x)
        return x

    def sample_action(self, obs, epsilon):
        obs = (torch.tensor(obs).permute(2,0,1) / 255.).unsqueeze(0).to(device)
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,2)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    s,a,r,s_prime,done_mask = memory.sample(batch_size)

    # print(f"s shape: {s.shape}, type: {s.type()}")
    # print(f"a shape: {a.shape}, type: {a.type()}")
    # print(f"r shape: {r.shape}, type: {r.type()}")
    # print(f"s_prime shape: {s_prime.shape}, type: {s_prime.type()}")
    # print(f"done mask shape: {done_mask.shape}, type: {done_mask.type()}")

    # Rt+1 + γ max_a Q(S_t+1, a; θt). where θ=θ- because we update target params to train params every t steps
    Q_next = q_target(s_prime).max(1)[0] # .unsqueeze(1)
    y_i = r + done_mask * gamma * Q_next

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # Q = q(s).gather(1,a)
    Q = q(s)[torch.arange(len(a)), a.to(torch.long).flatten()]

    assert Q.shape == y_i.shape, f"{Q.shape}, {y_i.shape}"

    loss = F.smooth_l1_loss(Q, y_i)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
    return Q.mean(), loss

# Convert image to greyscale, resize and normalise pixels
def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=2)
    return image

def main():
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    env = RGBImgObsWrapper(env) # Get pixel observations

    # set seed for reproducibility
    seed = int(sys.argv[1])
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    q = Qnet(1,3)
    q_target = Qnet(1,3)
    q_target.load_state_dict(q.state_dict())
    q.to(device)
    q_target.to(device)
    memory = ReplayBuffer(40, 1, buffer_limit, device)

    total_frames = 200  # Total number of frames for annealing
    print_interval = 1
    train_update_interval = 4
    target_update_interval = 5_000
    train_start = 1_000
    score = 0
    step = 0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    q_value = torch.tensor(0)
    loss = torch.tensor(0)

    with open(f'../data/minigrid/PER-DQN-Minigrid{sys.argv[1]}.csv', 'w', newline='') as csvfile:
        fieldnames = ['step', 'score', 'Q-value', 'loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for n_epi in range(500):
            observation, info = env.reset()

            # preprocess
            observation = preprocess(observation['image'])

            terminated = False
            truncated = False
            while not terminated and not truncated:
                epsilon = max(0.1, 1 - 0.01*(step/total_frames)) #Linear annealing from 1.0 to 0.1
                action = q.sample_action(observation, epsilon)      
                observation_prime, reward, terminated, truncated, info = env.step(action)
                done_mask = 0.0 if terminated else 1.0
                observation_prime = preprocess(observation_prime['image'])
                memory.add((observation,action,reward,observation_prime, int(done_mask)))
                
                observation = observation_prime

                if step>train_start and step%train_update_interval==0:
                    q_value, loss = train(q, q_target, memory, optimizer)
    
                if step>train_start and step%target_update_interval==0:
                    q_target.load_state_dict(q.state_dict())
                    
                score += reward
                step += 1
                if terminated:
                    break

            if n_epi%print_interval==0:
                print(f"episode :{n_epi}, step: {step}, score : {score/print_interval:.1f}, n_buffer : {memory.real_size}, eps : {epsilon*100:.1f}%")
                writer.writerow({"step": step, "score":score/print_interval, "Q-value":q_value.item(), "loss":loss.item()})
                csvfile.flush()
                score = 0.0

    env.close()

if __name__ == '__main__':
    main()