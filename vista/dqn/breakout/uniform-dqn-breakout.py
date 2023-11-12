import gymnasium as gym
import collections
import random
import numpy as np
import sys, os
import csv
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 1e-3
gamma         = 0.99
buffer_limit  = 50_000
batch_size    = 32

class ReplayBuffer:
    def __init__(self, state_size, state_size2, action_size, buffer_size, device):
        # state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, 1, state_size, state_size2, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, 1, state_size, state_size2, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        self.device = device

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(state).unsqueeze(0)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state).unsqueeze(0)
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

class Qnet(nn.Module):
    def __init__(self, n_input_channels, n_actions):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        # 22 and 16 come from
        # 22 =~ (((((210 - 8 + 1)/4) - 4 + 1) / 2) - 3 + 1) / 1 
        # 16 =~ (((((160 - 8 + 1)/4) - 4 + 1) / 2) - 3 + 1) / 1
        self.mlp = nn.Linear(17024, n_actions)

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))
        # Flatten before fully connected layers
        x = x.view(x.size(0), -1)
        x = self.mlp(x)    
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,3)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    s,a,r,s_prime,done_mask = memory.sample(batch_size)

    # Q(s′,argmax a ′Q(s ′,a ′;θ i);θ i−)
    Q_next = q_target(s_prime).max(1)[0]
    y_i = r + done_mask * gamma * Q_next

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    Q = q(s)[torch.arange(len(a)), a.to(torch.long).flatten()]

    assert Q.shape == y_i.shape, f"{Q.shape}, {y_i.shape}"

    loss = F.smooth_l1_loss(Q, y_i)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Convert image to greyscale, resize and normalise pixels
def preprocess(image, width, height, targetWidth, targetHeight):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = image[20:300, 0:200]  # crop off score
	image = cv2.resize(image, (targetWidth, targetHeight))
	image = image.reshape(targetWidth, targetHeight)
    
	return image

def main():
    env = gym.make('BreakoutNoFrameskip-v4')
    if torch.cuda.is_available():
        device= 'cuda:0'
    else:
        device = 'cpu'
    print(f'device:{device}')
        
    # set seed for reproducibility
    seed = int(sys.argv[1])
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    q = Qnet(1,4)
    q_target = Qnet(1,4)
    q_target.load_state_dict(q.state_dict())
    q.to(device)
    q_target.to(device)
    memory = ReplayBuffer(84, 336, 1, buffer_limit, device)

    print_interval = 1
    target_update_interval = 4
    train_update_interval = 4
    train_start = 1000
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    with open(f'../data/breakout/Uniform-DQN-Breakout{sys.argv[1]}.csv', 'w', newline='') as csvfile:
        fieldnames = ['step', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for n_epi in range(500):
            epsilon = max(0.01, 0.1 - 0.01*(n_epi/200)) #Linear annealing from 10% to 1%
            observation, info = env.reset()

            # preprocess
            observation = preprocess(observation, 210, 160, 84, 84)
            observation = np.expand_dims(observation, axis=2)

            img1 = np.zeros((84,84,1)).astype(np.uint8)
            img2 = np.zeros((84,84,1)).astype(np.uint8)
            img3 = np.zeros((84,84,1)).astype(np.uint8)
            history = deque([img1, img2, img3, observation])
            stacked_images = np.concatenate(list(history), axis=1)

            observation = (torch.tensor(stacked_images).permute(2,0,1) / 255.).unsqueeze(0).to(device)

            terminated = False
            truncated = False
            step = 1
            while not terminated and not truncated:
                action = q.sample_action(observation, epsilon)      
                observation_prime, reward, terminated, truncated, info = env.step(action)
                done_mask = 0.0 if terminated else 1.0
                
                observation_prime = preprocess(observation_prime, 210, 160, 84, 84)
                observation_prime = np.expand_dims(observation_prime, axis=2)

                history.popleft()
                history.append(observation_prime)

                stacked_images = np.concatenate(list(history), axis=1)

                # img = Image.fromarray(stacked_images[:,:,0])
                # img.save(f"file{step:03}.png")
                
                observation_prime = (torch.tensor(stacked_images).permute(2,0,1) / 255.).unsqueeze(0).to(device)
                memory.add((observation,action,reward,observation_prime, done_mask))
                
                observation = observation_prime
                    
                score += reward
                step += 1
                if terminated:
                    break
    
            if memory.real_size>train_start and n_epi % train_update_interval == 0:
                train(q, q_target, memory, optimizer)

            if n_epi%target_update_interval==0 and n_epi!=0:
                q_target.load_state_dict(q.state_dict())
                
            if n_epi%print_interval==0: #and n_epi!=0:
                print(f"step :{n_epi}, score : {score/print_interval:.1f}, n_buffer : {memory.real_size}, eps : {epsilon*100:.1f}%")
                writer.writerow({"step": n_epi, "score":score/print_interval})
                csvfile.flush()
                score = 0.0

    env.close()

if __name__ == '__main__':
    main()