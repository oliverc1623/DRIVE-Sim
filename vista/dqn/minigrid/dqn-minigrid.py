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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 0.00025
gamma         = 0.99
buffer_limit  = 10_000
batch_size    = 32

if torch.cuda.is_available():
    device= 'cuda:1'
else:
    device = 'cpu'
print(f'device:{device}')

class ReplayBuffer():
    def __init__(self, device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (torch.tensor(np.array(s_lst), dtype=torch.float).permute(0,3,1,2).to(self.device))/ 255., torch.tensor(a_lst).to(self.device), \
               torch.tensor(np.array(r_lst)).to(self.device), \
               (torch.tensor(np.array(s_prime_lst), dtype=torch.float).permute(0,3,1,2).to(self.device))/ 255., \
               torch.tensor(np.array(done_mask_lst)).to(self.device)

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, n_input_channels, n_actions):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.mlp1 = nn.Linear(64, 512)
        self.mlp2 = nn.Linear(512, n_actions)

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

    # Q(s′,argmax a ′Q(s ′,a ′;θ i);θ i−)
    Q_next = q_target(s_prime).max(1)[0].unsqueeze(1)
    # print(f"q targ: {q_target(s_prime)}")
    y_i = r + done_mask * gamma * Q_next

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    Q = q(s).gather(1,a)
    # print(f"Q-value estimate: {q(s)}\n")
    # print(f"Q_next: {Q_next}")
    # print(f"target: {y_i}")
    # print(f"TD Error: {(y_i - Q)**2}")

    assert Q.shape == y_i.shape, f"{Q.shape}, {y_i.shape}"

    loss = F.smooth_l1_loss(Q, y_i)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
    memory = ReplayBuffer(device)

    total_frames = 100  # Total number of frames for annealing
    print_interval = 1
    train_update_interval = 4
    target_update_interval = 1_000
    train_start = 1_000
    score = 0
    step = 0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    with open(f'../data/minigrid/Uniform-DQN-Minigrid{sys.argv[1]}.csv', 'w', newline='') as csvfile:
        fieldnames = ['step', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for n_epi in range(500):
            observation, info = env.reset()

            # preprocess
            observation = preprocess(observation['image'])

            terminated = False
            truncated = False
            while not terminated and not truncated:
                epsilon = max(0.1, 1.0 - 0.01*(step/total_frames)) #Linear annealing from 1.0 to 0.1
                action = q.sample_action(observation, epsilon)      
                observation_prime, reward, terminated, truncated, info = env.step(action)
                done_mask = 0.0 if terminated else 1.0
                observation_prime = preprocess(observation_prime['image'])
                memory.put((observation,action,reward,observation_prime, done_mask))
                observation = observation_prime

                if step>train_start and step%train_update_interval==0:
                    train(q, q_target, memory, optimizer)
    
                if step>train_start and step%target_update_interval==0:
                    q_target.load_state_dict(q.state_dict())
                    
                score += reward
                step += 1
                if terminated:
                    break

            if n_epi%print_interval==0:
                print(f"episode :{n_epi}, step: {step}, score : {score/print_interval:.1f}, n_buffer : {memory.size()}, eps : {epsilon*100:.1f}%")
                writer.writerow({"step": step, "score":score/print_interval})
                csvfile.flush()
                score = 0.0

    env.close()

if __name__ == '__main__':
    main()