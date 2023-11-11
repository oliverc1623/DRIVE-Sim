import gymnasium as gym
import collections
import random
import numpy as np
import sys, os
import csv
from tree import SumTree

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

class PrioritizedReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, device, eps=1e-2, alpha=0.6, beta=0.4):
        self.tree = SumTree(size=buffer_size)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # transition: state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        self.device = device
        self.beta_increment_per_sampling = 0.001

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

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
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            cumsum = random.uniform(a, b)
            tree_idx, priority, sample_idx = self.tree.get(cumsum)
            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total
        self.beta = torch.min(torch.tensor([1., self.beta + self.beta_increment_per_sampling]))
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()

        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device)
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer, device):
    for i in range(10):
        batch, weights, tree_idxs = memory.sample(batch_size)
        s,a,r,s_prime,done_mask = batch

        # Q(s′,argmax a ′Q(s ′,a ′;θ i);θ i−)
        Q_next = q_target(s_prime).max(1)[0]
        y_i = r + done_mask * gamma * Q_next

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        Q = q(s)[torch.arange(len(a)), a.to(torch.long).flatten()]

        assert Q.shape == y_i.shape, f"{Q.shape}, {y_i.shape}"

        if weights is None:
            weights = torch.ones_like(Q)

        td_error = torch.abs(Q - y_i).detach()
        
        loss = (weights.to(device) * F.smooth_l1_loss(Q, y_i)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        memory.update_priorities(tree_idxs, td_error.cpu().numpy())

def main():
    env = gym.make('CartPole-v1')
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


    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    q.to(device)
    q_target.to(device)
    memory = PrioritizedReplayBuffer(4, 1, buffer_limit, device)

    print_interval = 1
    target_update_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    with open(f'../data/cartpole/PER-DQN-Cartpole{sys.argv[1]}.csv', 'w', newline='') as csvfile:
        fieldnames = ['step', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for n_epi in range(10000):
            epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
            observation, info = env.reset()
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action = q.sample_action(torch.from_numpy(observation).float().to(device), epsilon)      
                observation_prime, reward, terminated, truncated, info = env.step(action)
                done_mask = 0.0 if terminated else 1.0
                memory.add((observation,action,reward/100.0,observation_prime, done_mask))
                observation = observation_prime
    
                score += reward
                if terminated:
                    break

            if memory.real_size>2000:
                train(q, q_target, memory, optimizer, device)

            if n_epi%target_update_interval==0 and n_epi!=0:
                q_target.load_state_dict(q.state_dict())

            if n_epi%print_interval==0 and n_epi!=0:
                print(f"step :{n_epi}, score : {score/print_interval:.1f}, n_buffer : {memory.real_size}, eps : {epsilon*100:.1f}%")
                writer.writerow({"step": n_epi, "score":score/print_interval})
                csvfile.flush()
                score = 0.0

    env.close()

if __name__ == '__main__':
    main()