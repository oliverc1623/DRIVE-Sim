import sys
import argparse
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

class PrioritizedReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, device, eps=1e-2, alpha=0.1, beta=0.1):
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

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
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
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

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
    print(f"Trial: {sys.argv[1]}")
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    steps_done = 0

    with open(f'no_frill_per_ddqn_trial{sys.argv[1]}.csv', 'w', newline='') as csvfile:
        fieldnames = ['episode', 'duration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        env = gym.make("CartPole-v1")    
        # torch.manual_seed(sys.argv[1])
        set_seed(env, int(sys.argv[1]))
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
        # Initialize replay memory D to capacity N
        N = 50_000
        replay_mem = PrioritizedReplayBuffer(4, 1, N, device)
        
        # Initialize action-value function Q with random weights
        policy_net = DQN(4, 2).to(device)
        target_net = DQN(4, 2).to(device)
        
        # target_net = DQN(4, 2).to(device)
        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        
        # For episode 1 --> M
        for i in range(600):
            
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
                    #state_b, action_b, reward_b, next_state_b, done_b = replay_mem.sample(BATCH_SIZE)
                    batch, weights, tree_idxs = replay_mem.sample(BATCH_SIZE)
                    state_b, action_b, reward_b, next_state_b, done_b = batch
                    
                    # Q(s′,argmax a ′Q(s ′,a ′;θ i);θ i−)
                    Q_next = target_net(next_state_b).max(1)[0]
                    y_i = reward + (1-done_b) * GAMMA * Q_next

                    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                    Q = policy_net(state_b)[torch.arange(len(action_b)), action_b.to(torch.long).flatten()]

                    assert Q.shape == y_i.shape, f"{Q.shape}, {y_i.shape}"
                    
                    if weights is None:
                        weights = torch.ones_like(Q)
                        
                    td_error = torch.abs(Q - y_i).detach()

                    loss = torch.mean((y_i - Q)**2 * weights.to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    with torch.no_grad():
                        # Soft update of the target network's weights
                        # θ′ ← τ θ + (1 −τ )θ′
                        for tp, sp in zip(target_net.parameters(), policy_net.parameters()):
                            tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)

                    replay_mem.update_priorities(tree_idxs, td_error.cpu().numpy())

                state = next_state

                # increment global step counter for 
                steps_done += 1
                
                if done:
                    writer.writerow({"episode": i, "duration":t})
                    csvfile.flush()
                    break

if __name__=="__main__":
    main()