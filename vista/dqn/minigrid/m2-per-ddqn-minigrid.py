import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
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

# Hyperparameters
learning_rate = 1e-4
gamma = 0.99
buffer_limit = 50_000
batch_size = 32

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"device:{device}")


class PrioritizedReplayBuffer:
    def __init__(
        self,
        state_size_width,
        state_size_height,
        action_size,
        buffer_size,
        device,
        eps=1e-2,
        alpha=0.6,
        beta=0.4,
    ):
        self.tree = SumTree(size=buffer_size)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # transition: state, action, reward, next_state, done
        self.state = torch.empty(
            buffer_size, state_size_width, state_size_height, 1, dtype=torch.int
        )
        self.action = torch.empty(buffer_size, action_size, dtype=torch.int64)
        self.reward = torch.empty(buffer_size, dtype=torch.float32)
        self.next_state = torch.empty(
            buffer_size, state_size_width, state_size_height, 1, dtype=torch.int
        )
        self.done = torch.empty(buffer_size, dtype=torch.float32)

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
        assert (
            self.real_size >= batch_size
        ), "buffer contains less samples than batch size"

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
        self.beta = torch.min(
            torch.tensor([1.0, self.beta + self.beta_increment_per_sampling])
        )
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()

        batch = (
            self.state[sample_idxs].permute(0, 3, 1, 2).to(self.device) / 255.0,
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].unsqueeze(1).to(self.device),
            self.next_state[sample_idxs].permute(0, 3, 1, 2).to(self.device) / 255.0,
            self.done[sample_idxs].unsqueeze(1).to(self.device),
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
    def __init__(self, n_input_channels, n_actions):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.mlp1 = nn.Linear(2560, 512)
        self.mlp2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.mlp1(x))
        x = self.mlp2(x)
        return x

    def sample_action(self, obs, epsilon):
        obs = (torch.tensor(obs).permute(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 5)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    batch, weights, tree_idxs = memory.sample(batch_size)
    s, a, r, s_prime, done_mask = batch

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    Q = q(s).gather(1, a)
    Q_prime_vals = q(s_prime)
    argmax_Q_a = Q_prime_vals.max(1)[1]

    # Rt+1 + γ max_a Q(S_t+1, a; θt). where θ=θ- because we update target params to train params every t steps
    Q_next = q_target(s_prime)
    q_target_s_a_prime = Q_next.gather(1, argmax_Q_a.unsqueeze(1))
    y_i = r + done_mask * gamma * q_target_s_a_prime

    # y_i = y_i.squeeze()
    assert Q.shape == y_i.shape, f"{Q.shape}, {y_i.shape}"
    if weights is None:
        weights = torch.ones_like(Q)
    td_error = torch.abs(Q - y_i).detach().squeeze()
    loss = torch.mean(weights.to(device) * F.smooth_l1_loss(Q, y_i))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    memory.update_priorities(tree_idxs, td_error.cpu().numpy())
    return Q.mean()


# Convert image to greyscale, resize and normalise pixels
def preprocess(image, history):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=2)
    # stack images
    history.popleft()
    history.append(image)
    image = np.concatenate(list(history), axis=1)
    return image


def main():
    env = gym.make("MiniGrid-Empty-6x6-v0", render_mode="rgb_array")
    env = RGBImgObsWrapper(env)  # Get pixel observations

    # set seed for reproducibility
    seed = int(sys.argv[1])
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    q = Qnet(1, 6)
    q_target = Qnet(1, 6)
    q_target.load_state_dict(q.state_dict())
    q.to(device)
    q_target.to(device)
    memory = PrioritizedReplayBuffer(48, 192, 1, buffer_limit, device, alpha=0.6, beta=0.4)

    total_frames = 300  # Total number of frames for annealing
    print_interval = 1
    train_update_interval = 4
    target_update_interval = 5_000
    train_start = 2_000
    score = 0
    step = 0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    q_value = torch.tensor(0)

    with open(
        f"../data/minigrid/PER-DDQN-Minigrid{sys.argv[1]}.csv", "w", newline=""
    ) as csvfile:
        fieldnames = ["step", "score", "Q-value"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        fg = plt.figure()
        ax = fg.gca()
        for n_epi in range(600):
            observation, info = env.reset()

            # preprocess
            img1 = np.zeros((48, 48, 1)).astype(np.uint8)
            img2 = np.zeros((48, 48, 1)).astype(np.uint8)
            img3 = np.zeros((48, 48, 1)).astype(np.uint8)
            img4 = np.zeros((48, 48, 1)).astype(np.uint8)
            history = deque([img1, img2, img3, img4])
            observation = preprocess(observation["image"], history)

            # h = ax.imshow(observation, cmap="gray")

            terminated = False
            truncated = False
            while not terminated and not truncated:
                epsilon = max(
                    0.1, 1.0 - 0.01 * (step / total_frames)
                )  # Linear annealing from 1.0 to 0.1
                action = q.sample_action(observation, epsilon)
                observation_prime, reward, terminated, truncated, info = env.step(
                    action
                )
                done_mask = 0.0 if terminated else 1.0
                observation_prime = preprocess(observation_prime["image"], history)

                # uncomment block to download frames as PNGs
                # img = Image.fromarray((np.squeeze(observation_prime)).astype(np.uint8), 'L')
                # img = img.resize((480,120), resample=Image.BOX)
                # img.save(f"frames/frame_{step:04d}.png")

                # render image
                # h.set_data(observation_prime)
                # plt.draw(), plt.pause(1e-3)

                memory.add((observation, action, reward, observation_prime, done_mask))
                observation = observation_prime

                if step > train_start and step % train_update_interval == 0:
                    q_value = train(q, q_target, memory, optimizer)

                if step > train_start and step % target_update_interval == 0:
                    q_target.load_state_dict(q.state_dict())

                score += reward
                step += 1
                if terminated:
                    break

            if n_epi % print_interval == 0:
                print(
                    f"episode :{n_epi}, step: {step}, score : {score/print_interval:.1f}, n_buffer : {memory.real_size}, eps : {epsilon*100:.1f}%"
                )
                writer.writerow(
                    {
                        "step": step,
                        "score": score / print_interval,
                        "Q-value": q_value.item(),
                    }
                )
                csvfile.flush()
                score = 0.0

    env.close()


if __name__ == "__main__":
    main()
