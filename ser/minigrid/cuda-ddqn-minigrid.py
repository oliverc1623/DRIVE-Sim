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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 1e-4
gamma = 0.99
buffer_limit = 50_000
batch_size = 32

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(f"device:{device}")

class ReplayBuffer:
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

        return (
            torch.tensor(np.array(s_lst), dtype=torch.int)
            .permute(0, 3, 1, 2)
            .to(self.device),
            torch.tensor(a_lst, dtype=torch.int64).to(self.device),
            torch.tensor(np.array(r_lst), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(s_prime_lst), dtype=torch.int)
            .permute(0, 3, 1, 2)
            .to(self.device),
            torch.tensor(np.array(done_mask_lst), dtype=torch.float32).to(self.device),
        )

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, n_input_channels, n_actions):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.mlp1 = nn.Linear(1024, 512)
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
    s, a, r, s_prime, done_mask = memory.sample(batch_size)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    Q = q(s / 255.0).gather(1, a)
    Q_prime_vals = q(s_prime / 255.0)
    argmax_Q_a = Q_prime_vals.max(1)[1]

    # Rt+1 + γ max_a Q(S_t+1, a; θt). where θ=θ- because we update target params to train params every t steps
    Q_next = q_target(s_prime / 255.0)
    q_target_s_a_prime = Q_next.gather(1, argmax_Q_a.unsqueeze(1))
    y_i = r + done_mask * gamma * q_target_s_a_prime

    # y_i = y_i.squeeze()
    assert Q.shape == y_i.shape, f"{Q.shape}, {y_i.shape}"
    loss = F.smooth_l1_loss(Q, y_i)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return Q.mean()


# Convert image to greyscale, resize and normalise pixels
def preprocess(image, history):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # uncomment if using gray scale
    # image = np.expand_dims(image, axis=2) 
    # stack images
    history.popleft()
    history.append(image)
    image = np.concatenate(list(history), axis=1)
    return image


def main():
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    env = RGBImgObsWrapper(env)  # Get pixel observations

    # set seed for reproducibility
    seed = int(sys.argv[1])
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    q = Qnet(3, 6)
    q_target = Qnet(3, 6)
    q_target.load_state_dict(q.state_dict())
    q.to(device)
    q_target.to(device)
    memory = ReplayBuffer(device)

    total_frames = 200  # Total number of frames for annealing
    print_interval = 10
    train_update_interval = 4
    target_update_interval = 5_000
    train_start = 1_000
    score = 0
    prev_score = -float('inf')
    step = 0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    q_value = torch.tensor(0)

    with open(
        f"logs/Uniform-DDQN-Minigrid{sys.argv[1]}.csv", "w", newline=""
    ) as csvfile:
        fieldnames = ["step", "score", "Q-value"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        fg = plt.figure()
        ax = fg.gca()
        for n_epi in range(600):
            observation, info = env.reset()

            # preprocess
            img1 = np.zeros((40, 40, 3)).astype(np.uint8)
            img2 = np.zeros((40, 40, 3)).astype(np.uint8)
            img3 = np.zeros((40, 40, 3)).astype(np.uint8)
            img4 = np.zeros((40, 40, 3)).astype(np.uint8)
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
                img = Image.fromarray((np.squeeze(observation_prime)).astype(np.uint8), 'RGB')
                img = img.resize((480,120), resample=Image.BOX)
                img.save(f"frames/frame_{step:04d}.png")

                # render image
                # h.set_data(observation_prime)
                # plt.draw(), plt.pause(1e-3)

                memory.put((observation, action, reward, observation_prime, done_mask))
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
                    f"episode :{n_epi}, step: {step}, score : {score/print_interval:.1f}, n_buffer : {memory.size()}, eps : {epsilon*100:.1f}%"
                )
                writer.writerow(
                    {
                        "step": step,
                        "score": score / print_interval,
                        "Q-value": q_value.item(),
                    }
                )
                if score > prev_score:
                    prev_score = score
                    torch.save(q.state_dict(), 'q-model-rgb.pth')
                csvfile.flush()
                score = 0.0

    env.close()


if __name__ == "__main__":
    main()
