import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import csv
import io


# Hyperparameters
learning_rate = 0.0005
gamma = 0.99
lmbda = 0.95
eps_clip = 0.2
K_epoch = 3
T_horizon = 128

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.conv1 = nn.Conv2d(3, 32, 2)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.conv3 = nn.Conv2d(64, 64, 2)
        self.fc1 = nn.Linear(87616, 512)
        self.fc_pi = nn.Linear(512, 6)
        self.fc_v = nn.Linear(512, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = (
            torch.tensor(np.array(s_lst), dtype=torch.float, device=device),
            torch.tensor(np.array(a_lst), device=device),
            torch.tensor(np.array(r_lst), device=device),
            torch.tensor(np.array(s_prime_lst), dtype=torch.float, device=device),
            torch.tensor(np.array(done_lst), dtype=torch.float, device=device),
            torch.tensor(np.array(prob_a_lst), device=device),
        )
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        print(f"s: {s.shape}")
        print(f"a: {a.shape}")
        print(f"r: {r.shape}")
        print(f"s_prime: {s_prime.shape}")
        print(f"done: {done_mask.shape}")
        print(f"prob_a: {prob_a.shape}\n")

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float, device=device)

            pi = self.pi(s, softmax_dim=0)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(
                torch.log(pi_a) - torch.log(prob_a)
            )  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(
                self.v(s), td_target.detach()
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def preprocess(self, x):
        print(f"preprocess shape: {x.shape}")
        x = torch.from_numpy(x).float() / 255.0
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            x = x.permute(0, 3, 1, 2).to(device)
        print(f"postprocess shape: {x.shape}")
        return x


def main():
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    env = RGBImgObsWrapper(env)  # Get pixel observations
    model = PPO().to(device)
    score = 0.0
    print_interval = 1
    # Create an in-memory string buffer
    output = io.StringIO()
    # Use the string buffer instead of a file
    fieldnames = ["episode", "score"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for n_epi in range(4):
        s, _ = env.reset()
        s = s["image"]
        s = model.preprocess(s)
        done = False
        truncated = False
        while not done and not truncated:
            for t in range(T_horizon):
                prob = model.pi(s)
                m = Categorical(prob[0])
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)
                s_prime = model.preprocess(s_prime["image"])

                model.put_data((s, a, r, s_prime, prob[0][a].item(), done))
                s = s_prime

                score += r
                if done or truncated:
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            
            print("# of episode :{}, avg score : {:.2f}".format(n_epi, score / print_interval))
            writer.writerow({"episode": n_epi, "score": score / print_interval})
            csv_content = output.getvalue()
            with open('episode_scores.csv', 'w', newline='') as csvfile:
                csvfile.write(csv_content)
            score = 0.0

    env.close()
    csv_content = output.getvalue()
    output.close()


if __name__ == "__main__":
    main()
