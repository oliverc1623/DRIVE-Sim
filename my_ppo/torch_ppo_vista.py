import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import vista
import os
from vista_helper import *
from replay_memory import Memory
import math
from common import estimate_advantages
import matplotlib.pyplot as plt
import numpy

dtype = torch.float32
torch.set_default_dtype(dtype)
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device(device)
print(f"Using {device} device")

#Hyperparameters
learning_rate   = 0.0005
gamma           = 0.95
lmbda           = 0.9
tau             = 0.95
eps_clip        = 0.2
K_epoch         = 10
minibatch_size  = 128
optim_batch_size = 32

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(8, 32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(16, 64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.GroupNorm(32, 128)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.GroupNorm(64, 256)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.GroupNorm(1, 2)
        self.relu5 = nn.ReLU()
        self.fc = nn.Linear(2 * 32 * 30, 2)
        self.fc_v = nn.Linear(3 * 32 * 20, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def pi(self, x):
        single_image_input = len(x.shape) == 3  # missing 4th batch dimension
        if single_image_input:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.relu5(self.norm5(self.conv5(x)))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        mu, log_sigma = torch.chunk(x, 2, dim=-1)
        mu = 1/8.0 * torch.tanh(mu)  # conversion
        sigma = 0.1 * torch.sigmoid(log_sigma) + 0.005  # conversion
        return mu, sigma
    
    def v(self, x):
        x = x.permute(0,3,1,2)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.relu5(self.norm5(self.conv5(x)))
        x = x.reshape(x.size(0), -1)
        v = self.fc_v(x)
        return v

    def collect_samples(self, mini_batch_size, world, display, car, camera):
        log = dict()
        memory = Memory()
        num_steps = 0
        total_reward = 0
        num_episodes = 0
        while num_steps < mini_batch_size:
            world.set_seed(47)
            vista_reset(world, display)
            prev_curvature = 0.0
            s = grab_and_preprocess_obs(car, camera).to(device)
            reward_episode = 0
            for t in range(10000):
                mu, std = self.pi(s)
                dist = Normal(mu, std)
                a = dist.sample()
                log_prob = dist.log_prob(a)
                curvature_action = a.item()
                vista_step(car, curvature_action)
                s_prime = grab_and_preprocess_obs(car, camera).to(device)
                r = calculate_reward(car, curvature_action, prev_curvature)
                prev_curvature = curvature_action
                reward_episode += r
                crash = check_crash(car)
                mask = 0 if crash else 1
                memory.push(s, a, mask, s_prime, r, log_prob.item())
                if crash:
                    break
                s = s_prime
            # log stats
            num_steps += (t + 1)
            num_episodes += 1
            total_reward += reward_episode
        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward.item()
        log['avg_reward'] = (total_reward / num_episodes).item()
        return memory.sample(), log

    def ppo_step(self, states, actions, advantages, fixed_log_probs, clip_epsilon, td_target):
        """update policy"""
        mu, std = self.pi(states)
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        vs = self.v(states)
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(vs , td_target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 40.0)
        self.optimizer.step()
        self.optimization_step += 1

    def calc_advantages(self, s, a, r, s_prime, done_mask):
        with torch.no_grad():
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
        delta = delta.cpu().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float)
        return advantage, td_target

    def update_params(self, batch, i_iter):
        states = torch.stack(batch.state)
        actions = torch.stack(batch.action).to(dtype).squeeze(1).to(device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device).unsqueeze(1)
        masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device).unsqueeze(1)
        fixed_log_probs = torch.from_numpy(np.stack(batch.log_probs)).to(dtype).to(device)
        s_primes = torch.stack(batch.next_state)
        # ** Might need to implement another way to get fixed log probs**
        """get advantage estimation from the trajectories"""
        advantages, td_target = self.calc_advantages(states, actions, rewards, s_primes, masks)
        advantages = advantages.to(device)
        td_target = td_target.to(device)

        """perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
        # print(f"optim iter: {optim_iter_num}")
        for _ in range(K_epoch):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)

            states, actions, advantages, fixed_log_probs, td_target = \
                states[perm].clone(), actions[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone(), \
                td_target[perm].clone()
            
            for i in range(optim_iter_num):
                ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, fixed_log_probs_b, td_target_b = \
                    states[ind], actions[ind], advantages[ind], fixed_log_probs[ind], td_target[ind]

                self.ppo_step(states_b, actions_b, advantages_b, fixed_log_probs_b, eps_clip, td_target_b)
                # Check gradients norms
                # total_norm = 0
                # for n, p in self.named_parameters():
                #     param_norm = p.grad.data.norm(2) # calculate the L2 norm of gradients
                #     total_norm += param_norm.item() ** 2 # accumulate the squared norm
                # total_norm = total_norm ** 0.5 # take the square root to get the total norm
                # print(f"Total gradient norm: {total_norm}")

def main():
    # Set up VISTA simulator
    trace_root = "trace"
    trace_path = [
        "20210726-154641_lexus_devens_center",
        "20210726-155941_lexus_devens_center_reverse",
        "20210726-184624_lexus_devens_center",
        "20210726-184956_lexus_devens_center_reverse",
    ]
    trace_path = [os.path.join(trace_root, p) for p in trace_path]
    world = vista.World(trace_path, trace_config={"road_width": 4})
    car = world.spawn_agent(
        config={
            "length": 5.0,
            "width": 2.0,
            "wheel_base": 2.78,
            "steering_ratio": 14.7,
            "lookahead_road": True,
        }
    )
    camera = car.spawn_camera(config={"size": (200, 320)})
    display = vista.Display(
        world, display_config={"gui_scale": 2, "vis_full_frame": False}
    )

    model = PPO().to(device)

    for n_epi in range(500):
        print(f"Episode: {n_epi}")
        batch, log = model.collect_samples(minibatch_size, world, display, car, camera)
        print(log)
        model.update_params(batch, n_epi)

if __name__ == '__main__':
    main()
