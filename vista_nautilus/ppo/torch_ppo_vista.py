import os
os.environ["DISPLAY"] = ":1.0"
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import vista
from replay_memory import Memory
import math
from common import estimate_advantages
import matplotlib.pyplot as plt
import numpy
import sys
sys.path.insert(1, '../vista_nautilus/models/')
sys.path.insert(1, '../vista_nautilus/')
from helper import * 
import a2c_cnn

dtype = torch.float32
torch.set_default_dtype(dtype)
device = ("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device(device)
print(f"Using {device} device")

#Hyperparameters
learning_rate   = 0.00005
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

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(8, 24)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(24, 36, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(9, 36)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(36, 48, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.GroupNorm(12, 48)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.GroupNorm(64, 64)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.GroupNorm(1, 2)
        self.relu5 = nn.ReLU()
        self.fc1 = nn.Linear(2 * 70 * 310, 100)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.relu7 = nn.ReLU() 
        self.fc3 = nn.Linear(100, 2)
        self.fc_v = nn.Linear(100, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def pi(self, x):
        single_image_input = len(x.shape) == 3  # missing 4th batch dimension
        if single_image_input:
            x = x.unsqueeze(0)
        else:
            x = x.permute(0, 3, 1, 2)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.relu5(self.norm5(self.conv5(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu6(self.fc1(x))
        x = self.relu7(self.fc2(x))
        x = self.fc3(x)
        mu, log_sigma = torch.chunk(x, 2, dim=-1)
        mu = 1/8.0 * torch.tanh(mu)  # conversion
        sigma = 0.1 * torch.sigmoid(log_sigma) + 0.005  # conversion
        return mu, sigma
    
    def v(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.relu5(self.norm5(self.conv5(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu6(self.fc1(x))
        x = self.relu7(self.fc2(x))
        v = self.fc_v(x)
        return v

    def collect_samples(self, mini_batch_size, env):
        log = dict()
        memory = Memory()
        num_steps = 0
        total_reward = 0
        num_episodes = 0
        while num_steps < mini_batch_size:
            ob = env.reset();
            s = grab_and_preprocess_obs(ob, env, device)
            trace_index = env.ego_agent.trace_index
            steps = 0
            initial_frame = env.ego_agent.frame_index
            reward_episode = 0
            for t in range(10000):
                mu, sigma = self.pi(s.permute(2,0,1))
                curvature_dist = dist.Normal(mu, sigma)
                actions = sample_actions(curvature_dist, env.world, env.ego_agent.id)
                a = torch.tensor([[actions[env.ego_agent.id][0]]]).to(device)
                log_prob = curvature_dist.log_prob(a)
                observations, rewards, dones, infos = env.step(actions)
                terminal_conditions = rewards[env.ego_agent.id][1]
                s_prime = grab_and_preprocess_obs(observations, env, device)
                done = terminal_conditions['done']
                reward = 0.0 if done else rewards[env.ego_agent.id][0]
                reward_episode += reward
                mask = 0 if done else 1
                memory.push(s, a, mask, s_prime, reward, log_prob.item())
                if done:
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
            print(f"s shape: {s.shape}")
            print(f"s prime shape: {s_prime.shape}")
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
        advantages, td_target = self.calc_advantages(states.detach(), 
                                                     actions.detach(),
                                                     rewards.detach(), 
                                                     s_primes.detach(), 
                                                     masks.detach())
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

def main():
    # Initialize the simulator
    trace_config = dict(
        road_width=4,
        reset_mode='default',
        master_sensor='camera_front',
    )
    car_config = dict(
        length=5.,
        width=2.,
        wheel_base=2.78,
        steering_ratio=14.7,
        lookahead_road=True,
    )
    sensors_config = [
        dict(
            type='camera',
            # camera params
            name='camera_front',
            size=(400, 640), # for lighter cnn 
            # rendering params
            depth_mode=DepthModes.FIXED_PLANE,
            use_lighting=False,
        )
    ]
    task_config = dict(n_agents=2,
                        mesh_dir="carpack01",
                        init_dist_range=[50., 60.],
                        init_lat_noise_range=[-3., 3.],
                        reward_fn=my_reward_fn)
    # display_config = dict(road_buffer_size=1000, )

    ego_car_config = copy.deepcopy(car_config)
    ego_car_config['lookahead_road'] = True
    trace_root = "vista_traces"
    trace_path = [
        "20210726-154641_lexus_devens_center", 
        "20210726-155941_lexus_devens_center_reverse", 
        "20210726-184624_lexus_devens_center", 
        "20210726-184956_lexus_devens_center_reverse", 
    ]
    trace_path = [os.path.join(trace_root, p) for p in trace_path]
    env = MultiAgentBase(trace_paths=trace_path,
                            trace_config=trace_config,
                            car_configs=[car_config] * task_config['n_agents'],
                            sensors_configs=[sensors_config] + [[]] *
                            (task_config['n_agents'] - 1),
                            task_config=task_config)

    model = PPO().to(device)
    print(model)

    # open and write to file to track progress
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = "results"
    model_results_dir = results_dir + "/CNN/" # TODO: make this into variable

    if not os.path.exists(model_results_dir):
        os.makedirs(model_results_dir)
    # Define the file path
    filename = "ppo_trial1"
    print("Writing log to: " + filename + ".txt")
    file_path = os.path.join(model_results_dir, filename + ".txt")
    f = open(file_path, "w")
    f.write("num_steps\tnum_episodes\ttotal_reward\tavg_reward\n")
    
    for n_epi in range(500):
        print(f"Episode: {n_epi}")
        batch, log = model.collect_samples(minibatch_size, env)
        print(log)
        # Write reward and loss to results txt file
        f.write(f"{log['num_steps']}\t{log['num_episodes']}\t{log['total_reward']}\t{log['avg_reward']}\n")
        f.flush()
        model.update_params(batch, n_epi)

if __name__ == '__main__':
    main()
