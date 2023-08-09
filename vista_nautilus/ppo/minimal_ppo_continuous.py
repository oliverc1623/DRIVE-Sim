# credit to: https://github.com/seolhokim
# import gymnasium as gym
import os
os.environ["DISPLAY"] = ":1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging, misc
from vista.tasks import MultiAgentBase
from vista.utils import transform
import sys
sys.path.insert(1, '../vista_nautilus/')
sys.path.insert(1, '../vista_nautilus/models/')
from helper import * 

#Hyperparameters
entropy_coef = 1e-2
critic_coef = 1
learning_rate = 0.0003
gamma         = 0.9
lmbda         = 0.9
eps_clip      = 0.2
K_epoch       = 10
T_horizon     = 20


class PPO(nn.Module):
    def __init__(self, device):
        super(PPO, self).__init__()
        self.data = []
        self.device = device
        
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

    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append(torch.tensor(r,dtype=torch.float32))
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s_lst = torch.stack(s_lst).to(self.device)
        r_lst = torch.stack(r_lst).to(self.device)
        s_prime_lst = torch.stack(s_prime_lst).to(dtype=torch.float32)
        
        s = s_lst 
        a = torch.tensor(a_lst).to(self.device)
        r = r_lst 
        s_prime = s_prime_lst.to(self.device)
        done_mask = torch.tensor(done_lst).to(self.device)
        prob_a = torch.tensor(prob_a_lst).to(self.device)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, old_log_prob = self.make_batch()
        # print(f"r shape: {r.shape}")
        for i in range(K_epoch):
            td_target = r.unsqueeze(1) + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = torch.flip(delta, (0,)) #delta

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float32).to(self.device)

            curr_mu,curr_sigma = self.pi(s)
            
            curr_dist = torch.distributions.Normal(curr_mu,curr_sigma)
            curr_log_prob = curr_dist.log_prob(a)
            entropy = curr_dist.entropy() * entropy_coef
            
            ratio = torch.exp(curr_log_prob - old_log_prob.detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            actor_loss = (-torch.min(surr1, surr2) - entropy).mean() 
            v = self.v(s).float()
            td = td_target.detach().float()
            critic_loss = critic_coef * F.smooth_l1_loss(v, td)
            loss = actor_loss + critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            


def main(render = False):
    if torch.cuda.is_available():
        device= 'cuda:1'
    else:
        device = 'cpu'
    print('device:{}'.format(device))
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
            size=(400, 640), # (200, 320) for lighter cnn, (355, 413) for 80x200
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
    display_config = dict(road_buffer_size=1000, )

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
    display = vista.Display(env.world, display_config=display_config)

    # start env
    env.reset();
    display.reset()
    model = PPO(device).to(device)

    import cv2

    print_interval = 1
    score = 0.0
    global_step = 0
    for n_epi in range(10000):
        print(f"n_episode: {n_epi}")
        observation = env.reset();
        observation = grab_and_preprocess_obs(observation, env, device)
        done = False
        while not done:
            for t in range(T_horizon):
                global_step += 1 
                mu,sigma = model.pi(observation.permute(2,0,1))
                dist = torch.distributions.Normal(mu,sigma)
                actions = sample_actions(dist, env.world, env.ego_agent.id)
                a = torch.tensor([[actions[env.ego_agent.id][0]]], dtype=torch.float32).to(device)
                log_prob = dist.log_prob(a)
                observation_prime, rewards, dones, infos = env.step(actions)
                observation_prime = grab_and_preprocess_obs(observation_prime, env, device)
                done = rewards[env.ego_agent.id][1]['done']
                reward = 0.0 if done else rewards[env.ego_agent.id][0]
                print(f"reward: {reward}")
                print(f"done: {done}")
    
                model.put_data((observation, a, reward, observation_prime, \
                                log_prob, done))
                observation = observation_prime
                
                score += reward
                if done:
                    break
            model.train_net()
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

main()