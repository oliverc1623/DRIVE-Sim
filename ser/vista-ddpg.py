import os
os.environ["DISPLAY"] = ":1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
# local imports
sys.path.append('../vista')
from VistaEnv import VistaEnv
import copy

import gymnasium as gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
batch_size   = 32
buffer_limit = 50000
tau          = 0.005 # for target network soft update

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
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return  torch.tensor(s_lst), \
                torch.tensor(np.array(a_lst), dtype=torch.float, device=self.device), \
                torch.tensor(np.array(r_lst), dtype=torch.float, device=self.device), \
                (torch.tensor(np.array(s_prime_lst), dtype=torch.float).permute(0,3,1,2).to(self.device))/ 255.0, \
                torch.tensor(np.array(done_mask_lst), dtype=torch.float, device=self.device)
    
    def size(self):
        return len(self.buffer)

class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(1634304, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))        
        mu = torch.tanh(self.fc_mu(x))
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32,1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
      
def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s,a,r,s_prime,done_mask  = memory.sample(batch_size)

    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.smooth_l1_loss(q(s,a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s,mu(s)).mean() # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()
    
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
    
def main():
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
    camera_config = {
        'type': 'camera',
        'name': 'camera_front',
        'rig_path': './RIG.xml',
        'optical_flow_root': '../data_prep/Super-SloMo/slowmo',
        'size': (400, 640)
    }
    ego_car_config = copy.deepcopy(car_config)
    ego_car_config['lookahead_road'] = True
    trace_root = "../vista/vista_traces"
    trace_path = [
        "20210726-154641_lexus_devens_center", 
        "20210726-155941_lexus_devens_center_reverse", 
        "20210726-184624_lexus_devens_center", 
        "20210726-184956_lexus_devens_center_reverse", 
    ]
    trace_paths = [os.path.join(trace_root, p) for p in trace_path]
    display_config = dict(road_buffer_size=1000, )
    preprocess_config = {
        "crop_roi": True,
        "resize": False,
        "grayscale": False,
        "binary": False
    }
    env = VistaEnv(
        trace_paths = trace_paths, 
        trace_config = trace_config,
        car_config = car_config,
        display_config = display_config,
        preprocess_config = preprocess_config,
        sensors_configs = [camera_config]
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    memory = ReplayBuffer(device)

    q, q_target = QNet().to(device), QNet().to(device)
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet().to(device), MuNet().to(device)
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 1

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer  = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False

        count = 0
        while count < 500 and not done:
            s = (torch.tensor(s).permute(0,3,1,2) / 255.0).to(device)
            a = mu(s)
            a = a.item() + ou_noise()[0]
            s_prime, r, done, truncated, info = env.step([a])
            memory.put((s,a,r/100.0,s_prime,done))
            score +=r
            s = s_prime
            count += 1

        print(f"memory size: {memory.size()}")
        if memory.size()>100:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q,  q_target)

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()