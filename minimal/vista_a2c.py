import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import torch.multiprocessing as mp
import numpy as np
import vista
import os
from vista_helper import *
import matplotlib.pyplot as plt
import cv2
from vista.entities.agents.Dynamics import curvature2tireangle
from vista.entities.agents.Dynamics import curvature2steering

# Hyperparameters
n_train_processes = 3
learning_rate = 0.0005
update_interval = 100
gamma = 0.95
max_train_steps = 60000
PRINT_INTERVAL = 5

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(3, 80, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(8, 80)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(80, 64, kernel_size=3, stride=1, padding=1)
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
        self.fc = nn.Linear(2 * 80 * 80, 2)
        self.fc_v = nn.Linear(2 * 80 * 80, 1)

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
        x = self.fc(x)

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
        v = self.fc_v(x)
        return v

def worker(worker_id, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
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
    world.set_seed(worker_id)
    prev_curvature = car.curvature
    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            curvature = data.item()
            vista_step(car, curvature)
            ob = grab_and_preprocess_obs(car, camera)
            # display.render()
            # plt.pause(0.5)
            reward = calculate_reward(car, prev_curvature)
            prev_curvature = curvature
            done = int(check_crash(car))
            if done:
                reward = torch.tensor(0.0, dtype=torch.float32)
                world.set_seed(worker_id)
                vista_reset(world, display)
                prev_curvature = 0.0
            worker_end.send((ob, reward, torch.tensor(done)))
        elif cmd == 'reset':
            # ob, info = env.reset()
            vista_reset(world, display)
            ob = grab_and_preprocess_obs(car, camera)
            worker_end.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            worker_end.send(ob)
        elif cmd == 'close':
            worker_end.close()
            break
        elif cmd == 'get_spaces':
            worker_end.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_train_processes):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones = zip(*results)
        return torch.stack(obs), torch.stack(rews), torch.stack(dones)

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        return torch.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True

def test(step_idx, model, world, car, display, camera, device):
    world.set_seed(47)
    vista_reset(world, display)
    score = 0.0
    num_test = 1
    for _ in range(num_test):
        step = 0
        world.set_seed(47)
        vista_reset(world, display)
        observation = grab_and_preprocess_obs(car, camera)
        done = False
        prev_curvature = 0.0
        while not done:
            mu, sigma = model.pi(observation.permute(2,0,1))
            dist = Normal(mu, sigma)
            action = dist.sample().item()
            vista_step(car, action)
            prev_curvature = action
            observation_prime = grab_and_preprocess_obs(car, camera)
            reward = calculate_reward(car, prev_curvature)
            done = int(check_crash(car))

            observation = observation_prime
            score += reward
            step += 1
        print(f"total steps: {step}")
        done = False
    print(f"Step # :{step_idx}, avg score : {score/num_test:.1f}")

def compute_target(v_final, r_lst, mask_lst):
    G = v_final.reshape(-1)
    td_target = list()

    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r.numpy() + gamma * G * mask.numpy()
        td_target.append(G)

    return torch.tensor(np.array(td_target[::-1])).float()

if __name__ == '__main__':
    device = 'cpu'
    print('device:{}'.format(device))
    envs = ParallelEnv(n_train_processes)

    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ### VISTA World for testing
    trace_root = "trace"
    trace_path = [
        "20210726-154641_lexus_devens_center",
        "20210726-155941_lexus_devens_center_reverse",
        "20210726-184624_lexus_devens_center",
        "20210726-184956_lexus_devens_center_reverse",
    ]
    trace_path = [os.path.join(trace_root, p) for p in trace_path]
    world_test = vista.World(trace_path, trace_config={"road_width": 4})
    car_test = world_test.spawn_agent(
        config={
            "length": 5.0,
            "width": 2.0,
            "wheel_base": 2.78,
            "steering_ratio": 14.7,
            "lookahead_road": True,
        }
    )
    camera_test = car_test.spawn_camera(config={"size": (200, 320)})
    display_test = vista.Display(
        world_test, display_config={"gui_scale": 2, "vis_full_frame": False}
    )
    ### 

    step_idx = 0
    s = envs.reset()
    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, mask_lst, s_primes = list(), list(), list(), list(), list()
        for _ in range(update_interval):
            mu, sigma = model.pi(s)
            dist = Normal(mu, sigma)
            a = dist.sample()
            s_prime, r, done = envs.step(a)
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            mask_lst.append(1 - done)

            s = s_prime
            step_idx += 1

        s_final = s_prime 
        v_final = model.v(s_final).detach().cpu().clone().numpy()
        td_target = compute_target(v_final, r_lst, mask_lst)
        td_target_vec = td_target.reshape(-1)
        s_vec = torch.stack(s_lst, dim=0)
        s_vec = s_vec.view(s_vec.shape[0] * s_vec.shape[1], 80, 80, 3)
        vs = model.v(s_vec).squeeze(1)

        a_vec = torch.stack(a_lst).reshape(-1).unsqueeze(1).to(device)
        advantage = td_target_vec.to(device) - vs
        print(f"reward: {r_lst}")

        mu, sigma  = model.pi(s_vec)
        dist = Normal(mu, sigma)
        log_probs = dist.log_prob(a_vec).squeeze(1)
        loss = -(log_probs * advantage.detach()).mean() +\
            F.smooth_l1_loss(model.v(s_vec).reshape(-1).to(device), td_target_vec.to(device))

        print("Calculating gradients...")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_idx % PRINT_INTERVAL == 0:
            print("Testing...")
            test(step_idx, model, world_test, car_test, display_test, camera_test, device)

    envs.close()