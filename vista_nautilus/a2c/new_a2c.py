import os
os.environ["DISPLAY"] = ":1.0"
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import torch.multiprocessing as mp

import vista
from vista_helper import *
from vista.entities.agents.Dynamics import curvature2tireangle
from vista.entities.agents.Dynamics import curvature2steering
import sys
sys.path.insert(1, '../vista_nautilus/models/')
sys.path.insert(1, '../vista_nautilus/')
from helper import * 
import a2c_cnn

import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import cv2

# Hyperparameters
# TODO: make hyperparameters args
n_train_processes = 3
learning_rate = 0.0005
update_interval = 10 # 100
gamma = 0.95
max_train_steps = 60000
PRINT_INTERVAL = 5

def worker(worker_id, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
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

    steering_history = [0.0, env.ego_agent.ego_dynamics.steering]
    
    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            ego_action = data.item()         
            actions = dict()
            for agent in env.world.agents:
                if agent.id != env.ego_agent.id:
                    actions[agent.id] = np.array([0.0,0.0])
                else:
                    actions[agent.id] = np.array([ego_action, agent.trace.f_speed(agent.timestamp)])
            observations, rewards, dones, infos = env.step(actions)
            
            terminal_conditions = rewards[env.ego_agent.id][1]
            steering = env.ego_agent.ego_dynamics.steering
            steering_history.append(steering)
            jitter_reward = calculate_jitter_reward(steering_history)
            ob = grab_and_preprocess_obs(observations, env, device)
            done = int(terminal_conditions['done'])
            reward = 0.0 if done else rewards[env.ego_agent.id][0] + jitter_reward
            if reward < 0.0:
                reward = 0.0
            if done:
                ob = env.reset();
                ob = grab_and_preprocess_obs(ob, env, device)
                steering_history = [0.0, env.ego_agent.ego_dynamics.steering]
            worker_end.send((ob, torch.tensor(reward), torch.tensor(done)))
        elif cmd == 'reset':
            ob = env.reset();
            ob = grab_and_preprocess_obs(ob, env, device)
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
        torch_obs = torch.stack(obs)
        torch_rews = torch.stack(rews)
        torch_dones = torch.stack(dones)
        return torch_obs, torch_rews, torch_dones

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

def test(step_idx, test_env, model, device):
    print("Testing...")
    score = 0.0
    total_steps = 0
    num_test = 5
    total_progress = 0.0
    done = False
    for i in range(num_test):
        print(f"num_test: {i}")
        step = 0
        time.sleep(1)
        ob = test_env.reset();
        trace_index = test_env.ego_agent.trace_index
        initial_frame = test_env.ego_agent.frame_index
        observation = grab_and_preprocess_obs(ob, test_env, device)
        steering_history = [0.0, test_env.ego_agent.ego_dynamics.steering]
        while not done:
            mu, sigma = model.pi(observation.permute(2,0,1).to(device))
            dist = Normal(mu, sigma)
            actions = sample_actions(dist, test_env.world, test_env.ego_agent.id)
            observations, rewards, dones, infos = test_env.step(actions)
            reward = rewards[test_env.ego_agent.id][0]
            terminal_conditions = rewards[test_env.ego_agent.id][1]
            steering = test_env.ego_agent.ego_dynamics.steering
            steering_history.append(steering)
            jitter_reward = calculate_jitter_reward(steering_history)
            observation = grab_and_preprocess_obs(observations, test_env, device)
            done = terminal_conditions['done']
            reward = 0.0 if done else reward + jitter_reward
            if reward < 0.0:
                reward = 0.0
            score += reward
            step += 1
        print(f"step: {step}")
        total_steps += step
        done = False
        progress = calculate_progress(test_env, initial_frame)
        total_progress += progress
    print(f"Step # :{step_idx}, avg score : {score/num_test:.1f}, avg progress: {total_progress/num_test}\n")
    # return avg score, avg steps, avg progress, trace_index
    return score/num_test, total_steps/num_test, total_progress/num_test, trace_index

def compute_target(v_final, r_lst, mask_lst):
    G = v_final.reshape(-1)
    td_target = list()

    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r.numpy() + gamma * G * mask.numpy()
        td_target.append(G)

    return torch.tensor(np.array(td_target[::-1])).float()

if __name__ == '__main__':
    # Set the start method to 'spawn'
    mp.set_start_method('spawn')
    
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:{}'.format(device))
    envs = ParallelEnv(n_train_processes)
    
    ### VISTA World for testing
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
    test_env = MultiAgentBase(trace_paths=trace_path,
                        trace_config=trace_config,
                        car_configs=[car_config] * task_config['n_agents'],
                        sensors_configs=[sensors_config] + [[]] *
                        (task_config['n_agents'] - 1),
                        task_config=task_config)
    ###
    
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    model = a2c_cnn.ActorCritic().to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    f = write_file("a2c_cnn")
    
    step_idx = 0
    episode = 0
    s = envs.reset().to(device)
    best_reward = float("-inf")
    while step_idx < max_train_steps:
        print(f"step indx: {step_idx}")
        s_lst, a_lst, r_lst, mask_lst, s_primes = list(), list(), list(), list(), list()
        for _ in range(update_interval):
            mu0, sigma0 = model.pi(s[0].permute(2,0,1).to(device))
            mu1, sigma1 = model.pi(s[1].permute(2,0,1).to(device))
            mu2, sigma2 = model.pi(s[2].permute(2,0,1).to(device))
            dist0 = Normal(mu0, sigma0)
            dist1 = Normal(mu1, sigma1)
            dist2 = Normal(mu2, sigma2)
            a0 = dist0.sample()
            a1 = dist1.sample()
            a2 = dist2.sample()
            a = torch.tensor([a0, a1, a2])
            s_prime, r, done = envs.step(a)
            print(r)
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            mask_lst.append(1 - done)

            s = s_prime
            step_idx += 1

        s_final = s_prime 
        v_final = model.v(s_final.to(device)).detach().cpu().clone().numpy()
        td_target = compute_target(v_final, r_lst, mask_lst)
        td_target_vec = td_target.reshape(-1)
        s_lst = [x.to(device) for x in s_lst]
        s_vec = torch.stack(s_lst, dim=0)
        s_vec = s_vec.view(s_vec.shape[0] * s_vec.shape[1], 70, 310, 3)
        vs = model.v(s_vec).squeeze(1)

        a_vec = torch.stack(a_lst).reshape(-1).unsqueeze(1).to(device)
        advantage = td_target_vec.to(device) - vs

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
            avg_score, avg_steps, avg_progress, trace_index = test(step_idx, test_env, model, device)
            # Write reward and loss to results txt file
            f.write(f"{avg_score}\t{avg_steps}\t{avg_progress}\t{trace_index}\n")
            f.flush()
            episode += 1
            if avg_score > best_reward:
                best_reward = avg_score
                print("Saving and exporting model...")
                checkpoint = {
                    'epoch': step_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': avg_score,
                }
                torch.save(checkpoint, f"saved_models/a2c_cnn_model_{timestamp}.pth")
        time.sleep(1)