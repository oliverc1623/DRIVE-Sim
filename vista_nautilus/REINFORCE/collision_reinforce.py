import os
os.environ["DISPLAY"] = ":1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import box as Box
from shapely import affinity
from typing import List
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.insert(1, '../vista_nautilus/models/')
import mycnn
import datetime

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging, misc
from vista.tasks import MultiAgentBase
from vista.utils import transform

import sys
sys.path.insert(1, '../vista_nautilus/')
from helper import * 

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        size=(400, 640), # (200, 320) for lighter cnn 
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
# display = vista.Display(env.world, display_config=display_config)

# start env
env.reset();
# display.reset()  # reset should be called after env reset

## Training parameters and initialization ##
driving_model = mycnn.CNN(70, 310).to(device)
learning_rate = 0.0005
episodes = 500
max_curvature, max_std = 1/8.0, 0.01
clip = 100
optimizer = optim.Adam(driving_model.parameters(), lr=learning_rate)
# instantiate Memory buffer
memory = Memory()

## Driving training! Main training block. ##
max_batch_size = 300
best_reward = float("-inf")  # keep track of the maximum reward acheived during training

# file to log progress
f = write_file("collision6")
# frame_dir = save_as_video()

for i_episode in range(episodes):
    print(f"Episode: {i_episode}")
    observation = env.reset();
    # display.reset()
    trace_index = env.ego_agent.trace_index
    observation = grab_and_preprocess_obs(observation, env, device)
    steering_history = [0.0, env.ego_agent.ego_dynamics.steering]
    steps = 0
    initial_frame = env.ego_agent.frame_index
    memory.add_to_memory(observation, torch.tensor(0.0), 1.0)

    while True:
        curvature_dist = run_driving_model(driving_model, observation, max_curvature, max_std)
        actions = sample_actions(curvature_dist, env.world, env.ego_agent.id)
        observations, rewards, dones, infos = env.step(actions)
        reward = rewards[env.ego_agent.id][0]
        terminal_conditions = rewards[env.ego_agent.id][1]

        steering = env.ego_agent.ego_dynamics.steering
        steering_history.append(steering)
        jitter_reward = calculate_jitter_reward(steering_history)
        observation = grab_and_preprocess_obs(observations, env, device)
        done = terminal_conditions['done']
        reward = 0.0 if done else reward + jitter_reward
        if reward < 0.0:
            reward = 0.0
        curvature = actions[env.ego_agent.id][0]
        memory.add_to_memory(observation, torch.tensor(curvature,dtype=torch.float32), reward)
        steps +=1

        if done:
            total_reward = sum(memory.rewards)
            progress = calculate_progress(env, initial_frame)
            terminal_condition = ""
            for key, value in terminal_conditions.items():
                if value:
                    terminal_condition = key
                    print(f"{key}: {value}")
            print(f"total reward: {total_reward}")
            print(f"Progress: {progress*100:.2f}%")
            print(f"steps: {steps}\n")

            batch_size = min(len(memory), max_batch_size)
            i = torch.randperm(len(memory))[:batch_size].to(device)

            batch_observations = torch.stack(memory.observations, dim=0)
            batch_observations = torch.index_select(batch_observations, dim=0, index=i)

            batch_actions = torch.stack(memory.actions).to(device)
            batch_actions = torch.index_select(batch_actions, dim=0, index=i)

            batch_rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device)
            batch_rewards = discount_rewards(batch_rewards)[i]

            train_step(
                driving_model,
                optimizer,
                observations=batch_observations,
                actions=batch_actions,
                discounted_rewards=batch_rewards,
                clip=clip
            )

            # Write reward and loss to results txt file
            f.write(f"{total_reward}\t{steps}\t{progress}\t{trace_index}\t{terminal_condition}\n")
            f.flush()
            # reset the memory
            memory.clear()
            break