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

# ### Agent Memory ###
# class Memory:
#     def __init__(self):
#         self.clear()

#     # Resets/restarts the memory buffer
#     def clear(self):
#         self.observations = []
#         self.actions = []
#         self.rewards = []

#     # Add observations, actions, rewards to memory
#     def add_to_memory(self, new_observation, new_action, new_reward):
#         self.observations.append(new_observation)
#         self.actions.append(new_action)
#         self.rewards.append(new_reward)

#     def __len__(self):
#         return len(self.actions)

def compute_overlap(poly: Box, polys: List[Box]) -> List[float]:
    n_polys = len(polys)
    overlap = np.zeros((n_polys))
    for i in range(n_polys):
        intersection = polys[i].intersection(poly)
        overlap[i] = intersection.area
    return overlap

def my_reward_fn(task, agent_id, **kwargs):
    agent = [_a for _a in task.world.agents if _a.id == agent_id][0]
    other_agents = [_a for _a in task.world.agents if _a.id != agent_id]

    # Lane reward
    q_lat = np.abs(agent.relative_state.x)
    road_width = agent.trace.road_width
    z_lat = road_width / 2
    lane_reward = round(1 - (q_lat/z_lat)**2, 4)

    # collision avoidance reward
    agent2poly = lambda _x: misc.agent2poly(
        _x, ref_dynamics=agent.human_dynamics)
    poly = agent2poly(agent).buffer(5)
    other_polys = list(map(agent2poly, other_agents))
    overlap = (compute_overlap(poly, other_polys) / poly.area) * 10

    reward = lane_reward - overlap[0]
    return (reward, kwargs), {}

def calculate_jitter_reward(steering_history):
    first_derivative = np.gradient(steering_history)
    second_derivative = np.gradient(first_derivative)
    jitter_reward = -np.abs(second_derivative[-1])
    return jitter_reward

def vista_step(car, curvature=None, speed=None):
    if curvature is None:
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None:
        speed = car.trace.f_speed(car.timestamp)

    car.step_dynamics(action=np.array([curvature, speed]), dt=1/15.0)
    car.step_sensors()

def preprocess(full_obs, env):
    # Extract ROI
    i1, j1, i2, j2 = env.ego_agent.sensors[0].camera_param.get_roi()
    # cropping hood out of ROI
    i1 += 10
    i2 -=10
    obs = full_obs[i1:i2, j1:j2]
    return obs

def grab_and_preprocess_obs(observation, env, device):
    observation = observation[env.ego_agent.id]['camera_front']
    cropped_obs = preprocess(observation, env)
    normalized_cropped = cropped_obs / 255.0
    return torch.from_numpy(normalized_cropped).to(torch.float32).to(device)

## The self-driving learning algorithm ##
def run_driving_model(driving_model, image, max_curvature, max_std):
    single_image_input = len(image.shape) == 3  # missing 4th batch dimension
    if single_image_input:
        image = image.unsqueeze(0)
    image = image.permute(0,3,1,2)
    mu, logsigma = driving_model(image)
    mu = max_curvature * torch.tanh(mu)  # conversion
    sigma = max_std * torch.sigmoid(logsigma) + 0.005  # conversion
    pred_dist = dist.Normal(mu, sigma)
    return pred_dist

### Training step (forward and backpropagation) ###
def train_step(driving_model, optimizer, observations, actions, discounted_rewards, clip):
    max_curvature, max_std = 1/8.0, 0.01
    optimizer.zero_grad()
    # Forward propagate through the agent network
    prediction = run_driving_model(driving_model, observations, max_curvature, max_std)
    # back propagate
    neg_logprob = -1 * prediction.log_prob(actions)
    loss = (neg_logprob * discounted_rewards).mean()
    loss.backward()
    nn.utils.clip_grad_norm_(driving_model.parameters(), clip)
    optimizer.step()

def sample_actions(curvature_dist, world, ego_id):
    actions = dict()
    for agent in world.agents:
        if agent.id != ego_id:
            actions[agent.id] = np.array([0.0,0.0])
        else:
            curvature = curvature_dist.sample()[0,0].cpu()
            actions[agent.id] = np.array([curvature, agent.trace.f_speed(agent.timestamp)])
    return actions

def normalize(x):
    x -= torch.mean(x)
    x /= torch.std(x)
    return x

# Compute normalized, discounted, cumulative rewards (i.e., return)
def discount_rewards(rewards, gamma=0.95):
    discounted_rewards = torch.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        # update the total discounted reward
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
    return normalize(discounted_rewards)

def write_file(filename):
    # open and write to file to track progress
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = "results"
    model_results_dir = results_dir + "/CNN/" # TODO: make this into variable

    if not os.path.exists(model_results_dir):
        os.makedirs(model_results_dir)
    # Define the file path
    print("Writing log to: " + filename + ".txt")
    file_path = os.path.join(model_results_dir, filename + ".txt")
    f = open(file_path, "w")
    f.write("reward\tsteps\tprogress\ttrace\n")
    return f

def save_as_video():
    frames_dir = "frames"
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    model_frame_dir = (
        frames_dir + f"/CNN_frames_{timestamp}/"
    )
    if not os.path.exists(model_frame_dir):
        os.makedirs(model_frame_dir)
    return model_frame_dir

def calculate_progress(env, initial_frame):
    total_frames = len(env.ego_agent.trace.good_frames['camera_front'][0])
    track_left = total_frames - initial_frame
    progress = env.ego_agent.frame_index - initial_frame
    progress_percentage = np.round(progress/track_left, 4)
    return progress_percentage
