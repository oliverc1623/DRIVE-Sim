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

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

### Agent Memory ###
class Memory:
    def __init__(self):
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

    def __len__(self):
        return len(self.actions)

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

def grab_and_preprocess_obs(observation, env):
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

def sample_actions(curvature_dist, world):
    actions = dict()
    for agent in world.agents:
        if agent.id != env.ego_agent.id:
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
    f.write("reward\tsteps\tprogress\ttrace\tterminal_condition\n")
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
        size=(355, 413), # (200, 320) for lighter cnn
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
display.reset()  # reset should be called after env reset

## Training parameters and initialization ##
driving_model = mycnn.CNN(60, 200).to(device)
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
    display.reset()
    trace_index = env.ego_agent.trace_index
    observation = grab_and_preprocess_obs(observation, env)
    steering_history = [0.0, env.ego_agent.ego_dynamics.steering]
    driving_model.eval() # set to eval for inference loop
    steps = 0
    initial_frame = env.ego_agent.frame_index
    memory.add_to_memory(observation, torch.tensor(0.0), 1.0)

    while True:
        curvature_dist = run_driving_model(driving_model, observation, max_curvature, max_std)
        actions = sample_actions(curvature_dist, env.world)
        observations, rewards, dones, infos = env.step(actions)
        reward = rewards[env.ego_agent.id][0]
        terminal_conditions = rewards[env.ego_agent.id][1]

        steering = env.ego_agent.ego_dynamics.steering
        steering_history.append(steering)
        jitter_reward = calculate_jitter_reward(steering_history)
        observation = grab_and_preprocess_obs(observations, env)
        done = terminal_conditions['done']
        reward = 0.0 if done else reward + jitter_reward
        if reward < 0.0:
            reward = 0.0
        curvature = actions[env.ego_agent.id][0]

        memory.add_to_memory(observation, torch.tensor(curvature,dtype=torch.float32), reward)
        steps +=1

        if done:
            driving_model.train()
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