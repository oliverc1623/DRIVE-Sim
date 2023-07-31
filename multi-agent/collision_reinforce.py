import os
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
from light_cnn import CNN

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging, misc
from vista.tasks import MultiAgentBase
from vista.utils import transform

device = "cpu"

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
    overlap = compute_overlap(poly, other_polys) / poly.area

    reward = lane_reward - overlap[0]
    return reward, {}

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
    optimizer.zero_grad()
    # Forward propagate through the agent network
    prediction = run_driving_model(observations)
    # back propagate
    neg_logprob = -1 * prediction.log_prob(actions)
    loss = (neg_logprob * discounted_rewards).mean()
    loss.backward()
    nn.utils.clip_grad_norm_(driving_model.parameters(), clip)
    optimizer.step()
    return loss.item()

def sample_actions(curvature_dist, world):
    actions = dict()
    for agent in world.agents:
        if agent.id != env.ego_agent.id:
            actions[agent.id] = np.array([0.0,0.0])
        else:
            curvature = curvature_dist.sample()[0,0]
            actions[agent.id] = np.array([curvature, agent.trace.f_speed(agent.timestamp)])
    return actions

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
        size=(200, 320),
        # rendering params
        depth_mode=DepthModes.FIXED_PLANE,
        use_lighting=False,
    )
]
task_config = dict(n_agents=2,
                    mesh_dir="carpack01",
                    init_dist_range=[15., 30.],
                    init_lat_noise_range=[-3., 3.],
                    reward_fn=my_reward_fn)
display_config = dict(road_buffer_size=1000, )

ego_car_config = copy.deepcopy(car_config)
ego_car_config['lookahead_road'] = True
trace_root = "trace"
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
driving_model = CNN()
learning_rate = 0.00005
episodes = 500
max_curvature, max_std = 1/8.0, 0.1
clip = 5
optimizer = optim.Adam(driving_model.parameters(), lr=learning_rate, weight_decay=1e-5)
# instantiate Memory buffer
memory = Memory()

## Driving training! Main training block. ##
max_batch_size = 300
best_reward = float("-inf")  # keep track of the maximum reward acheived during training

for i_episode in range(episodes):
    print(f"Episode: {i_episode}")
    env.world.set_seed(47) 
    observation = env.reset();
    display.reset()
    observation = grab_and_preprocess_obs(observation, env)
    steering_history = [0.0]
    driving_model.eval() # set to eval for inference loop
    steps = 0

    while True:
        curvature_dist = run_driving_model(driving_model, observation, max_curvature, max_std)
        actions = sample_actions(curvature_dist, env.world)
        observations, rewards, dones, infos = env.step(actions)

        steering = env.ego_agent.ego_dynamics.steering
        steering_history.append(steering)
        jitter_reward = calculate_jitter_reward(steering_history)
        observation = grab_and_preprocess_obs(observations, env)
        done = dones[env.ego_agent.id]
        reward = 0 if done else rewards[env.ego_agent.id] + jitter_reward
        curvature = actions[env.ego_agent.id][0]

        memory.add_to_memory(observation, curvature, reward)

        img = display.render()
        cv2.imshow("test", img[:, :, ::-1])
        key = cv2.waitKey(20)
        plt.pause(.05)
        if key == ord('q'):
            break

        if done:
            print(done)
            driving_model.train()
            total_reward = sum(memory.rewards)
            
            break