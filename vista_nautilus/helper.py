import vista
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from vista.entities.agents.Dynamics import curvature2steering

def preprocess(full_obs, camera):
    i1, j1, i2, j2 = camera.camera_param.get_roi()
    obs = full_obs[i1:i2, j1:j2]
    obs = obs / 255.
    return obs

def grab_and_preprocess_obs(car, camera):
    full_obs = car.observations[camera.name]
    cropped_obs = preprocess(full_obs, camera)
    torch_obs = torch.from_numpy(cropped_obs).to(torch.float32)
    return torch_obs

def vista_reset(world, display):
    world.reset()
    display.reset()

def vista_step(car, curvature=None, speed=None):
    if curvature is None: 
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None: 
        speed = car.trace.f_speed(car.timestamp)
    
    car.step_dynamics(action=np.array([curvature, speed]), dt=1/15.)
    car.step_sensors()

def check_out_of_lane(car):
    distance_from_center = np.abs(car.relative_state.x)
    road_width = car.trace.road_width 
    half_road_width = road_width / 2
    if distance_from_center > half_road_width:
        return True
    else:
        return False

def check_exceed_max_rot(car):
    maximal_rotation = np.pi / 10.
    current_rotation = np.abs(car.relative_state.yaw)
    if current_rotation > maximal_rotation:
        return True
    else:
        return False

def check_crash(car): 
    return check_out_of_lane(car) or check_exceed_max_rot(car) or car.done

def calculate_reward(car, curvature_action, prev_curvature):
    q_lat = np.abs(car.relative_state.x)
    road_width = car.trace.road_width
    z_lat = road_width / 2
    lane_reward = torch.round(torch.tensor(1 - (q_lat/z_lat)**2, dtype=torch.float32), decimals=3)
    differential = 0.0 if prev_curvature==0.0 else -np.abs(curvature_action - prev_curvature)
    reward = (lane_reward + differential) if not check_crash(car) else torch.tensor(0.0, dtype=torch.float32)
    if reward < 0:
        reward = torch.tensor(0.0, dtype=torch.float32)
    return reward