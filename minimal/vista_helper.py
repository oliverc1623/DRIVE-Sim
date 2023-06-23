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
    resized_obs = cv2.resize(cropped_obs, (80, 80))
    torch_obs = torch.from_numpy(resized_obs).to(torch.float32)
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
        # print("Out of lane")
        return True
    else:
        return False

def check_exceed_max_rot(car):
    # Max rotation is pi/10 = 18 degrees
    maximal_rotation = np.pi / 10.
    current_rotation = np.abs(car.relative_state.yaw)
    if current_rotation > maximal_rotation:
        print("Exceed max rotation")
        return True
    else:
        return False

def check_crash(car): 
    return check_out_of_lane(car) or check_exceed_max_rot(car) or car.done

def calculate_reward(car, prev_curvature):
    q_lat = np.abs(car.relative_state.x)
    maximal_rotation = np.pi / 10.
    current_rotation = np.abs(car.relative_state.yaw)
    print(f"prev curvature: {curvature2steering(prev_curvature, 2.78, 14.7)}")
    print(f"curvature: {curvature2steering(car.curvature, 2.78, 14.7)}")
    differential = -np.abs(car.curvature - prev_curvature)
    # print(f"difference: {differential}")

    road_width = car.trace.road_width
    z_lat = road_width / 2
    if q_lat > z_lat or current_rotation > maximal_rotation:
        return torch.tensor(0.0, dtype=torch.float32)
    else:
        lane_reward = torch.round(torch.tensor(1 - (q_lat/z_lat)**2, dtype=torch.float32), decimals=3)
        reward = lane_reward + differential
        return reward