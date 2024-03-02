import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper, ReseedWrapper
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import seaborn as sns
# sns.set(rc={'figure.figsize':(11.7,8.27)})
import pandas as pd
from IPython.display import Video
import matplotlib.pyplot as plt
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta
from torch.autograd import Function
from utils import reparameterize

device = "cuda:0"

def to_grayscale(image):
    # Convert to grayscale using the luminosity method
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

def increase_contrast(image, factor=1.5):
    # Increase contrast by scaling the difference from the mean
    mean = np.mean(image)
    return np.clip((1 + factor) * (image - mean) + mean, 0, 255)

def decrease_brightness(image, factor=50):
    # Decrease brightness
    return np.clip(image - factor, 0, 255)

def color_shift_red(image, factor=50):
    # Shift color towards red
    red_channel = image[:,:,0] + factor
    red_channel[red_channel > 255] = 255
    image = np.dstack((red_channel, image[:,:,1], image[:,:,2]))
    image[48:56, 48:56, 0] = 222
    image[48:56, 48:56, 1] = 255
    image[48:56, 48:56, 2] = 239
    return image

def invert_colors(image):
    # Invert image colors
    return 255 - image

def color_shift_blue(image, factor=50):
    # Shift color towards blue
    blue_channel = image[:,:,2] + factor
    blue_channel[blue_channel > 255] = 255
    return np.dstack((image[:,:,0], image[:,:,1], blue_channel))

def color_shift_green(image, factor=50):
    # Shift color towards green
    green_channel = image[:,:,1] + factor
    green_channel[green_channel > 255] = 255
    image = np.dstack((image[:,:,0], green_channel, image[:,:,2]))
    image[48:56, 48:56, 0] = 0
    image[48:56, 48:56, 1] = 0
    image[48:56, 48:56, 2] = 255
    return image

def purple_tint(image, factor=50):
    # Apply a purple tint (red + blue)
    red_channel = image[:,:,0] + factor
    red_channel[red_channel > 255] = 255
    blue_channel = image[:,:,2] + factor
    blue_channel[blue_channel > 255] = 255
    image = np.dstack((red_channel, image[:,:,1], blue_channel))
    image[48:56, 48:56, 0] = 255
    image[48:56, 48:56, 1] = 255
    image[48:56, 48:56, 2] = 153
    return image

def cyan_tint(image, factor=50):
    # Apply a cyan tint (green + blue)
    green_channel = image[:,:,1] + factor
    green_channel[green_channel > 255] = 255
    blue_channel = image[:,:,2] + factor
    blue_channel[blue_channel > 255] = 255
    image = np.dstack((image[:,:,0], green_channel, blue_channel))
    image[48:56, 48:56, 0] = 255
    image[48:56, 48:56, 1] = 132
    image[48:56, 48:56, 2] = 0
    return image

def yellow_tint(image, factor=50):
    # Apply a yellow tint (red + green)
    red_channel = image[:,:,0] + factor
    red_channel[red_channel > 255] = 255
    green_channel = image[:,:,1] + factor
    green_channel[green_channel > 255] = 255
    image = np.dstack((red_channel, green_channel, image[:,:,2]))
    image[48:56, 48:56, 0] = 255
    image[48:56, 48:56, 1] = 201
    image[48:56, 48:56, 2] = 248
    return image

def main():
    env = gym.make('MiniGrid-Empty-8x8-v0', render_mode="rgb_array")
    env = RGBImgObsWrapper(env)
    obs, info = env.reset() # This now produces an RGB tensor only
    obs = obs["image"]

    ################## Original Domain ##################
    data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid1/0.npz", obs=data_array)
    
    data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid1/1.npz", obs=data_array)
    
    data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid1/2.npz", obs=data_array)
    
    data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid1/3.npz", obs=data_array)
    
    data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid1/4.npz", obs=data_array)
    
    data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid1/5.npz", obs=data_array)
    
    data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid1/6.npz", obs=data_array)
    
    data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid1/7.npz", obs=data_array)
    
    data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid1/8.npz", obs=data_array)
    
    data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid1/9.npz", obs=data_array)


    ################## Green Domain ##################
    red_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_green(obs)
        red_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid2/0.npz", obs=red_data_array)
    
    red_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_green(obs)
        red_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid2/1.npz", obs=red_data_array)
    
    red_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_green(obs)
        red_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid2/2.npz", obs=red_data_array)
    
    red_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_green(obs)
        red_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid2/3.npz", obs=red_data_array)
    
    red_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_green(obs)
        red_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid2/4.npz", obs=red_data_array)
    
    red_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_green(obs)
        red_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid2/5.npz", obs=red_data_array)
    
    red_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_green(obs)
        red_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid2/6.npz", obs=red_data_array)
    
    red_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_green(obs)
        red_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid2/7.npz", obs=red_data_array)
    
    red_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_green(obs)
        red_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid2/8.npz", obs=red_data_array)
    
    red_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_green(obs)
        red_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid2/9.npz", obs=red_data_array)

    ################## Red Domain ##################
    
    cyan_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_red(obs, 109)
        cyan_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid3/0.npz", obs=cyan_data_array)
    
    cyan_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_red(obs, 109)
        cyan_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid3/1.npz", obs=cyan_data_array)
    
    cyan_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_red(obs, 109)
        cyan_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid3/2.npz", obs=cyan_data_array)
    
    cyan_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_red(obs, 109)
        cyan_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid3/3.npz", obs=cyan_data_array)
    
    cyan_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_red(obs, 109)
        cyan_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid3/4.npz", obs=cyan_data_array)
    
    cyan_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_red(obs, 109)
        cyan_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid3/5.npz", obs=cyan_data_array)
    
    cyan_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_red(obs, 109)
        cyan_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid3/6.npz", obs=cyan_data_array)
    
    cyan_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_red(obs, 109)
        cyan_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid3/7.npz", obs=cyan_data_array)
    
    cyan_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_red(obs, 109)
        cyan_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid3/8.npz", obs=cyan_data_array)
    
    cyan_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_red(obs, 109)
        cyan_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid3/9.npz", obs=cyan_data_array)

    ################## Blue Domain ##################

    pruple_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_blue(obs)
        pruple_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid4/0.npz", obs=pruple_data_array)
    
    pruple_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_blue(obs)
        pruple_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid4/1.npz", obs=pruple_data_array)
    
    pruple_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_blue(obs)
        pruple_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid4/2.npz", obs=pruple_data_array)
    
    pruple_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_blue(obs)
        pruple_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid4/3.npz", obs=pruple_data_array)
    
    pruple_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_blue(obs)
        pruple_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid4/4.npz", obs=pruple_data_array)
    
    pruple_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_blue(obs)
        pruple_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid4/5.npz", obs=pruple_data_array)
    
    pruple_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_blue(obs)
        pruple_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid4/6.npz", obs=pruple_data_array)
    
    pruple_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_blue(obs)
        pruple_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid4/7.npz", obs=pruple_data_array)
    
    pruple_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_blue(obs)
        pruple_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid4/8.npz", obs=pruple_data_array)
    
    pruple_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(timestamps)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = color_shift_blue(obs)
        pruple_data_array[t] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid4/9.npz", obs=pruple_data_array)

    ################## Green Domain ##################
    green_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(1, timestamps + 1)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = yellow_tint(obs, 109)
        green_data_array[t-1] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid5/0.npz", obs=green_data_array)
    
    green_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(1, timestamps + 1)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = yellow_tint(obs, 109)
        green_data_array[t-1] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid5/1.npz", obs=green_data_array)
    
    green_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(1, timestamps + 1)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = yellow_tint(obs, 109)
        green_data_array[t-1] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid5/2.npz", obs=green_data_array)
    
    green_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(1, timestamps + 1)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = yellow_tint(obs, 109)
        green_data_array[t-1] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid5/3.npz", obs=green_data_array)
    
    green_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(1, timestamps + 1)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = yellow_tint(obs, 109)
        green_data_array[t-1] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid5/4.npz", obs=green_data_array)
    
    green_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(1, timestamps + 1)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = yellow_tint(obs, 109)
        green_data_array[t-1] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid5/5.npz", obs=green_data_array)
    
    green_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(1, timestamps + 1)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = yellow_tint(obs, 109)
        green_data_array[t-1] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid5/6.npz", obs=green_data_array)
    
    
    green_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(1, timestamps + 1)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = yellow_tint(obs, 109)
        green_data_array[t-1] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid5/7.npz", obs=green_data_array)
    
    green_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(1, timestamps + 1)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = yellow_tint(obs, 109)
        green_data_array[t-1] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid5/8.npz", obs=green_data_array)
    
    green_data_array = np.zeros((10_000, 64, 64, 3)).astype(np.uint8)
    timestamps = 10_000
    obs, info = env.reset()
    for t in tqdm(range(1, timestamps + 1)):
        action = np.random.randint(3)
        obs, r, done, _, _ = env.step(action)
        obs = obs['image']
        obs = yellow_tint(obs, 109)
        green_data_array[t-1] = obs.astype(np.uint8)
        if done:
            obs, info = env.reset()
    np.savez("data/minigrid8x8/grid5/9.npz", obs=green_data_array)


if __name__ == "__main__":
    main()