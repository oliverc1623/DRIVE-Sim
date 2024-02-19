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

    # Assuming 'image' is your numpy array of shape (64, 64, 3)
    image = obs
    
    # Apply transformations
    grayscale_image = to_grayscale(image)
    contrast_image = increase_contrast(image).astype(np.uint8)
    brightness_image = decrease_brightness(image)
    red_shift_image = color_shift_red(image)
    inverted_image = invert_colors(image)
    blue_shift_image = color_shift_blue(image)
    green_shift_image = color_shift_green(image)
    pruple_image = purple_tint(image)
    cyan_image = cyan_tint(image)
    yellow_image = yellow_tint(image)
    
    # Make sure to convert the grayscale image back to 3 channels if needed for consistency
    grayscale_image_3ch = np.stack((grayscale_image,)*3, axis=-1).astype(np.uint8)
    
    concatenated_images = np.concatenate(
        [obs, 
        grayscale_image_3ch, 
        contrast_image, 
        brightness_image, 
        red_shift_image, 
        inverted_image, 
        blue_shift_image,
        green_shift_image, 
        pruple_image,
        cyan_image,
        yellow_image], 1)
    
    plt.imshow(concatenated_images)
    plt.show()

if __name__ == "__main__":
    main()