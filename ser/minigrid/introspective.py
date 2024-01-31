import os
import glob
import time
from datetime import datetime

import torch
from torch.distributions import Bernoulli
import numpy as np

import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
from IntrospectiveEnv import IntrospectiveEnv

from PPO import PPO

def introspective(state, teacher_policy, student_policy, t):
    h = 0
    burn_in = 0
    inspection_threshold = 0.9
    introspection_decay = 0.99999
    
    return h
    