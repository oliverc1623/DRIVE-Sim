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


def introspect(
    state,
    teacher_source_policy,
    teacher_target_policy,
    t,
    inspection_threshold=0.9,
    introspection_decay=0.99999,
    burn_in=0,
):
    h = 0
    probability = introspection_decay**(max(0, t - burn_in))
    p = Bernoulli(probability).sample()
    if t > burn_in and p == 1:
        _, _, teacher_source_val = teacher_source_policy.act(state)
        _, _, teacher_target_val = teacher_target_policy.act(state)
        if abs(teacher_target_val - teacher_source_val) <= inspection_threshold:
            h = 1
    return h
