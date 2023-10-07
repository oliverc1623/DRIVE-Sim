import torch
from vit_pytorch.vivit import ViT

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
import torch.nn as nn

import numpy as np

class SeqTransformer(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        
        self.v = nn.Sequential(
            ViT(
                image_size = 128,          # image size
                frames = 8,               # number of frames
                image_patch_size = 16,     # image patch size
                frame_patch_size = 2,      # frame patch size
                num_classes = features_dim,
                dim = 1024,
                spatial_depth = 6,         # depth of the spatial transformer
                temporal_depth = 6,        # depth of the temporal transformer
                heads = 8,
                mlp_dim = 2048
            )
        )
        # Compute shape by doing one forward pass
        # with th.no_grad():            
        #     n_flatten = self.v(
        #         self.reshape_ob(th.tensor(observation_space.sample()[None])/255).float()
        #     ).shape[1]

        # self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = self.reshape_ob(observations)
        return self.v(observations) # self.linear(self.v(observations))
    
    def reshape_ob(self, a: th.Tensor) -> th.Tensor:
        batch_size = a.shape[0]
        squares = []
        for i in range(batch_size):
            s0, s1, s2, s3, s4, s5, s6, s7 = a[0].split(128, 2)
            squares.append(torch.stack([s0, s1, s2, s3, s4, s5, s6, s7]))
        transposed_squares = [s.permute(1, 0, 2, 3) for s in squares]
        s = torch.stack(transposed_squares)
        return s