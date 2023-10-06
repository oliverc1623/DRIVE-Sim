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
        # print(f"ob space: {observation_space}")
        
        self.v = nn.Sequential(
            ViT(
                image_size = 128,          # image size
                frames = 4,               # number of frames
                image_patch_size = 16,     # image patch size
                frame_patch_size = 2,      # frame patch size
                num_classes = 64,
                dim = 64,
                spatial_depth = 6,         # depth of the spatial transformer
                temporal_depth = 6,        # depth of the temporal transformer
                heads = 8,
                mlp_dim = 64
            ),
            nn.ReLU(),
            nn.Flatten()
        )

        # print(f"reshape: {self.reshape_ob(th.as_tensor(observation_space.sample()[None])).shape}")

        # Compute shape by doing one forward pass
        with th.no_grad():            
            n_flatten = self.v(
                self.reshape_ob(th.as_tensor(observation_space.sample()[None])).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = self.reshape_ob(observations)
        # print(f"ob shape: {observations.shape}")
        return self.linear(self.v(observations))
    
    def reshape_ob(self, observations: th.Tensor) -> th.Tensor:
        num_groups = 3
        channels_per_group = observations.shape[1] // num_groups

        # Reshape the image matrix
        target_shape = (observations.shape[0], num_groups, channels_per_group, observations.shape[2], observations.shape[3])
        reshaped_observations = observations.view(*target_shape)
        # print(f"reshaped obs: {reshaped_observations.shape}")
        return reshaped_observations