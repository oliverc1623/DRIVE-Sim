import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 24, kernel_size=5, stride=2, padding=1),
            nn.GroupNorm(8, 24),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=1),
            nn.GroupNorm(9, 36),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(12, 48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(64, 64),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
