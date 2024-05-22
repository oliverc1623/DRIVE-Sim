import torch as th
import torch.nn as nn
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer

class LSTMCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64, lstm_hidden_size: int = 256):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.zeros((1, 1, 84, 84))
            ).shape[1]

        # LSTM layer
        self.lstm = nn.LSTM(lstm_hidden_size, features_dim)
        self.lstm_hidden_size = lstm_hidden_size
        self._features_dim = features_dim
        # Linear layer
        self.linear = nn.Sequential(
            layer_init(nn.Linear(features_dim, features_dim)), 
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        hidden = None
        batch_size, seq_len, h, w = observations.size()
        for t in range(seq_len):
            single_frame = observations[:, t, :, :].unsqueeze(1)
            x = self.cnn(single_frame).unsqueeze(0)
            lstm_out, hidden = self.lstm(x, hidden)
        out = self.linear(hidden[0][-1]) 
        return out
    
    @property
    def features_dim(self):
        return self._features_dim
