import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class LSTMCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64, lstm_hidden_size: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=5, stride=2, padding=1),
            nn.GroupNorm(128, 256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=1),
            nn.GroupNorm(128, 256),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.zeros((1, 1, 84, 84))
            ).shape[1]
        # LSTM layer
        self.lstm = nn.LSTM(n_flatten, lstm_hidden_size, num_layers=2)
        self.lstm_hidden_size = lstm_hidden_size
        self._features_dim = lstm_hidden_size
        # Linear layer
        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim), 
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
