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
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.zeros((1, 1, 84, 84))
            ).shape[1]
        # LSTM layer
        self.lstm = nn.LSTM(n_flatten, lstm_hidden_size, batch_first=True)
        self.lstm_hidden_size = lstm_hidden_size
        self._features_dim = lstm_hidden_size
        # Linear layer
        self.linear = nn.Sequential(
            nn.Linear(features_dim, features_dim), 
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Prepare LSTM input
        batch_size, seq_len, h, w = observations.size()
        cnn_out = []
        for t in range(seq_len):
            single_frame = observations[:, t, :, :].unsqueeze(1)
            cnn_out.append(self.cnn(single_frame))
        cnn_out = th.stack(cnn_out, dim=1)
        lstm_out, lstm_state = self.lstm(cnn_out)
        out = self.linear(lstm_out[:, -1, :]) 
        return out
    
    @property
    def features_dim(self):
        return self._features_dim
