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

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.ReLU()
        )

        # LSTM layer
        self.lstm = nn.LSTM(features_dim, lstm_hidden_size, batch_first=True)
        self.lstm_hidden_size = lstm_hidden_size
        
        self._features_dim = lstm_hidden_size

    def forward(self, observations: th.Tensor, lstm_state: tuple = None, dones: th.Tensor = None) -> th.Tensor:
        # Prepare LSTM input
        batch_size, seq_len, h, w = observations.size()
        cnn_out = []
        for t in range(seq_len):
            single_frame = observations[:, t, :, :].unsqueeze(1)
            cnn_out.append(self.linear(self.cnn(single_frame)))
        cnn_out = th.stack(cnn_out, dim=1)
        
        if lstm_state is None:
            h0 = th.zeros((1, batch_size, self.lstm_hidden_size)).to(cnn_out.device)
            c0 = th.zeros((1, batch_size, self.lstm_hidden_size)).to(cnn_out.device)
        else:
            h0, c0 = lstm_state
        
        lstm_out, lstm_state = self.lstm(cnn_out, (h0, c0))
        
        # Output features for the policy
        return lstm_out[:, -1, :]
    
    @property
    def features_dim(self):
        return self._features_dim
