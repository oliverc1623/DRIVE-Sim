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
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 256, kernel_size=5, stride=2, padding=1),
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
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        # LSTM layer
        self.lstm = nn.LSTM(features_dim, lstm_hidden_size, batch_first=True)
        self.lstm_hidden_size = lstm_hidden_size
        self.features_dim = lstm_hidden_size

    def forward(self, observations: th.Tensor, lstm_state: tuple = None, dones: th.Tensor = None) -> th.Tensor:
        cnn_out = self.linear(self.cnn(observations))
        
        # Prepare LSTM input
        batch_size, seq_len, _ = cnn_out.size()
        lstm_in = cnn_out.view(batch_size, seq_len, -1)
        
        if lstm_state is None:
            h0 = th.zeros((1, batch_size, self.lstm_hidden_size)).to(cnn_out.device)
            c0 = th.zeros((1, batch_size, self.lstm_hidden_size)).to(cnn_out.device)
        else:
            h0, c0 = lstm_state
        
        lstm_out, lstm_state = self.lstm(lstm_in, (h0, c0))
        
        # Output features for the policy
        return lstm_out[:, -1, :], lstm_state
