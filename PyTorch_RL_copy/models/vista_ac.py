import torch.nn as nn
import torch
from utils.math import *

class VistaAC(nn.Module):
    def __init__(self):
        super(VistaAC, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(8, 32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(16, 64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.GroupNorm(32, 128)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.GroupNorm(64, 256)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.GroupNorm(1, 2)
        self.relu5 = nn.ReLU()
        self.fc = nn.Linear(2 * 32 * 30, 2)
        self.fc_v = nn.Linear(3 * 32 * 20, 1)

        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # self.optimization_step = 0

    def pi(self, x):
        single_image_input = len(x.shape) == 3  # missing 4th batch dimension
        if single_image_input:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.relu5(self.norm5(self.conv5(x)))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        mu, log_sigma = torch.chunk(x, 2, dim=-1)
        mu = 1/8.0 * torch.tanh(mu)  # conversion
        sigma = 0.1 * torch.sigmoid(log_sigma) + 0.005  # conversion
        return mu, sigma
    
    def v(self, x):
        x = x.permute(0,3,1,2)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.relu5(self.norm5(self.conv5(x)))
        x = x.reshape(x.size(0), -1)
        v = self.fc_v(x)
        return v