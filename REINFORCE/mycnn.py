import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.norm5(x)
        x = self.relu5(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        mu, log_sigma = torch.chunk(x, 2, dim=-1)
        return mu, log_sigma