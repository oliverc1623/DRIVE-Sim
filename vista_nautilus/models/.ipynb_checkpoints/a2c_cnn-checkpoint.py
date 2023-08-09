import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(8, 24)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(24, 36, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(9, 36)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(36, 48, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.GroupNorm(12, 48)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.GroupNorm(64, 64)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.GroupNorm(1, 2)
        self.relu5 = nn.ReLU()
        self.fc1 = nn.Linear(2 * 70 * 310, 100)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.relu7 = nn.ReLU() 
        self.fc3 = nn.Linear(100, 2)
        self.fc_v = nn.Linear(100, 1)

    def pi(self, x):
        single_image_input = len(x.shape) == 3  # missing 4th batch dimension
        if single_image_input:
            x = x.unsqueeze(0)
        else:
            x = x.permute(0, 3, 1, 2)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.relu5(self.norm5(self.conv5(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu6(self.fc1(x))
        x = self.relu7(self.fc2(x))
        x = self.fc3(x)
        mu, log_sigma = torch.chunk(x, 2, dim=-1)
        mu = 1/8.0 * torch.tanh(mu)  # conversion
        sigma = 0.1 * torch.sigmoid(log_sigma) + 0.005  # conversion
        return mu, sigma
    
    def v(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.relu5(self.norm5(self.conv5(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu6(self.fc1(x))
        x = self.relu7(self.fc2(x))
        v = self.fc_v(x)
        return v