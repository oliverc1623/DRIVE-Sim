import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
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
        self.fc1_cnn = nn.Linear(2 * 80 * 200, 100)
        self.relu6 = nn.ReLU()
        self.fc2_cnn = nn.Linear(100, 100)

        self.lstm = nn.LSTM(input_size=100, hidden_size=64, num_layers=1)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, input_x):
        for t in range(input_x.size(1)):
            x = input_x[:, t, :, :, :]
            x = self.relu1(self.norm1(self.conv1(x)))
            x = self.relu2(self.norm2(self.conv2(x)))
            x = self.relu3(self.norm3(self.conv3(x)))
            x = self.relu4(self.norm4(self.conv4(x)))
            x = self.relu5(self.norm5(self.conv5(x)))
            x = x.reshape(x.size(0), -1)
            x = self.relu6(self.fc1_cnn(x))
            x = self.fc2_cnn(x)
            out, hidden = self.lstm(x.unsqueeze(0))
        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        mu, sigma = torch.chunk(x, 2, dim=-1)
        return mu, sigma