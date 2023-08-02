import torch
import torch.nn as nn
import torchvision.models as models

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()

        # Load the pre-trained ResNet50 model
        self.resnet34 = models.resnet34()

        # Replace the last fully connected layer to output two numerical values
        num_ftrs = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x = self.resnet34(x)
        mu, log_sigma = torch.chunk(x, 2, dim=-1)
        return mu, log_sigma
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        # Load the pre-trained ResNet50 model
        self.resnet18 = models.resnet18()

        # Replace the last fully connected layer to output two numerical values
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x = self.resnet18(x)
        mu, log_sigma = torch.chunk(x, 2, dim=-1)
        return mu, log_sigma