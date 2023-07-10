import torch
import torch.nn as nn
from htm_pytorch import HTMAttention


class CNNHTMModel(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim):
        super(CNNHTMModel, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 8 * 8, hidden_dim)

        self.attn = HTMAttention(
            dim=hidden_dim,
            heads=8,
            dim_head=64,
            topk_mems=8,
            mem_chunk_size=32,
            add_pos_enc=True
        )

        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, queries, memories, mask):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        x = self.fc1(x)

        x = self.attn(queries, memories, mask=mask)

        x = self.fc2(x)
        return x


# Example usage
input_channels = 3  # RGB channels
hidden_dim = 512
output_dim = 5  # Number of output classes

model = CNNHTMModel(input_channels, hidden_dim, output_dim)
input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, RGB images of size 32x32
queries = torch.randn(1, 128, hidden_dim)
memories = torch.randn(1, 20000, hidden_dim)
mask = torch.ones(1, 20000).bool()

output_tensor = model(input_tensor, queries, memories, mask)
print(output_tensor.shape)