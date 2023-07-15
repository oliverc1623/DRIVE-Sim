import torch
import torch.nn as nn
import convlstm as convLSTM

class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.convlstm = convLSTM.ConvLSTM(3, 128, (3,3), 
                                          4, False, True, False)
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(128*30*32, 256)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(256, 2)
        
    def forward(self, x):
        _, lstm_output = self.convlstm(x)
        x = self.flat(lstm_output[0][0])
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        mu, sigma = torch.chunk(x, 2, dim=-1)
        return mu, sigma