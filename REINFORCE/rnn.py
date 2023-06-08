import torch
import torch.nn as nn
import ConvLSTM as convLSTM

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_mean = nn.Linear(hidden_size, 1)
        self.fc_std = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Set initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)

        # Decode hidden state of last time step
        mean = self.fc_mean(out[:, -1, :])
        std = self.fc_std(out[:, -1, :])

        return mean, std


class LSTMLaneFollower(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlstm = convLSTM.ConvLSTM(3, 32, (3,3), 6, True, True, False) 
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(32*30*32, 128)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(128, 2)

    def forward(self, x):
        """
        Does a forward pass of the given data through the layers of the neural network.
        
        :param img: (tensor) tensor of rgb values that represent an image
        """
        _, lstm_output = self.convlstm(x)
        x = self.flat(lstm_output[0][0])
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x
