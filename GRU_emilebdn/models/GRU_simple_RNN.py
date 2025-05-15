"""
This script implements a simple RNN-based recurrent neural network for sequence prediction tasks. 

Author: @emilebdn  
Created date: 2025-04-15
"""
# created by @emilebdn on 2025/04/15

import torch.nn as nn

# Define a simple RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        output = self.fc(rnn_out)
        return output