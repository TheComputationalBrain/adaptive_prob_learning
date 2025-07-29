"""
This module defines two neural network architectures, SimpleRNN and SimpleGRU,
for sequence prediction tasks using PyTorch. SimpleRNN implements a basic Recurrent Neural Network,
while SimpleGRU implements a Gated Recurrent Unit network.

Author: @emilebdn  
Created date: 2025-04-15  
Updated: 2025-07-29
"""
import sys

import os.path as op
import torch.nn as nn

sys.path.append(op.dirname(op.dirname(op.dirname(__file__))))

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the RNN hidden state.
            output_size (int): Size of the output features.
        """
        super(SimpleRNN, self).__init__()

        self.input_size = input_size

        self.rnn_input_size = input_size

        self.rnn = nn.RNN(self.rnn_input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Tensor: Output predictions of shape (batch, seq_len, output_size).
        """
        rnn_out, _ = self.rnn(x)  
        output = self.fc(rnn_out)  
        return output
    
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the GRU hidden state.
            output_size (int): Size of the output features.
        """
        super(SimpleGRU, self).__init__()
        self.input_size = input_size

        self.rnn_input_size = input_size

        self.gru = nn.GRU(self.rnn_input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, input_size).
        Returns:
            Tensor: Output predictions of shape (batch, seq_len, output_size).
        """
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out)
        return output