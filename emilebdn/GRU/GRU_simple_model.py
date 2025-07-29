"""
This script implements a simple RNN-based recurrent neural network for sequence prediction tasks, 
with optional subject-specific embeddings for modeling individual differences.

Author: @emilebdn  
Created date: 2025-04-15  
Updated: 2025-05-15
"""
#%%
import sys
import torch

import os.path as op
import torch.nn as nn

sys.path.append(op.dirname(op.dirname(op.dirname(__file__))))

#%%
# from emilebdn.config.variables import subject_embedding_dim

subject_embedding_dim = None
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_subjects=None, subject_embedding_dim=subject_embedding_dim):
        """
        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the RNN hidden state.
            output_size (int): Size of the output features.
            num_subjects (int, optional): Number of unique subjects for embeddings.
            subject_embedding_dim (int): Dimensionality of the subject embedding.
        """
        super(SimpleRNN, self).__init__()

        # self.use_subject_embedding = num_subjects is not None
        self.input_size = input_size

        # if self.use_subject_embedding:
        #     self.subject_embedding = nn.Embedding(num_subjects, subject_embedding_dim)
        #     self.rnn_input_size = input_size + subject_embedding_dim
        # else:
        #     self.rnn_input_size = input_size

        self.rnn_input_size = input_size

        self.rnn = nn.RNN(self.rnn_input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, subject_ids=None):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, input_size).
            subject_ids (Tensor, optional): Tensor of shape (batch,) containing subject indices.

        Returns:
            Tensor: Output predictions of shape (batch, seq_len, output_size).
        """
        # if self.use_subject_embedding:
        #     if subject_ids is None:
        #         raise ValueError("subject_ids must be provided when using subject embeddings.")
        #     embedded = self.subject_embedding(subject_ids)  # (batch, embed_dim)
        #     embedded = embedded.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, seq_len, embed_dim)
        #     x = torch.cat((x, embedded), dim=2)  # (batch, seq_len, input_size + embed_dim)

        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_size)
        output = self.fc(rnn_out)  # (batch, seq_len, output_size)
        return output
    
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_subjects=None, subject_embedding_dim=subject_embedding_dim):
        """
        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the GRU hidden state.
            output_size (int): Size of the output features.
            num_subjects (int, optional): Number of unique subjects for embeddings.
            subject_embedding_dim (int): Dimensionality of the subject embedding.
        """
        super(SimpleGRU, self).__init__()
        # self.use_subject_embedding = num_subjects is not None
        self.input_size = input_size

        # if self.use_subject_embedding:
        #     self.subject_embedding = nn.Embedding(num_subjects, subject_embedding_dim)
        #     self.rnn_input_size = input_size + subject_embedding_dim
        # else:
        #     self.rnn_input_size = input_size

        self.rnn_input_size = input_size

        self.gru = nn.GRU(self.rnn_input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, subject_ids=None):
        """
        Forward pass of the model.
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, input_size).
            subject_ids (Tensor, optional): Tensor of shape (batch,) containing subject indices.
        Returns:
            Tensor: Output predictions of shape (batch, seq_len, output_size).
        """
        # if self.use_subject_embedding:
        #     if subject_ids is None:
        #         raise ValueError("subject_ids must be provided when using subject embeddings.")
        #     embedded = self.subject_embedding(subject_ids)
        #     embedded = embedded.unsqueeze(1).repeat(1, x.size(1), 1)
        #     x = torch.cat((x, embedded), dim=2)

        gru_out, _ = self.gru(x)  # Utilisation de GRU au lieu de RNN
        output = self.fc(gru_out)
        return output