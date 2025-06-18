"""
This module provides functions for processing sequences, training GRU models, 
and evaluating their performance in adaptive learning tasks. It includes utilities 
for data preparation, cross-validation, and hyperparameter tuning.

Author: @emilebdn  
Created date: 2025-06-06
"""
#%%
import sys
import time
import torch

import numpy as np
import os.path as op
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

sys.path.append(op.dirname(op.dirname(op.dirname(op.abspath(__file__)))))

from emilebdn.config.variables import (
    input_size, 
    hidden_size, 
    output_size, 
    learning_rate, 
    num_epochs, 
    batch_size,
)
from emilebdn.GRU.GRU_simple_model import SimpleRNN

today = datetime.now().strftime("%Y%m%d")

#%%
def import_sequences(data, path, task):
    """
    Import sequences from experimental or simulated data and optionally attach subject labels.
    """
    data = pd.read_csv(path)

    groupby_variables = ['subject', 'session_idx', 'sequence_id']

    # Filter the data to only the specified task
    data_filtered = data[data['task'] == task]
    grouped = data_filtered.groupby(groupby_variables)

    unique_subjects = sorted(data_filtered['subject'].unique())

    sequences = []

    # Iterate over each grouped sequence
    for key, group in grouped:
        group_sorted = group.sort_values('outcome_idx')
        hidden_parms = np.array(group_sorted['hidden_parameter'].values, dtype=np.float32)
        hidden_parms = np.expand_dims(hidden_parms, axis=1)
        input_seq = torch.tensor(group_sorted['outcome'].values, dtype=torch.float32).unsqueeze(1)
        target_seq = torch.tensor(group_sorted['estimate'].values, dtype=torch.float32).unsqueeze(1)

        subject = key[0]  # subject is the first key in both experiment and simulation
        sequences.append((hidden_parms, input_seq, target_seq, subject))  # Use real subject ID

    return sequences

def split_sequences_for_cv(sequences, n_splits):
    """
    Split the data into n subsequences for cross-validation.
    """
    # Define cross-validation splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []

    for train_index, test_index in kf.split(sequences):
        train_seqs = [sequences[i] for i in train_index]
        test_seqs = [sequences[i] for i in test_index]
        splits.append((train_seqs, test_seqs))

    return splits

#%%
def train_and_evaluate_gru(train_sequences, test_sequences, input_size=input_size, hidden_size=hidden_size,
                           output_size=output_size, learning_rate=learning_rate, num_epochs=num_epochs,
                           batch_size=batch_size, return_pred=False):
    """
    Train and evaluate a GRU-based RNN for sequence prediction.
    If return_pred is True, returns a list of tuples (output_seq, predicted_seq).
    """
    model = SimpleRNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def collate_fn(batch):
        hidden_parms, inputs, targets, _ = zip(*batch)
        inputs = torch.stack(inputs, dim=0)  # batch_size x seq_len x input_size
        targets = torch.stack(targets, dim=0)  # batch_size x seq_len x output_size
        return hidden_parms, inputs, targets

    train_loader = DataLoader(train_sequences, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_sequences, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            _, input_seq, target_seq = batch
            output_seq = model(input_seq)
            loss = criterion(output_seq, target_seq)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    training_time = time.time() - start_time

    model.eval()
    mse = 0.0  # Initialize list to store MSE values for each batch
    all_targets = []
    all_predictions = []
    sequence_pairs = []  # To store tuples of (output_seq, predicted_seq)

    with torch.no_grad():
        for batch in test_loader:
            hidden_parms, input_seq, target_seq = batch
            output_seq = model(input_seq)
            loss = criterion(output_seq, target_seq)
            mse += loss.item()  # Append MSE for the current batch
            all_targets.extend(target_seq.numpy().flatten())
            all_predictions.extend(output_seq.numpy().flatten())

            if return_pred:
                # Store the sequences
                sequence_pairs.extend(list(zip(hidden_parms, input_seq.numpy(), target_seq.numpy(), output_seq.numpy())))

    evf = explained_variance_score(all_targets, all_predictions)
    mse = mse/len(test_loader)  # Calculate the mean of MSE values

    result = {
        "training_time": training_time,
        "mse": mse,  # Use the calculated mean MSE
        "evf": evf,
    }

    if return_pred:
        result["sequence_pairs"] = sequence_pairs

    return result

def train_and_evaluate_subject_gru(subject_sequences, subject_id, data, task, train_size_ratio, hidden_size):
    """
    Train and evaluate a GRU model for a single subject using cross-validation.
    """
    n_splits = int(1 / (1 - train_size_ratio))
    cross_val_splits = split_sequences_for_cv(subject_sequences, n_splits)
    mse = 0.0
    evfs = []
    training_times = []
    for train_seq, test_seq in cross_val_splits:
        print(f"Train sequences: {len(train_seq)}, Test sequences: {len(test_seq)}")
        results = train_and_evaluate_gru(train_seq, test_seq, hidden_size=hidden_size)
        mse += results['mse']
        evfs.append(results['evf'])
        training_times.append(results['training_time'])
    return (subject_id, {
        'data': data,
        'task': task,
        'hidden_size': hidden_size,
        'subject_id': subject_id,
        'training_time': np.mean(training_times),
        'mse': mse/len(cross_val_splits),
        'evf': np.mean(evfs)
    })