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

import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

sys.path.append(op.dirname(op.dirname(op.dirname(__file__))))

from emilebdn.config.variables import (
    random_state,
    input_size, 
    hidden_size, 
    output_size, 
    learning_rate, 
    num_epochs, 
    batch_size,
)
from emilebdn.GRU.GRU_simple_model import SimpleGRU, SimpleRNN

today = datetime.now().strftime("%Y%m%d")

#%%
def format_sequences(path, task, data=None):
    """
    Import sequences from experimental or simulated data and optionally attach subject labels.
    """
    if data is None:
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

def flatten_sequences(sequences, tested_sequence=None):
    """
    Flatten a list of sequences into a single DataFrame.
    If tested_sequence is provided, its records will appear first in the DataFrame.
    Each sequence is a tuple: (hidden_parms, input_seq, target_seq, subject)
    """
    tested_records = []
    other_records = []

    for seq_idx, sequence in enumerate(sequences):
        hidden_parms, input_seq, target_seq, subject = sequence
        seq_len = len(input_seq)

        for t in range(seq_len):
            record = {
                'subject': subject,
                'sequence_id_2': seq_idx,
                'outcome_idx': t,
                'hidden_parameter': hidden_parms[t].item(),
                'outcome': input_seq[t].item(),
                'estimate': target_seq[t].item()
            }

            if tested_sequence is not None and id(sequence) == id(tested_sequence):
                tested_records.append(record)
            else:
                other_records.append(record)

    # Return tested sequence first
    df = pd.DataFrame.from_records(tested_records + other_records)
    return df

def split_sequences_for_cv(sequences, n_splits):
    """
    Split the data into n subsequences for cross-validation.
    """
    # Define cross-validation splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    for train_index, test_index in kf.split(sequences):
        train_seqs = [sequences[i] for i in train_index]
        test_seqs = [sequences[i] for i in test_index]
        splits.append((train_seqs, test_seqs))

    return splits

#%%
def train_and_evaluate_gru(train_sequences, test_sequences, input_size=input_size, hidden_size=hidden_size,
                           output_size=output_size, learning_rate=learning_rate, num_epochs=num_epochs,
                           batch_size=batch_size, return_pred=False, RNN=False):
    """
    Train and evaluate a GRU-based RNN for sequence prediction.
    If return_pred is True, returns a list of tuples (output_seq, predicted_seq).
    """
    if RNN:
        model = SimpleRNN(input_size, hidden_size, output_size)
    else:
        model = SimpleGRU(input_size, hidden_size, output_size)
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

def plot_subject_sequence(
    outcome_seq,
    hidden_parms,
    len_train_seq_sb,
    len_train_seq_gr,
    random_subject,
    hidden_sizes,
    subject_seq=None,
    mean_subject_behavior=None,
    gru_sb_predicted_seq=None,
    gru_gr_predicted_seq=None,
    hmm_sb_predicted_seq=None,
    hmm_gr_predicted_seq=None
):
    """
    Plots the behavioral sequence of a subject along with model predictions and hidden parameters.
    Parameters
    ----------
    outcome_seq : array-like of int (0 or 1)
        # Sequence of observed outcomes (e.g., stimuli), where 1 = blue, 0 = yellow.
    hidden_parms : array-like or np.ndarray
        # Sequence of hidden parameter values (e.g., latent state probabilities or model parameters).
        # Can be 1D (length T) or 2D (shape [T, D]) where D is the number of hidden parameters.
    len_train_seq_sb : int
        # Length of the subject-level training sequence.
    len_train_seq_gr : int
        # Length of the group-level training sequence.
    random_subject : int or str
        # Identifier for the subject being plotted.
    hidden_sizes : dict
        # Dictionary with keys 'subject' and 'group', values are the number of GRU units for each model.
    subject_seq : array-like of float, optional
        # Sequence of subject's behavioral estimates (e.g., probability of choosing blue).
    mean_subject_behavior : array-like of float, optional
        # Sequence of mean behavioral estimates across subjects.
    gru_sb_predicted_seq : array-like of float, optional
        # Sequence of subject-level GRU model predictions.
    gru_gr_predicted_seq : array-like of float, optional
        # Sequence of group-level GRU model predictions.
    hmm_sb_predicted_seq : array-like of float, optional
        # Sequence of subject-level HMM model predictions.
    hmm_gr_predicted_seq : array-like of float, optional
        # Sequence of group-level HMM model predictions.
    Returns
    -------
    None
        # Displays a matplotlib plot visualizing the outcome sequence, hidden parameters, subject and mean behavior,
        # and model predictions (GRU and HMM) for the specified subject.
    """
    plt.figure(figsize=(14, 10))  
    ax1 = plt.subplot(1, 1, 1)
    
    ### Outcome sequence and hidden parameters plotting
    # Plot outcome as dots: blue if 1, yellow if 0, with larger size
    outcome_colors = ['yellow' if o == 0 else 'blue' for o in outcome_seq]
    ax1.scatter(
        range(len(outcome_seq)),
        outcome_seq,
        c=outcome_colors,
        label='Stimuli (Outcome)',
        s=80,  # Increased point size
        marker='o',
        edgecolor='black',
        zorder=3
    )
    # Plot hidden parameter in dark gray, dashed line
    ax2 = ax1.twinx()
    if hasattr(hidden_parms, 'shape') and len(hidden_parms.shape) == 2 and hidden_parms.shape[1] > 1:
        mean_hidden = hidden_parms.mean(axis=1)
        ax2.plot(
            mean_hidden,
            label='Hidden Param (mean)',
            color='dimgray',
            linestyle='--',
            alpha=0.8,
            linewidth=2
        )
    else:
        ax2.plot(
            hidden_parms,
            label='Hidden Param',
            color='dimgray',
            linestyle='--',
            alpha=0.8,
            linewidth=2
        )

    ### Subject and mean subject behavior plotting
    # Plot subject estimate as light blue line
    ax1.plot(subject_seq, label='Subject Behavior', color='deepskyblue', alpha=0.8, linewidth=4)
    # Plot mean subject behavior as a dark blue line
    if mean_subject_behavior is not None:
        ax1.plot(mean_subject_behavior, label='Mean Subject Behavior', color='darkblue', alpha=0.8, linewidth=2)

    ### GRU estimates plotting
    # Plot subject-level GRU model estimate as red line
    if gru_sb_predicted_seq is not None:
        ax1.plot(gru_sb_predicted_seq, label=f"Subject-level GRU (nb units: {hidden_sizes['subject']})", color='red', alpha=0.8, linewidth=2)
    # Plot group-level GRU model estimate as pink line
    if gru_gr_predicted_seq is not None:
        ax1.plot(gru_gr_predicted_seq, label=f"Group-level GRU (nb units: {hidden_sizes['group']})", color='pink', alpha=0.8, linewidth=2)
    
    ### HMM estimates plotting
    # Plot subject-level HMM estimate as green line
    if hmm_sb_predicted_seq is not None:
        ax1.plot(
            hmm_sb_predicted_seq,
            label='Subject-level HMM',
            color='green',
            alpha=0.8,
            linewidth=2
        )
    # Plot group-level HMM estimate as dark green line
    if hmm_gr_predicted_seq is not None:
        ax1.plot(
            hmm_gr_predicted_seq,
            label='Group-level HMM',
            color='darkgreen',
            alpha=0.8,
            linewidth=2
        )

    ax1.set_ylabel(r'$\mathbb{P}(\mathrm{blue})$')
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Time step')
    ax1.set_title(f'Sequence 1 (Subject {random_subject})')

    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelcolor='black')  
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Set legend positions: main legend top right, secondary legend bottom right
    main_legend_loc = (0.97, 0.97)
    secondary_legend_loc = (0.97, 0.03)
    # Main legend for lines
    legend1 = ax1.legend(
        lines + lines2,
        labels + labels2,
        loc='upper right',
        bbox_to_anchor=main_legend_loc,
        frameon=True,  # Boxed
        fancybox=True
    )
    ax1.add_artist(legend1)
    # Secondary legend for train sequence info (with box)
    legend2 = ax1.legend(
        [
            f"Train seq (subject): {len_train_seq_sb}",
            f"Train seq (group): {len_train_seq_gr}"
        ],
        loc='lower right',
        bbox_to_anchor=secondary_legend_loc,
        frameon=True,  # Boxed
        fancybox=True,
        handlelength=0,
        handletextpad=0
    )
    ax1.add_artist(legend2)
    plt.tight_layout()
    plt.show()