"""
This module provides functions for processing sequences, training GRU models, 
and evaluating their performance in adaptive learning tasks. It includes utilities 
for data preparation, cross-validation, and hyperparameter tuning.

Author: @emilebdn  
Created date: 2025-05-05
"""
#%%
import sys
import time
import torch

import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import random as rd
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from joblib import Parallel, delayed
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# Add the root of the repository to sys.path
sys.path.append(op.dirname(op.dirname(op.dirname(op.abspath(__file__)))))

from emilebdn.GRU.GRU_simple_model import SimpleRNN
from emilebdn.config.paths import computed_data_emile_path, data_outcome_level_preprocessed_path, data_outcome_level_simulated_path
from emilebdn.config.variables import n_jobs, nb_subjects, train_size_ratio, input_size, hidden_size, max_hidden_size, output_size, \
    learning_rate, num_epochs, batch_size, subject_embedding_dim

today = datetime.now().strftime("%Y%m%d")

#%%
def import_sequences(data, path, task, use_subject_embedding=False):
    """
    Import sequences from experimental or simulated data and optionally attach subject labels.

    Args:
        data_type (str): 'experiment' or 'simulation' to determine data source.
        path (str): Path to the CSV file containing the data.
        task (str): Task name to filter the dataset.
        use_subject_embedding (bool): Whether to include subject IDs in the sequence tuples.

    Returns:
        sequences (list): List of (input_seq, target_seq[, subject_id]).
        num_subjects (int, optional): Number of unique subjects (only if use_subject_embedding=True).
    """
    data = pd.read_csv(path)

    groupby_variables = ['subject', 'session_idx', 'sequence_id']

    # Filter the data to only the specified task
    data_filtered = data[data['task'] == task]
    grouped = data_filtered.groupby(groupby_variables)

    if use_subject_embedding:
        unique_subjects = sorted(data_filtered['subject'].unique())

    sequences = []

    # Iterate over each grouped sequence
    for key, group in grouped:
        group_sorted = group.sort_values('outcome_idx')
        input_seq = torch.tensor(group_sorted['outcome'].values, dtype=torch.float32).unsqueeze(1)
        target_seq = torch.tensor(group_sorted['estimate'].values, dtype=torch.float32).unsqueeze(1)

        if use_subject_embedding:
            subject = key[0]  # subject is the first key in both experiment and simulation
            sequences.append((input_seq, target_seq, subject))  # Use real subject ID
        else:
            sequences.append((input_seq, target_seq))

    if use_subject_embedding:
        return sequences, len(unique_subjects)
    else:
        return sequences, None

# def split_sequences(sequences, train_size_ratio):
#     """
#     Split the sequences into training and testing sets based on the specified ratio.

#     Args:
#         sequences (list): List of sequences to be split.
#         train_size_ratio (float): Ratio of training data to total data.

#     Returns:
#         tuple: A tuple containing training and testing sequences.
#     """
#     n_train = int(len(sequences) * train_size_ratio)
#     rd.shuffle(sequences)
#     train_sequences = sequences[:n_train]
#     test_sequences = sequences[n_train:]
#     return train_sequences, test_sequences

def split_sequences_for_cv(sequences, n_splits):
    """
    Split the data into n subsequences for cross-validation.

    Args:
        task (str): Task type ('ada-prob' or 'ada-pos').
        data_type (str): Data source type ('experiment' or 'simulation').
        n_splits (int): Number of splits for cross-validation.

    Returns:
        list: A list of tuples, where each tuple contains training and testing sequences for a split.
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
def train_and_evaluate_gru(train_sequences, test_sequences, input_size=input_size, hidden_size=hidden_size, \
                           output_size=output_size, learning_rate=learning_rate, num_epochs=num_epochs, \
                           batch_size=batch_size, use_subject_embedding=False, \
                            subject_embedding_dim=subject_embedding_dim, return_pred=False):
    """
    Train and evaluate a GRU-based RNN for sequence prediction.

    Args:
        train_sequences (list): Training data sequences; either (input_seq, target_seq) or (input_seq, target_seq, subject_id).
        test_sequences (list): Testing data sequences, same format as training.
        input_size (int): Input size for the GRU model.
        hidden_size (int): Hidden layer size for the GRU model.
        output_size (int): Output size for the GRU model.
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loaders.
        use_subject_embedding (bool): Whether to use subject embedding as extra input features.
        subject_embedding_dim (int): Dimension size of subject embedding vectors.

    Returns:
        dict: A dictionary containing training time, test loss, and EVF.
    """
    # if use_subject_embedding:
    #     model_input_size = input_size + subject_embedding_dim
    # else:
    #     model_input_size = input_size

    model_input_size = input_size
    
    model = SimpleRNN(model_input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if use_subject_embedding:
        # Map string subject IDs to integer indices for use in tensor
        unique_subject_ids = sorted(set(seq[2] for seq in train_sequences + test_sequences))
        subject_to_index = {subj: idx for idx, subj in enumerate(unique_subject_ids)}

        # Apply mapping to the train/test sequences
        train_sequences = [(inp, tgt, subject_to_index[subj]) for inp, tgt, subj in train_sequences]
        test_sequences = [(inp, tgt, subject_to_index[subj]) for inp, tgt, subj in test_sequences]

    def collate_fn(batch):
        if use_subject_embedding:
            inputs, targets, subjects = [], [], []
            for inp, tgt, subj in batch:
                inputs.append(inp)
                targets.append(tgt)
                subjects.append(subj)
            inputs = torch.stack(inputs, dim=0)   # batch_size x seq_len x input_size
            targets = torch.stack(targets, dim=0) # batch_size x seq_len x output_size
            subjects = torch.tensor(subjects, dtype=torch.long)
            return inputs, targets, subjects
        else:
            inputs, targets = zip(*batch)
            inputs = torch.stack(inputs, dim=0)   # batch_size x seq_len x input_size
            targets = torch.stack(targets, dim=0) # batch_size x seq_len x output_size
            return inputs, targets

    train_loader = DataLoader(train_sequences, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_sequences, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # if use_subject_embedding:
    #     subject_embedding_layer = nn.Embedding(num_embeddings=nb_subjects, embedding_dim=subject_embedding_dim) 
    #     subject_embedding_layer.train()

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            if use_subject_embedding:
                input_seq, target_seq, subject_ids = batch
                output_seq = model(input_seq, subject_ids)
            else:
                input_seq, target_seq = batch
                output_seq = model(input_seq)

            loss = criterion(output_seq, target_seq)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        # for batch in train_loader:
        #     optimizer.zero_grad()
        #     if use_subject_embedding:
        #         input_seq, target_seq, subject_ids = batch
        #         subject_embeds = subject_embedding_layer(subject_ids).unsqueeze(1).repeat(1, input_seq.size(1), 1)
        #         input_seq = torch.cat((input_seq, subject_embeds), dim=2)

        #     else:
        #         input_seq, target_seq = batch

        #     output_seq = model(input_seq)
        #     loss = criterion(output_seq, target_seq)
        #     loss.backward()
        #     optimizer.step()
        #     epoch_loss += loss.item()

    training_time = time.time() - start_time

    model.eval()
    test_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        with torch.no_grad():
            for batch in test_loader:
                if use_subject_embedding:
                    input_seq, target_seq, subject_ids = batch
                    output_seq = model(input_seq, subject_ids)
                else:
                    input_seq, target_seq = batch
                    output_seq = model(input_seq)

                loss = criterion(output_seq, target_seq)
                test_loss += loss.item()
                all_targets.extend(target_seq.numpy().flatten())
                all_predictions.extend(output_seq.numpy().flatten())

        # for batch in test_loader:
        #     if use_subject_embedding:
        #         input_seq, target_seq, subject_ids = batch
        #         subject_embeds = subject_embedding_layer(subject_ids).unsqueeze(1).repeat(1, input_seq.size(1), 1)
        #         input_seq = torch.cat((input_seq, subject_embeds), dim=2)
        #     else:
        #         input_seq, target_seq = batch

        #     output_seq = model(input_seq)
        #     loss = criterion(output_seq, target_seq)
        #     test_loss += loss.item()
        #     all_targets.extend(target_seq.numpy().flatten())
        #     all_predictions.extend(output_seq.numpy().flatten())

    evf = explained_variance_score(all_targets, all_predictions)

    if return_pred:
        return all_predictions
    else:
        return {
            "training_time": training_time,
            "test_loss": test_loss / len(test_loader),
            "evf": evf
        }

# # Plot predictions vs. ground truth for a few sequences
# with torch.no_grad():
#     for i, (input_seq, target_seq) in enumerate(test_loader):
#         if i >= 10:
#             break
#         output_seq = model(input_seq)
#         plt.figure(figsize=(10, 5))
#         plt.plot(target_seq[0].squeeze().numpy(), label="Data", marker='o')
#         plt.plot(output_seq[0].squeeze().numpy(), label="GRU prediction", marker='x')
#         plt.title(f"Sequence {i+1} - Predictions vs Ground Truth")
#         plt.xlabel("Time Step")
#         plt.ylabel("Value")
#         plt.legend()
#         plt.show()

# # Plot the distribution of estimates and predictions
# plt.figure(figsize=(10, 5))
# plt.hist(all_targets, bins=50, alpha=0.5, label="Targets", color='blue')
# plt.hist(all_predictions, bins=50, alpha=0.5, label="Predictions", color='orange')
# plt.title("Distribution of Estimates and Predictions")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.legend()
# plt.show()

# Evaluation function for one hidden size
def evaluate_hidden_size(hidden_size, cross_val_splits, use_subject_embedding=False, subject_embedding_dim=subject_embedding_dim):
    """
    Evaluate the GRU model performance for a given hidden layer size using cross-validation splits.

    Args:
        hidden_size (int): The size of the GRU hidden layer to evaluate.
        cross_val_splits (list): List of (train_sequences, test_sequences) tuples for cross-validation.
        use_subject_embedding (bool): Whether to use subject embedding.
        subject_embedding_dim (int): Dimension of the subject embedding.

    Returns:
        tuple: (hidden_size, mean_test_loss, mean_evf)
    """
    print(f"Testing hidden layer size: {hidden_size}")
    test_losses = []
    evfs = []

    for train_sequences, test_sequences in cross_val_splits:
        result = train_and_evaluate_gru(
            train_sequences, test_sequences, hidden_size=hidden_size, \
                use_subject_embedding=use_subject_embedding, subject_embedding_dim=subject_embedding_dim)
        test_losses.append(result['test_loss'])
        evfs.append(result['evf'])

    mean_test_loss = np.mean(test_losses)
    mean_evf = np.mean(evfs)
    print(f'mean_test_loss: {mean_test_loss}, mean_evf: {mean_evf}')
    return (hidden_size, mean_test_loss, mean_evf)

# Plot the results of GRU hidden layer size comparison
def plot_results(task, results_df):
    """
    Plot Test Loss and Explained Variance Fraction as a function of Hidden Layer Size.

    Args:
        task (str): Name of the task for plot title.
        results_df (pd.DataFrame): DataFrame with columns "Hidden Size", "Test Loss", and "Explained Variance Fraction".
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["Hidden Size"], results_df["Test Loss"], label="Test Loss", marker='o')
    plt.plot(results_df["Hidden Size"], results_df["Explained Variance Fraction"], label="Explained Variance Fraction", marker='x')
    plt.xscale('log')
    plt.xticks(results_df["Hidden Size"], labels=results_df["Hidden Size"], rotation=45)  # Explicitly show hidden sizes
    plt.xlabel("Hidden Layer Size (log scale)")
    plt.ylabel("Metric Value")
    plt.title(f" {task} - GRU Performance vs Hidden Layer Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def find_best_GRU_hidden_layer_size(sequences, train_size_ratio=train_size_ratio, max_hidden_size=max_hidden_size, n_jobs=n_jobs \
                                    , use_subject_embedding=False, subject_embedding_dim=subject_embedding_dim):
    """
    Find the best GRU hidden layer size by evaluating multiple sizes using cross-validation.

    Args:
        sequences (list): List of input sequences for training and evaluation.
        train_size_ratio (float): Ratio of data to use for training.
        max_hidden_size (int): Maximum hidden layer size to consider.
        n_jobs (int): Number of parallel jobs for evaluation.
        use_subject_embedding (bool): Whether to use subject embedding.
        subject_embedding_dim (int): Dimension of subject embedding vectors.

    Returns:
        pd.DataFrame: DataFrame containing hidden sizes, test losses, and explained variance fractions.
    """
    # Generate hidden layer sizes dynamically: [1, 2, 4, ..., max_hidden_size]
    hidden_layer_sizes = [2**i for i in range(1, int(np.log2(max_hidden_size)))]

    # Cross-validation setup
    n_splits = int(1 / (1-train_size_ratio))
    
    cross_val_splits = split_sequences_for_cv(sequences, n_splits)

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_hidden_size)(hs, cross_val_splits, use_subject_embedding, subject_embedding_dim) for hs in hidden_layer_sizes
    )

    # Find best hidden size by maximizing EVF
    best_hidden_size = max(results, key=lambda x: x[2])
    print("\nBest Hidden Layer Size:")
    print(f"Hidden Size: {best_hidden_size[0]}, Test Loss: {best_hidden_size[1]}, EVF: {best_hidden_size[2]}")

    # Save results to a CSV file
    results_df = pd.DataFrame(results, columns=["Hidden Size", "Test Loss", "Explained Variance Fraction"])

    return results_df

def process_subject(sequences, subject_id, data_type, task, use_subject_embedding):
    """
    Process a single subject by filtering their sequences and running GRU hidden layer size evaluation.

    Args:
        sequences (list): List of all input sequences (with subject IDs).
        subject_id (int or str): The subject identifier to filter sequences.
        data_type (str): Type of data ('experiment' or 'simulation').
        task (str): Task name for labeling results.
        use_subject_embedding (bool): Whether subject embedding is used.

    Returns:
        pd.DataFrame: DataFrame containing evaluation results for the subject.
    """
    subject_sequences = [seq for seq in sequences if seq[2] == subject_id]
    results_df = find_best_GRU_hidden_layer_size(subject_sequences, use_subject_embedding=use_subject_embedding)
    results_df['task'] = task
    results_df['subject_id'] = subject_id
    results_df['data_type'] = data_type
    results_df['use_subject_embedding'] = use_subject_embedding
    return results_df

def train_and_evaluate_subject_gru(sequences, subject_id, data_type, task, train_size_ratio, input_size, hidden_size, use_subject_embedding):
    """
    Train and evaluate a GRU model for a single subject using cross-validation.

    Args:
        sequences (list): List of all input sequences (with subject IDs).
        subject_id (int or str): The subject identifier to filter sequences.
        data_type (str): Type of data ('experiment' or 'simulation').
        task (str): Task name for labeling results.
        train_size_ratio (float): Ratio of data to use for training.
        hidden_size (int): Hidden layer size for the GRU model.
        use_subject_embedding (bool): Whether to use subject embedding.

    Returns:
        tuple: (subject_id, dict with training time, test loss, and explained variance)
    """
    subject_sequences = [seq for seq in sequences if seq[2] == subject_id]
    n_splits = int(1 / (1 - train_size_ratio))
    cross_val_splits = split_sequences_for_cv(subject_sequences, n_splits)
    test_losses = []
    evfs = []
    training_times = []
    for train_seq, test_seq in cross_val_splits:
        results = train_and_evaluate_gru(train_seq, test_seq, input_size, hidden_size, use_subject_embedding=use_subject_embedding)
        test_losses.append(results['test_loss'])
        evfs.append(results['evf'])
        training_times.append(results['training_time'])
    return (subject_id, {
        'data_type': data_type,
        'task': task,
        'hidden_size': hidden_size,
        'subject_id': subject_id,
        'training_time': np.mean(training_times),
        'test_loss': np.mean(test_losses),
        'explained_variance_fraction': np.mean(evfs)
    })