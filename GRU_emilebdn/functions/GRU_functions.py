# created 20250505

#%%

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

from models.GRU_simple_RNN import SimpleRNN
from config.paths import computed_data_emile_path, data_outcome_level_preprocessed_path, data_outcome_level_simulated_path
from config.variables import n_jobs, train_size_ratio, input_size, hidden_size, max_hidden_size, output_size, learning_rate, num_epochs, batch_size

today_date = datetime.now().strftime("%Y%m%d")

#%%
def import_sequences(task, data_type):
    # Load the dataset
    if data_type == 'experiment':
        data_outcome_level = pd.read_csv(data_outcome_level_preprocessed_path)
        data_outcome_level_filtered = data_outcome_level[data_outcome_level['task'] == task]
        grouped = data_outcome_level_filtered.groupby(['subject', 'session_idx', 'sequence_id'])
    
    if data_type == 'simulation':
        data_outcome_level_simulated = pd.read_csv(data_outcome_level_simulated_path)
        data_outcome_level_simulated_filtered = data_outcome_level_simulated[data_outcome_level_simulated['task'] == task]
        grouped = data_outcome_level_simulated_filtered.groupby(['subject', 'sequence_id'])

    sequences = []

    for _, group in grouped:
        group_sorted = group.sort_values('outcome_idx')
        input_seq = torch.tensor(group_sorted['outcome'].values, dtype=torch.float32).unsqueeze(1)
        target_seq = torch.tensor(group_sorted['estimate'].values, dtype=torch.float32).unsqueeze(1)
        sequences.append((input_seq, target_seq))

    return sequences

def split_sequences(sequences, train_size_ratio):
    """
    Split the sequences into training and testing sets based on the specified ratio.

    Args:
        sequences (list): List of sequences to be split.
        train_size_ratio (float): Ratio of training data to total data.

    Returns:
        tuple: A tuple containing training and testing sequences.
    """
    n_train = int(len(sequences) * train_size_ratio)
    rd.shuffle(sequences)
    train_sequences = sequences[:n_train]
    test_sequences = sequences[n_train:]
    return train_sequences, test_sequences


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
                           output_size=output_size, learning_rate=learning_rate, num_epochs=num_epochs, batch_size=batch_size):
    """
    Train and evaluate a GRU-based RNN for sequence prediction.

    Args:
        task (str): Task type ('ada-prob' or 'ada-pos').
        train_size_ratio (float): Ratio of training data to total data.
        input_size (int): Input size for the GRU model.
        hidden_size (int): Hidden layer size for the GRU model.
        output_size (int): Output size for the GRU model.
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loaders.

    Returns:
        dict: A dictionary containing training time, test loss, and EVF.
    """
    
    #print(f"The modeled task is: {task}")
    #print(f"The data source is: {data_type}")

    # Initialize the model, loss function, and optimizer
    model = SimpleRNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_sequences, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_sequences, batch_size=batch_size, shuffle=False)

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for input_seq, target_seq in train_loader:
            optimizer.zero_grad()
            output_seq = model(input_seq)
            loss = criterion(output_seq, target_seq)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # if epoch in [0, 24, 49, 74, 99]: 
            # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}")

    training_time = time.time() - start_time
    # print(f"Total Training Time: {training_time:.2f} seconds")

    # Evaluation loop
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for input_seq, target_seq in test_loader:
            output_seq = model(input_seq)
            loss = criterion(output_seq, target_seq)
            test_loss += loss.item()
            all_targets.extend(target_seq.numpy().flatten())
            all_predictions.extend(output_seq.numpy().flatten())

    # print(f"Test Loss: {test_loss/len(test_loader)}")
    evf = explained_variance_score(all_targets, all_predictions)
    # print(f"Explained Variance Fraction (EVF): {evf}")

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

    return {
        "training_time": training_time,
        "test_loss": test_loss / len(test_loader),
        "evf": evf
    }

# Evaluation function for one hidden size
def evaluate_hidden_size(hidden_size, cross_val_splits):
    print(f"Testing hidden layer size: {hidden_size}")
    test_losses = []
    evfs = []

    for train_sequences, test_sequences in cross_val_splits:
        result = train_and_evaluate_gru(
            train_sequences, test_sequences, hidden_size=hidden_size)
        test_losses.append(result['test_loss'])
        evfs.append(result['evf'])

    mean_test_loss = np.mean(test_losses)
    mean_evf = np.mean(evfs)
    print(f'mean_test_loss: {mean_test_loss}, mean_evf: {mean_evf}')
    return (hidden_size, mean_test_loss, mean_evf)

# Plot the results of GRU hidden layer size comparison
def plot_results(task, results_df):
    """
    Plots the Test Loss and Explained Variance Fraction against Hidden Layer Size.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing the results with columns 
      "Hidden Size", "Test Loss", and "Explained Variance Fraction".
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

def find_best_GRU_hidden_layer_size(task, data_type, train_size_ratio=train_size_ratio, max_hidden_size=max_hidden_size, n_jobs=n_jobs):
    # Generate hidden layer sizes dynamically: [1, 2, 4, ..., max_hidden_size]
    hidden_layer_sizes = [2**i for i in range(1, int(np.log2(max_hidden_size)))]

    # Cross-validation setup
    n_splits = int(1 / (1-train_size_ratio))
    
    sequences = import_sequences(task, data_type)
    cross_val_splits = split_sequences_for_cv(sequences, n_splits)

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_hidden_size)(hs, cross_val_splits) for hs in hidden_layer_sizes
    )

    # Find best hidden size by maximizing EVF
    best_hidden_size = max(results, key=lambda x: x[2])
    print("\nBest Hidden Layer Size:")
    print(f"Hidden Size: {best_hidden_size[0]}, Test Loss: {best_hidden_size[1]}, EVF: {best_hidden_size[2]}")

    # Save results to a CSV file
    results_df = pd.DataFrame(results, columns=["Hidden Size", "Test Loss", "Explained Variance Fraction"])

    results_filename = f"{today_date}_{task}_{data_type}_results.csv"
    results_path = op.join(computed_data_emile_path, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    # Call the function to plot the results
    plot_results(results_df, task)

    # Save the plot with the same path as results but change '.csv' to '.png'
    plot_filename = results_filename.replace('.csv', '.png')
    plot_path = op.join(computed_data_emile_path, plot_filename)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    return best_hidden_size[0]