#20250626
"""
(...)
"""

#%%
import datetime
import random
import sys
import torch

import numpy as np
import os.path as op
import pandas as pd

from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error

# Add the root of the repository to sys.path
sys.path.append(op.dirname(op.dirname(op.dirname(__file__))))

from emilebdn.config.paths import data_outcome_level_preprocessed_path
from emilebdn.config.variables import (
    n_jobs,
    nb_subjects,
    train_size_ratio,
    length, 
    n_sequences_for_each_subject
)
from emilebdn.GRU.GRU_functions_2 import (
    format_sequences,
    train_and_evaluate_gru,
)
from emilebdn.HMM.HMM_functions import predict_sequences_with_HMM

today = datetime.datetime.now().strftime("%Y%m%d")
#%%
model = 'GRU'
hidden_size = 512
path = data_outcome_level_preprocessed_path.replace('.csv', '_ada-prob.csv')
task = 'ada-prob'

n_sb = nb_subjects
n_tri = length
n_seq = n_sequences_for_each_subject[task]  # Number of sequences for each subject

n_test_seqs = int(n_seq*(1 - train_size_ratio)) # Number of sequences per test sequences set for model fitting
nb_test_seqs = int(n_seq / n_test_seqs) # Number of test sequences per subject

data_ada_prob = pd.read_csv(path)

subjects = pd.read_csv(path)['subject'].drop_duplicates().tolist()
subjects_nb = {
    subject: i for i, subject in enumerate(subjects)
}

data_outcome_level_with_pred = pd.read_csv(path)

#%%
### Subject_wise GRU Predictions
data_outcome_level_with_pred['subject_GRU'] = np.full(n_sb*n_tri*n_seq, np.nan)

for subject, i in subjects_nb.items():
    subject_data = data_ada_prob[data_ada_prob['subject'] == subject]
    subject_sequences = format_sequences(subject_data, task, subject_data)
    # print(subject_sequences[0][1])
    # for n, sequence in enumerate(subject_sequences):
    #     outcome_seq = subject_data.iloc[n_tri*n : n_tri*(n + 1)]['outcome']
    #     outcome_seq = np.array(outcome_seq.tolist())
    #     outcome_seq = torch.from_numpy(outcome_seq)
    #     outcome_seq = outcome_seq.unsqueeze(1)  # shape (N, 1), so each i -> [i]
    #     if not np.array_equal(sequence[1], outcome_seq):
    #         raise ValueError(f"Outcome sequence mismatch for subject {subject} at sequence {n}.")

    for m in range(nb_test_seqs):
        test_sequences = subject_sequences[n_test_seqs*m:n_test_seqs*(m+1)]
        train_sequences = subject_sequences[0:n_test_seqs*m] + subject_sequences[n_test_seqs*(m+1):]
        GRU_subject = train_and_evaluate_gru(train_sequences, test_sequences, hidden_size=hidden_size, return_pred=True)['sequence_pairs']
        for p, tuple in enumerate(GRU_subject):
            _, _, _, gru_sb_predicted_seq = tuple
            lower_bound =  n_seq*n_tri*i + n_test_seqs*n_tri*m + n_tri*p
            # Assign predictions to the correct rows, preserving order
            pred_values = [q[0] for q in gru_sb_predicted_seq]
            data_outcome_level_with_pred.iloc[lower_bound:lower_bound + n_tri - 1, 'subject_GRU'] = pred_values

# Compute MSE between 'estimate' and 'subject_GRU'

mse_subject_gru = mean_squared_error(
    data_outcome_level_with_pred['estimate'],
    data_outcome_level_with_pred['mean_estimate']
)
print(f"MSE between 'estimate' and 'subject_GRU': {mse_subject_gru}")


data_outcome_level_with_pred.to_csv(path.replace('.csv', '_with_predictions.csv'), index=False)

#%%
### Group-wise GRU Predictions
data_outcome_level_with_pred['group_GRU'] = np.full(n_sb*n_tri*n_seq, np.nan)
all_sequences = format_sequences(path, task, data_ada_prob)

for subject, i in subjects_nb.items():
    subject_data = data_ada_prob[data_ada_prob['subject'] == subject]
    subject_sequences = format_sequences(subject_data, task, subject_data)
    
    for m in range(nb_test_seqs):
        test_sequences = subject_sequences[n_test_seqs*m:n_test_seqs*(m+1)]
        
        # Filter all_sequences to exclude any sequence whose outcome sequence matches any in test_sequences
        test_outcomes = [np.array(seq[1]) for seq in test_sequences]
        all_sequences_filtered = [
            seq for seq in all_sequences
            if not any(np.array_equal(np.array(seq[1]), test_outcome) for test_outcome in test_outcomes)
        ]
        
        train_sequences = random.sample(all_sequences_filtered, len(test_sequences))

        GRU_group = train_and_evaluate_gru(train_sequences, test_sequences, hidden_size=hidden_size, return_pred=True)['sequence_pairs']
        
        for p, tuple in enumerate(GRU_subject):
            _, _, _, gru_sb_predicted_seq = tuple
            lower_bound = n_tri*n_seq*i + n_test_seqs*n_tri*m + n_tri*p
            # Assign predictions to the correct rows, preserving order
            pred_values = [q[0] for q in gru_sb_predicted_seq]
            data_outcome_level_with_pred.iloc[lower_bound:lower_bound + n_tri - 1, 'group_GRU'] = pred_values

data_outcome_level_with_pred.to_csv(path.replace('.csv', '_with_predictions.csv'), index=False)

#%%
### Mean subject behavior
data_outcome_level_with_pred['mean_estimate'] = np.full(n_sb*n_tri*n_seq, np.nan)

outcome_estimate_pairs = [(data_outcome_level_with_pred['outcome'].tolist()[i:i+n_tri], data_outcome_level_with_pred['estimate'].tolist()[i:i+n_tri]) for i in range(0, len(data_outcome_level_with_pred), n_tri)]
print(len(outcome_estimate_pairs))
for subject, i in subjects_nb.items():
    for n in range(n_seq):
        outcome = data_outcome_level_with_pred['outcome'].tolist()[n_tri*n_seq*i + n_tri*n:n_tri*n_seq*i + n_tri*(n+1)]
        estimates = []
        for pair in outcome_estimate_pairs:
            if np.array_equal(np.array(pair[0]), np.array(outcome)):
                estimates.append(pair[1])
        mean_estimate = np.mean(estimates, axis=0)
        data_outcome_level_with_pred.iloc[n_tri*n_seq*i + n_tri*n:n_tri*n_seq*i + n_tri*(n+1) - 1, 'mean_estimate'] = mean_estimate       

#%%
### HMM subject-wise Predictions
data_outcome_level = pd.read_csv(data_outcome_level_preprocessed_path)
data_outcome_level = data_outcome_level[data_outcome_level['task'] == task]

data_outcome_level_with_pred['subject_HMM'] = np.full(n_sb*n_tri*n_seq, np.nan)

for subject, i in subjects_nb.items():
    for n in range(nb_test_seqs):
        train_data = pd.concat([
            data_outcome_level.iloc[n_tri*n_seq*i : n_tri*n_seq*i + n_test_seqs*n_tri*n],
            data_outcome_level.iloc[n_tri*n_seq*i + n_test_seqs*n_tri*(n+1) : n_tri*n_seq*(i+1)]
        ])
        for m in range(n_test_seqs):
            test_data = data_outcome_level.iloc[n_tri*n_seq*i + n_test_seqs*n_tri*n + n_tri*m:n_tri*n_seq*i + n_test_seqs*n_tri*n + n_tri*(m+1) - 1]
            HMM_subject = predict_sequences_with_HMM(train_data, test_data, task)
            data_outcome_level_with_pred.iloc[n_tri*n_seq*i + n_test_seqs*n_tri*n:n_tri*n_seq*i + n_test_seqs*n_tri*(n+1) - 1, 'subject_HMM'] = HMM_subject


#%%
