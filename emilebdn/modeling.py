### '20250709'
#%%
### 0 - Imports
import os
import random
import sys

import numpy as np
import os.path as op
import pandas as pd

from joblib import Parallel, delayed
from sklearn.neural_network import MLPRegressor

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.append(op.dirname(op.dirname(__file__)))

from emilebdn.config.paths import data_outcome_level_preprocessed_path
from emilebdn.config.variables import (
    n_jobs,
    random_state,
    nb_subjects,
    train_size_ratio,
    length, 
    n_sequences_for_each_subject,
    window_size,
    max_iter
)
from emilebdn.GRU.GRU_functions_2 import (
    format_sequences,
    train_and_evaluate_gru,
)
from emilebdn.HMM.HMM_functions import (
    predict_sequences_with_HMM
)

task = 'ada-prob'

n_sb = nb_subjects # Number of subjects
n_tri = length # Number of trials per sequence
n_seq = n_sequences_for_each_subject[task]  # Number of sequences for each subject

n_test_seqs = int((1 - train_size_ratio)*n_seq) + 1 # Number of sequences per test sequences set for model fitting
nb_test_seqs = int(n_seq / n_test_seqs) # Number of test sequences sets per subject

data_outcome_level = pd.read_csv(data_outcome_level_preprocessed_path)
data = data_outcome_level[data_outcome_level['task'] == task]

subjects = data['subject'].drop_duplicates().tolist()

data_path = \
    data_outcome_level_preprocessed_path.replace('.csv', '_ada-prob_with_predictions.csv')

#%%
# ### 1 - Mean subject behavior
# data = pd.read_csv(data_path)
# cv = False
# model = 'mean_estimate_with_cv' if cv else 'mean_estimate_without_cv'
# data.loc[:, model] = np.full(n_sb * n_tri * n_seq, np.nan)

# outcome_estimate_pairs = [
#     (data['outcome'].tolist()[i:i + n_tri],
#      data['estimate'].tolist()[i:i + n_tri])
#     for i in range(0, len(data), n_tri)
# ]

# def process_subject(subject, i):
#     print(f'Subject {subject} with id {i} processed for {model}')

#     # Cross-validation filtering
#     if cv:
#         filtered_data = data[data['subject'] != subject]
#         subject_outcome_estimate_pairs = [
#             (filtered_data['outcome'].tolist()[j:j + n_tri],
#              filtered_data['estimate'].tolist()[j:j + n_tri])
#             for j in range(0, len(filtered_data), n_tri)
#         ]
#     else:
#         subject_outcome_estimate_pairs = outcome_estimate_pairs

#     updates = []

#     for n in range(n_seq):
#         start_idx = n_tri * n_seq * i + n_tri * n
#         end_idx = n_tri * n_seq * i + n_tri * (n + 1)
#         outcome = data.loc[start_idx:end_idx - 1, 'outcome']

#         estimates = []
#         for pair in subject_outcome_estimate_pairs:
#             if np.array_equal(np.array(pair[0]), np.array(outcome)):
#                 estimates.append(pair[1])

#         mean_estimate = np.mean(estimates, axis=0)
#         updates.append((start_idx, end_idx, mean_estimate))

#     return updates

# # Run in parallel
# all_updates = Parallel(n_jobs=n_jobs)(
#     delayed(process_subject)(subject, i) for i, subject in enumerate(subjects)
# )

# # Apply updates
# for updates in all_updates:
#     for start_idx, end_idx, mean_estimate in updates:
#         data.loc[start_idx:end_idx - 1, model] = mean_estimate      

# data.to_csv(data_path, index=False)

### Temporary
# #%%
### 2 - Subject_wise RNN Predictions
for hidden_size in [32, 512, 1024, 2048]:
    data = pd.read_csv(data_path)
    model = f'subject_RNN_2_{hidden_size}'
    data[model] = np.full(n_sb*n_tri*n_seq, np.nan)

    def subject_gru_prediction(subject, i):
        print(f'Subject {subject} with id {i} processed for {model}')
        subject_data = data[data['subject'] == subject]
        subject_sequences = format_sequences(None, task, subject_data)
        preds = []
        idxs = []
        for m in range(nb_test_seqs):
            test_sequences = subject_sequences[n_test_seqs*m:n_test_seqs*(m + 1)]
            train_sequences = subject_sequences[0:n_test_seqs*m] + subject_sequences[n_test_seqs*(m + 1):]
            GRU_subject = train_and_evaluate_gru(train_sequences, test_sequences, hidden_size=hidden_size, return_pred=True, RNN=True)['sequence_pairs']
            for p, tuple in enumerate(GRU_subject):
                _, _, _, gru_sb_predicted_seq = tuple
                lower_bound = n_seq*n_tri*i + n_test_seqs*n_tri*m + n_tri*p
                pred_values = [q[0] for q in gru_sb_predicted_seq]
                idxs.append((lower_bound, lower_bound + n_tri))
                preds.append(pred_values)
        return i, idxs, preds

    results = Parallel(n_jobs=n_jobs)(
        delayed(subject_gru_prediction)(subject, i) for i, subject in enumerate(subjects)
    )

    for i, idxs, preds in results:
        for (start, end), pred_values in zip(idxs, preds):
            data.loc[start:end-1, model] = pred_values

    data.to_csv(data_path, index=False)

# #%%
# ### 2 - Subject_wise GRU Predictions
# for hidden_size in [32, 512, 2048]:
#     data = pd.read_csv(data_path)
#     model = f'subject_GRU_{hidden_size}'
#     data[model] = np.full(n_sb*n_tri*n_seq, np.nan)

#     def subject_gru_prediction(subject, i):
#         print(f'Subject {subject} with id {i} processed for {model}')
#         subject_data = data[data['subject'] == subject]
#         subject_sequences = format_sequences(None, task, subject_data)
#         preds = []
#         idxs = []
#         for m in range(nb_test_seqs):
#             test_sequences = subject_sequences[n_test_seqs*m:n_test_seqs*(m + 1)]
#             train_sequences = subject_sequences[0:n_test_seqs*m] + subject_sequences[n_test_seqs*(m + 1):]
#             GRU_subject = train_and_evaluate_gru(train_sequences, test_sequences, hidden_size=hidden_size, return_pred=True)['sequence_pairs']
#             for p, tuple in enumerate(GRU_subject):
#                 _, _, _, gru_sb_predicted_seq = tuple
#                 lower_bound = n_seq*n_tri*i + n_test_seqs*n_tri*m + n_tri*p
#                 pred_values = [q[0] for q in gru_sb_predicted_seq]
#                 idxs.append((lower_bound, lower_bound + n_tri))
#                 preds.append(pred_values)
#         return i, idxs, preds

#     results = Parallel(n_jobs=n_jobs)(
#         delayed(subject_gru_prediction)(subject, i) for i, subject in enumerate(subjects)
#     )

#     for i, idxs, preds in results:
#         for (start, end), pred_values in zip(idxs, preds):
#             data.loc[start:end-1, model] = pred_values

#     data.to_csv(data_path, index=False)

### Temporary
#%%
### 3 - Group-wise RNN Predictions
### with (for train sequences) all sequences 
### except the sequences with the same outcome as the test sequences, 
### except the sequences with the same subject than the test sequences' one, 
### and the same number of train sequences than in subject-wise GRU
for hidden_size in [2048]:
    data = pd.read_csv(data_path)
    model = f'group_RNN_2_{hidden_size}'
    data[model] = np.full(n_sb*n_tri*n_seq, np.nan)
    all_sequences = format_sequences(None, task, data)

    def group_gru_prediction(subject, i):
        print(f'Subject {subject} with id {i} processed for {model}')
        subject_data = data[data['subject'] == subject]
        subject_sequences = format_sequences(None, task, subject_data)
        preds = []
        idxs = []
        for m in range(nb_test_seqs):
            test_sequences = subject_sequences[n_test_seqs*m:n_test_seqs*(m + 1)]
            test_outcomes = [np.array(seq[1]) for seq in test_sequences]
            all_sequences_filtered = [
                seq for seq in all_sequences
                if not any(np.array_equal(np.array(seq[1]), test_outcome) for test_outcome in test_outcomes)
            ]
            all_sequences_filtered = [
                seq for seq in all_sequences_filtered
                if seq[3] != subject
            ]
            train_sequences = random.sample(all_sequences_filtered, n_seq - len(test_sequences))
            GRU_group = train_and_evaluate_gru(train_sequences, test_sequences, hidden_size=hidden_size, return_pred=True, RNN=True)['sequence_pairs']
            for p, tuple in enumerate(GRU_group):
                _, _, _, gru_sb_predicted_seq = tuple
                lower_bound = n_tri*n_seq*i + n_test_seqs*n_tri*m + n_tri*p
                pred_values = [q[0] for q in gru_sb_predicted_seq]
                idxs.append((lower_bound, lower_bound + n_tri))
                preds.append(pred_values)
        return i, idxs, preds

    results = Parallel(n_jobs=n_jobs)(
        delayed(group_gru_prediction)(subject, i) for i, subject in enumerate(subjects)
    )

    for i, idxs, preds in results:
        for (start, end), pred_values in zip(idxs, preds):
            data.loc[start:end-1, model] = pred_values

    data.to_csv(data_path, index=False)

#%%
# ### 3 - Group-wise GRU Predictions
# ### with (for train sequences) all sequences 
# ### except the sequences with the same outcome as the test sequences, 
# ### except the sequences with the same subject than the test sequences' one, 
# ### and the same number of train sequences than in subject-wise GRU
# for hidden_size in [32, 512, 1024, 2048]:
#     data = pd.read_csv(data_path)
#     model = f'group_GRU_{hidden_size}'
#     data[model] = np.full(n_sb*n_tri*n_seq, np.nan)
#     all_sequences = format_sequences(None, task, data)

#     def group_gru_prediction(subject, i):
#         print(f'Subject {subject} with id {i} processed for {model}')
#         subject_data = data[data['subject'] == subject]
#         subject_sequences = format_sequences(None, task, subject_data)
#         preds = []
#         idxs = []
#         for m in range(nb_test_seqs):
#             test_sequences = subject_sequences[n_test_seqs*m:n_test_seqs*(m + 1)]
#             test_outcomes = [np.array(seq[1]) for seq in test_sequences]
#             all_sequences_filtered = [
#                 seq for seq in all_sequences
#                 if not any(np.array_equal(np.array(seq[1]), test_outcome) for test_outcome in test_outcomes)
#             ]
#             all_sequences_filtered = [
#                 seq for seq in all_sequences_filtered
#                 if seq[3] != subject
#             ]
#             train_sequences = random.sample(all_sequences_filtered, n_seq - len(test_sequences))
#             GRU_group = train_and_evaluate_gru(train_sequences, test_sequences, hidden_size=hidden_size, return_pred=True)['sequence_pairs']
#             for p, tuple in enumerate(GRU_group):
#                 _, _, _, gru_sb_predicted_seq = tuple
#                 lower_bound = n_tri*n_seq*i + n_test_seqs*n_tri*m + n_tri*p
#                 pred_values = [q[0] for q in gru_sb_predicted_seq]
#                 idxs.append((lower_bound, lower_bound + n_tri))
#                 preds.append(pred_values)
#         return i, idxs, preds

#     results = Parallel(n_jobs=n_jobs)(
#         delayed(group_gru_prediction)(subject, i) for i, subject in enumerate(subjects)
#     )

#     for i, idxs, preds in results:
#         for (start, end), pred_values in zip(idxs, preds):
#             data.loc[start:end-1, model] = pred_values

#     data.to_csv(data_path, index=False)

# #%%
# ### 7 - Group-wise HMM Predictions
# ### with (for train sequences) all sequences 
# ### except the sequences with the same outcome as the test sequences, 
# ### except the sequences with the same subject than the test sequences' one, 
# ### if model == 'group_HMM':
#     ### and the same number of train sequences than in subject-wise HMM

# for model in ['group_HMM', 'big_group_HMM']:
#     data = pd.read_csv(data_path)
#     data[model] = np.full(n_sb*n_seq*n_tri, np.nan)

#     def hmm_group_prediction(subject, i):
#         print(f'Subject {subject} with id {i} processed for {model}')
#         subject_data = data[data['subject'] == subject]
#         subject_indices = subject_data.index.tolist()
#         print(subject_indices)
#         preds = np.full(len(subject_indices), np.nan)
#         for n in range(nb_test_seqs):
#             test_sequences = subject_data.iloc[n_test_seqs*n_tri*n: n_test_seqs*n_tri*(n + 1)]
            
#             train_data = data[data['subject'] != subject]
#             test_outcomes = [test_sequences.iloc[m*n_tri:(m + 1)*n_tri]['outcome'].values for m in range(int(len(test_sequences) / n_tri))]
#             sequences_to_remove = []
#             for m in range(int(len(train_data) / n_tri)):
#                 g_start = m * n_tri
#                 g_end = (m + 1) * n_tri
#                 group_outcome = train_data['outcome'].iloc[g_start:g_end].values
#                 if any(np.array_equal(group_outcome, t_o) for t_o in test_outcomes):
#                     sequences_to_remove.extend(range(g_start, g_end))
#             train_sequences = train_data.drop(index=train_data.index[sequences_to_remove]).reset_index(drop=True)

#             if model == 'group_HMM':
#                 # Randomly select (n_seq - len(test_sequences) // n_tri) sequences, each of length n_tri
#                 num_train_seqs = n_seq - (len(test_sequences) // n_tri)
#                 train_seq_indices = np.arange(0, len(train_sequences), n_tri)
#                 selected_indices = np.random.choice(train_seq_indices, size=num_train_seqs, replace=False)
#                 selected_rows = []
#                 for idx in selected_indices:
#                     selected_rows.extend(range(idx, idx + n_tri))
#                 train_sequences = train_sequences.iloc[selected_rows].reset_index(drop=True)

#             HMM_subject = predict_sequences_with_HMM(train_sequences, test_sequences, task, int(1 / (1 - train_size_ratio)))
#             start_idx = n_test_seqs*n_tri*n
#             preds[start_idx:start_idx + len(HMM_subject)] = HMM_subject
#         return subject, subject_indices, preds

#     results = Parallel(n_jobs=n_jobs)(
#         delayed(hmm_group_prediction)(subject, i) for i, subject in enumerate(subjects)
#     )

#     for subject, subject_indices, preds in results:
#         data.loc[subject_indices, model] = preds

#     data.to_csv(data_path, index=False)

### Temporary
# #%%
### 4 - Group-wise GRU Predictions
### with (for train sequences) all sequences
### except the sequences with the same outcome as the test sequences, 
### except the sequences with the same subject than the test sequences' one
for hidden_size in [32, 512, 1024]:
    data = pd.read_csv(data_path)
    model = f'big_group_RNN_2_{hidden_size}'
    data[model] = np.full(n_sb*n_tri*n_seq, np.nan)
    all_sequences = format_sequences(None, task, data)

    def big_group_gru_prediction(subject, i):
        print(f'Subject {subject} with id {i} processed for {model}')
        subject_data = data[data['subject'] == subject]
        subject_sequences = format_sequences(None, task, subject_data)
        preds = []
        idxs = []
        for m in range(nb_test_seqs):
            test_sequences = subject_sequences[n_test_seqs*m:n_test_seqs*(m + 1)]
            test_outcomes = [np.array(seq[1]) for seq in test_sequences]
            all_sequences_filtered = [
                seq for seq in all_sequences
                if not any(np.array_equal(np.array(seq[1]), test_outcome) for test_outcome in test_outcomes)
            ]
            all_sequences_filtered = [
                seq for seq in all_sequences_filtered
                if seq[3] != subject
            ]
            train_sequences = all_sequences_filtered  # Use all remaining as train
            GRU_big_group = train_and_evaluate_gru(train_sequences, test_sequences, hidden_size=hidden_size, return_pred=True, RNN=True)['sequence_pairs']
            for p, tuple in enumerate(GRU_big_group):
                _, _, _, gru_sb_predicted_seq = tuple
                lower_bound = n_tri*n_seq*i + n_test_seqs*n_tri*m + n_tri*p
                pred_values = [q[0] for q in gru_sb_predicted_seq]
                idxs.append((lower_bound, lower_bound + n_tri))
                preds.append(pred_values)
        return i, idxs, preds

    results = Parallel(n_jobs=n_jobs)(
        delayed(big_group_gru_prediction)(subject, i) for i, subject in enumerate(subjects)
    )

    for i, idxs, preds in results:
        for (start, end), pred_values in zip(idxs, preds):
            data.loc[start:end-1, model] = pred_values

    data.to_csv(data_path, index=False)

# #%%
### 4 - Group-wise GRU Predictions
### with (for train sequences) all sequences
### except the sequences with the same outcome as the test sequences, 
### except the sequences with the same subject than the test sequences' one
# for hidden_size in [1024]:
#     data = pd.read_csv(data_path)
#     model = f'big_group_GRU_{hidden_size}'
#     data[model] = np.full(n_sb*n_tri*n_seq, np.nan)
#     all_sequences = format_sequences(None, task, data)

#     def big_group_gru_prediction(subject, i):
#         print(f'Subject {subject} with id {i} processed for {model}')
#         subject_data = data[data['subject'] == subject]
#         subject_sequences = format_sequences(None, task, subject_data)
#         preds = []
#         idxs = []
#         for m in range(nb_test_seqs):
#             test_sequences = subject_sequences[n_test_seqs*m:n_test_seqs*(m + 1)]
#             test_outcomes = [np.array(seq[1]) for seq in test_sequences]
#             all_sequences_filtered = [
#                 seq for seq in all_sequences
#                 if not any(np.array_equal(np.array(seq[1]), test_outcome) for test_outcome in test_outcomes)
#             ]
#             all_sequences_filtered = [
#                 seq for seq in all_sequences_filtered
#                 if seq[3] != subject
#             ]
#             train_sequences = all_sequences_filtered  # Use all remaining as train
#             GRU_big_group = train_and_evaluate_gru(train_sequences, test_sequences, hidden_size=hidden_size, return_pred=True)['sequence_pairs']
#             for p, tuple in enumerate(GRU_big_group):
#                 _, _, _, gru_sb_predicted_seq = tuple
#                 lower_bound = n_tri*n_seq*i + n_test_seqs*n_tri*m + n_tri*p
#                 pred_values = [q[0] for q in gru_sb_predicted_seq]
#                 idxs.append((lower_bound, lower_bound + n_tri))
#                 preds.append(pred_values)
#         return i, idxs, preds

#     results = Parallel(n_jobs=n_jobs)(
#         delayed(big_group_gru_prediction)(subject, i) for i, subject in enumerate(subjects)
#     )

#     for i, idxs, preds in results:
#         for (start, end), pred_values in zip(idxs, preds):
#             data.loc[start:end-1, model] = pred_values

#     data.to_csv(data_path, index=False)

# #%%
# ### 5 - Subject-wise HMM Predictions
# data = pd.read_csv(data_path)
# model = 'subject_HMM'
# data[model] = np.full(n_sb*n_seq*n_tri, np.nan)

# def hmm_subject_prediction(subject, i):
#     print(f'Subject {subject} with id {i} processed for {model}')
#     subject_data = data[data['subject'] == subject]
#     subject_indices = subject_data.index.tolist()
#     preds = np.full(len(subject_indices), np.nan)
#     for n in range(nb_test_seqs):
#         test_sequences = subject_data.iloc[n_test_seqs*n_tri*n: n_test_seqs*n_tri*(n + 1)]
#         train_sequences = subject_data.drop(test_sequences.index)
#         HMM_subject = predict_sequences_with_HMM(train_sequences, test_sequences, task, int(1 / (1 - train_size_ratio)))
#         start_idx = n_test_seqs*n_tri*n
#         preds[start_idx:start_idx + len(HMM_subject)] = HMM_subject
#     return subject, subject_indices, preds

# results = Parallel(n_jobs=n_jobs)(
#     delayed(hmm_subject_prediction)(subject, i) for i, subject in enumerate(subjects)
# )

# for subject, subject_indices, preds in results:
#     data.loc[subject_indices, model] = preds

# data.to_csv(data_path, index=False)

# #%%
# ### 6 - Optimal HMM Predictions
# ### with fixed p_c = 1/20 (change-point probability)
# data = pd.read_csv(data_path)
# model = 'optimal_HMM'
# data[model] = np.full(n_sb*n_seq*n_tri, np.nan)

# def hmm_subject_prediction(subject, i):
#     print(f'Subject {subject} with id {i} processed for {model}')
#     subject_data = data[data['subject'] == subject]
#     subject_indices = data.index.tolist()
#     preds = np.full(len(subject_indices), np.nan)
#     for n in range(nb_test_seqs):
#         test_sequences = subject_data.iloc[n_test_seqs*n_tri*n: n_test_seqs*n_tri*(n + 1)]
#         train_sequences = subject_data.drop(test_sequences.index)
#         HMM_subject = predict_sequences_with_HMM(train_sequences, test_sequences, task, int(1 / (1 - train_size_ratio)), p_c_optimal=True)
#         start_idx = n_test_seqs*n_tri*n
#         preds[start_idx:start_idx + len(HMM_subject)] = HMM_subject
#     return subject, subject_indices, preds

# results = Parallel(n_jobs=n_jobs)(
#     delayed(hmm_subject_prediction)(subject, i) for i, subject in enumerate(subjects)
# )

# for subject, subject_indices, preds in results:
#     data.loc[subject_indices, model] = preds

# data.to_csv(data_path, index=False)

# #%%
# ### 8 - Fit NN to HMM to approximate decision function - group level
# ### For each subject, for each sequence, for each trial k, subject_HMM 
# ### predictions of trials k-(window_size + 1) to k are inputs of a feed-forward neural
# ### network trained to fit the estimate. The training data comes from:
# ### if level == 'subject':
#     ### the subject data without test data;
# ### if level == 'big_group':
#     ### the group data from which the concerned subject has been removed, and
#     ### sequences with the same outcomes are also removed from the training
#     ### data for better conservativity.
# ### For trials 1 to window_size - 1, the values of subject_HMM predictions
# ### remain unchanged.
# data = pd.read_csv(data_path)
# ref_model = 'subject_HMM' # better performance than 'group_HMM'
# target = 'estimate'
# nb_units = 16
# level = 'big_group' # 'big_group' or 'big_subject'
# model = f'{level}_HMM_with_FNN_{nb_units}'
# window_size = 10 #window_size
# max_iter = max_iter
# random_state = random_state

# data[model] = np.full(n_sb*n_seq*n_tri, np.nan)

# def predict_decision_function(train_data, test_data, nb_units):
#     # Prepare training sequences
#     train_seqs = [train_data.iloc[n*n_tri:(n + 1)*n_tri] for n in range(int(len(train_data) // n_tri))]
#     train_x = []
#     train_y = []

#     for seq in train_seqs:
#         for t in range(window_size - 1, n_tri):
#             x = seq.iloc[t + 1 - window_size : t + 1][ref_model].values.reshape(-1)
#             y = seq.iloc[t][target]
#             train_x.append(x)
#             train_y.append(y)

#     train_x = np.array(train_x)
#     train_y = np.array(train_y)

#     # Prepare test sequences
#     test_seqs = [test_data.iloc[n*n_tri:(n + 1)*n_tri] for n in range(int(len(test_data) // n_tri))]
#     test_x = []

#     for seq in test_seqs:
#         for t in range(window_size - 1, n_tri):
#             x = seq.iloc[t + 1 - window_size : t + 1][ref_model].values.reshape(-1)
#             test_x.append(x)

#     test_x = np.array(test_x)

#     # Train and predict
#     mlp = MLPRegressor(hidden_layer_sizes=(nb_units,), activation='logistic', solver='adam', max_iter=max_iter, random_state=random_state)
#     mlp.fit(train_x, train_y)
#     preds = mlp.predict(test_x)

#     return preds

# def process_subject_group_augmented_HMM(subject, i):
#     print(f"Subject {subject} with id {i} processed for {model}")
#     subject_mask = data['subject'] == subject
#     subject_data = data[subject_mask].copy()
#     sbj_start = i * n_seq * n_tri

#     # Initialize a NumPy array to store predictions
#     predictions_array = np.empty(len(subject_data))
#     predictions_array[:] = np.nan

#     for n in range(nb_test_seqs):
#         seqs_start = n * n_test_seqs * n_tri
#         seqs_end = (n + 1) * n_test_seqs * n_tri
#         test_data = subject_data.iloc[seqs_start:seqs_end]
#         test_outcomes = [test_data.loc[m * n_tri:(m + 1) * n_tri - 1, 'outcome'] for m in range(int(len(test_data) / n_tri))]
        
#         if level == 'subject':
#             train_data = subject_data.drop(test_data.index)

#         if level == 'big_group':
#             group_data = data[~subject_mask]
#             group_seqs = int(len(group_data) / n_tri)
#             sequences_to_remove = []

#             for m in range(group_seqs):
#                 g_start = m * n_tri
#                 g_end = (m + 1) * n_tri
#                 group_outcome = group_data['outcome'].iloc[g_start:g_end].values
#                 if any(np.array_equal(group_outcome, t_o) for t_o in test_outcomes):
#                     sequences_to_remove.extend(range(g_start, g_end))

#             train_data = group_data.drop(index=group_data.index[sequences_to_remove]).reset_index(drop=True)
        
#         preds = predict_decision_function(train_data, test_data, nb_units)
#         print("preds: ", preds)

#         for m in range(n_test_seqs):
#             seq_start = m * n_tri
#             seq_end = (m + 1) * n_tri
#             seq_global_idx = data.iloc[sbj_start + seqs_start + seq_start:sbj_start + seqs_start + seq_end].index

#             # Fill the first (window_size - 1) values with HMM baseline
#             first_part_idx = seq_global_idx[:window_size - 1]
#             for idx in first_part_idx:
#                 predictions_array[idx - sbj_start] = data.at[idx, ref_model]

#             # Fill remaining values with predictions
#             pred_part_idx = seq_global_idx[window_size - 1:]
#             pred_slice = preds[m * (n_tri - window_size + 1):(m + 1) * (n_tri - window_size + 1)]
#             for idx, pred in zip(pred_part_idx, pred_slice):
#                 predictions_array[idx - sbj_start] = pred
    
#     print(f"Done for subject {subject} with id {i} for {model}")
#     return(predictions_array)

# results = Parallel(n_jobs=n_jobs)(
#     delayed(process_subject_group_augmented_HMM)(subject, i) for i, subject in enumerate(subjects)
# )

# for i, subject in enumerate(subjects):
#     predictions_array = results[i]
#     start_idx = i*n_seq*n_tri
#     end_idx = start_idx + len(predictions_array)
#     data.loc[start_idx:end_idx - 1, model] = predictions_array

# data.to_csv(data_path, index=False)
