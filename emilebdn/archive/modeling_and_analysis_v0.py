### '20250709'
#%%
import datetime
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import seaborn as sns

from joblib import Parallel, delayed
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Add the root of the repository to sys.path
sys.path.append(op.dirname(op.dirname(op.dirname(__file__))))

from data_analysis_utils import fit_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from emilebdn.config.paths import (
    data_outcome_level_preprocessed_path,
    data_outcome_level_simulated_path
)

data_outcome_level_with_pred_path = data_outcome_level_preprocessed_path.replace('.csv', '_ada-prob_with_predictions.csv')

from emilebdn.config.variables import (
    n_jobs,
    nb_subjects,
    task_types,
    train_size_ratio,
    expID,
    length, 
    n_sequences_for_each_subject
)
from emilebdn.GRU.GRU_functions_2 import (
    flatten_sequences,
    format_sequences,
    plot_subject_sequence,
    train_and_evaluate_gru,
    train_and_evaluate_subject_gru
)
from emilebdn.HMM.HMM_functions import (
    HMM_prediction,
    predict_sequences_with_HMM
)

today = datetime.datetime.now().strftime("%Y%m%d")
model = 'GRU'
hidden_size = 1024
path = data_outcome_level_preprocessed_path.replace('.csv', '_ada-prob.csv')
task = 'ada-prob'

n_sb = nb_subjects
n_tri = length
n_seq = n_sequences_for_each_subject[task]  # Number of sequences for each subject

n_test_seqs = 3 # Number of sequences per test sequences set for model fitting
nb_test_seqs = 5 # Number of test sequences sets per subject

data_ada_prob = pd.read_csv(path)

subjects = pd.read_csv(path)['subject'].drop_duplicates().tolist()
subjects_nb = {
    subject: i for i, subject in enumerate(subjects)
}

data_outcome_level_with_pred = pd.read_csv(path)

#%%
### Mean subject behavior
# Load data
data_outcome_level_with_pred = pd.read_csv(data_outcome_level_with_pred_path)

data_outcome_level_with_pred['mean_estimate'] = np.full(n_sb*n_tri*n_seq, np.nan)

outcome_estimate_pairs = [(data_outcome_level_with_pred['outcome'].tolist()[i:i + n_tri], data_outcome_level_with_pred['estimate'].tolist()[i:i + n_tri]) for i in range(0, len(data_outcome_level_with_pred), n_tri)]
print(len(outcome_estimate_pairs))
for subject, i in subjects_nb.items():
    for n in range(n_seq):
        outcome = data_outcome_level_with_pred['outcome'].tolist()[n_tri*n_seq*i + n_tri*n:n_tri*n_seq*i + n_tri*(n + 1)]
        estimates = []
        for pair in outcome_estimate_pairs:
            if np.array_equal(np.array(pair[0]), np.array(outcome)):
                estimates.append(pair[1])
        mean_estimate = np.mean(estimates, axis=0)
        data_outcome_level_with_pred.iloc[n_tri*n_seq*i + n_tri*n:n_tri*n_seq*i + n_tri*(n + 1) - 1, 'mean_estimate'] = mean_estimate       

#%%
### Subject_wise GRU Predictions
# Load data
data_outcome_level_with_pred = pd.read_csv(data_outcome_level_with_pred_path)

data_outcome_level_with_pred[f'subject_GRU_{hidden_size}'] = np.full(n_sb*n_tri*n_seq, np.nan)

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
        test_sequences = subject_sequences[n_test_seqs*m:n_test_seqs*(m + 1)]
        train_sequences = subject_sequences[0:n_test_seqs*m] + subject_sequences[n_test_seqs*(m + 1):]
        GRU_subject = train_and_evaluate_gru(train_sequences, test_sequences, hidden_size=hidden_size, return_pred=True)['sequence_pairs']
        for p, tuple in enumerate(GRU_subject):
            _, _, _, gru_sb_predicted_seq = tuple
            lower_bound =  n_seq*n_tri*i + n_test_seqs*n_tri*m + n_tri*p
            # Assign predictions to the correct rows, preserving order
            pred_values = [q[0] for q in gru_sb_predicted_seq]
            data_outcome_level_with_pred.loc[lower_bound:lower_bound + n_tri - 1, f'subject_GRU_{hidden_size}'] = pred_values

data_outcome_level_with_pred.to_csv(path.replace('.csv', '_with_predictions.csv'), index=False)
print("Saved all predictions to CSV.")

#%%
### Group-wise GRU Predictions
### with (for train sequences) all sequences 
### except the sequences with the same outcome as the test sequences, 
### except the sequences with the same subject than the test sequences' one, 
### and the same number of train sequences than in subject-wise GRU

# Load data
data_outcome_level_with_pred = pd.read_csv(data_outcome_level_with_pred_path)

data_outcome_level_with_pred[f'group_GRU_{hidden_size}'] = np.full(n_sb*n_tri*n_seq, np.nan)
all_sequences = format_sequences(path, task, data_ada_prob)

def group_gru_prediction(subject, i):
    print(f"Processing subject {subject} with index {i}")
    subject_data = data_ada_prob[data_ada_prob['subject'] == subject]
    subject_sequences = format_sequences(subject_data, task, subject_data)
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
        train_sequences = random.sample(all_sequences_filtered, len(test_sequences))
        GRU_group = train_and_evaluate_gru(train_sequences, test_sequences, hidden_size=hidden_size, return_pred=True)['sequence_pairs']
        for p, tuple in enumerate(GRU_group):
            _, _, _, gru_sb_predicted_seq = tuple
            lower_bound = n_tri*n_seq*i + n_test_seqs*n_tri*m + n_tri*p
            pred_values = [q[0] for q in gru_sb_predicted_seq]
            idxs.append((lower_bound, lower_bound + n_tri))
            preds.append(pred_values)
    return i, idxs, preds

results = Parallel(n_jobs=n_jobs)(
    delayed(group_gru_prediction)(subject, i) for subject, i in subjects_nb.items()
)

for i, idxs, preds in results:
    for (start, end), pred_values in zip(idxs, preds):
        data_outcome_level_with_pred.loc[start:end-1, f'group_GRU_{hidden_size}'] = pred_values

data_outcome_level_with_pred.to_csv(path.replace('.csv', '_with_predictions.csv'), index=False)
print("Saved all predictions to CSV.")

#%%
### Group-wise GRU Predictions
### with (for train sequences) all sequences
### except the sequences with the same outcome as the test sequences, 
### except the sequences with the same subject than the test sequences' one

hidden_size = 32
# hidden_size = 512

# Load data
data_outcome_level_with_pred = pd.read_csv(data_outcome_level_with_pred_path)

data_outcome_level_with_pred[f'big_group_GRU_{hidden_size}'] = np.full(n_sb*n_tri*n_seq, np.nan)
all_sequences = format_sequences(path, task, data_ada_prob)

def big_group_gru_prediction(subject, i):
    print(f"Processing subject {subject} with index {i}")
    subject_data = data_ada_prob[data_ada_prob['subject'] == subject]
    subject_sequences = format_sequences(subject_data, task, subject_data)
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
        GRU_big_group = train_and_evaluate_gru(train_sequences, test_sequences, hidden_size=hidden_size, return_pred=True)['sequence_pairs']
        for p, tuple in enumerate(GRU_big_group):
            _, _, _, gru_sb_predicted_seq = tuple
            lower_bound = n_tri*n_seq*i + n_test_seqs*n_tri*m + n_tri*p
            pred_values = [q[0] for q in gru_sb_predicted_seq]
            idxs.append((lower_bound, lower_bound + n_tri))
            preds.append(pred_values)
    return i, idxs, preds

results = Parallel(n_jobs=n_jobs)(
    delayed(big_group_gru_prediction)(subject, i) for subject, i in subjects_nb.items()
)

for i, idxs, preds in results:
    for (start, end), pred_values in zip(idxs, preds):
        data_outcome_level_with_pred.loc[start:end-1, f'big_group_GRU_{hidden_size}'] = pred_values

data_outcome_level_with_pred.to_csv(path.replace('.csv', '_with_predictions.csv'), index=False)
print("Saved all predictions to CSV.")

#%%
### Subject-wise HMM Predictions
# Load data
data_outcome_level_with_pred = pd.read_csv(data_outcome_level_with_pred_path)

# Initialize the 'subject_HMM' column with NaN values
data_outcome_level_with_pred['subject_HMM'] = np.full(n_sb*n_seq*n_tri, np.nan)

def hmm_subject_prediction(subject):
    data = data_outcome_level_with_pred[data_outcome_level_with_pred['subject'] == subject]
    subject_indices = data.index.tolist()
    preds = np.full(len(subject_indices), np.nan)
    for n in range(nb_test_seqs):
        test_sequences = data.iloc[n_test_seqs*n_tri*n: n_test_seqs*n_tri*(n + 1)]
        train_sequences = data.drop(test_sequences.index)
        HMM_subject = predict_sequences_with_HMM(train_sequences, test_sequences, task, int(1 / (1 - train_size_ratio)))
        start_idx = n_test_seqs*n_tri*n
        preds[start_idx:start_idx + len(HMM_subject)] = HMM_subject
    return subject, subject_indices, preds

results = Parallel(n_jobs=n_jobs)(
    delayed(hmm_subject_prediction)(subject) for subject in subjects_nb.keys()
)

for subject, subject_indices, preds in results:
    data_outcome_level_with_pred.loc[subject_indices, 'subject_HMM'] = preds

# Save the final DataFrame with all predictions
data_outcome_level_with_pred.to_csv(
    data_outcome_level_with_pred,
    index=False
)
print("Saved all predictions to CSV.")

#%%
### Optimal HMM Predictions
# Load data
data_outcome_level_with_pred = pd.read_csv(data_outcome_level_with_pred_path)

# Initialize the 'subject_HMM' column with NaN values
data_outcome_level_with_pred['optimal_HMM'] = np.full(n_sb*n_seq*n_tri, np.nan)

def hmm_subject_prediction(subject):
    data = data_outcome_level_with_pred[data_outcome_level_with_pred['subject'] == subject]
    subject_indices = data.index.tolist()
    preds = np.full(len(subject_indices), np.nan)
    for n in range(nb_test_seqs):
        test_sequences = data.iloc[n_test_seqs*n_tri*n: n_test_seqs*n_tri*(n + 1)]
        train_sequences = data.drop(test_sequences.index)
        HMM_subject = predict_sequences_with_HMM(train_sequences, test_sequences, task, int(1 / (1 - train_size_ratio)), p_c_optimal=True)
        start_idx = n_test_seqs*n_tri*n
        preds[start_idx:start_idx + len(HMM_subject)] = HMM_subject
    return subject, subject_indices, preds

results = Parallel(n_jobs=n_jobs)(
    delayed(hmm_subject_prediction)(subject) for subject in subjects_nb.keys()
)

for subject, subject_indices, preds in results:
    data_outcome_level_with_pred.loc[subject_indices, 'optimal_HMM'] = preds

# Save the final DataFrame with all predictions
data_outcome_level_with_pred.to_csv(
    data_outcome_level_with_pred,
    index=False
)
print("Saved all predictions to CSV.")

# #%%
# ### Group-wise HMM Predictions
# # Load data
# data_outcome_level_with_pred = pd.read_csv(data_outcome_level_with_pred_path)

# # Initialize the 'group_HMM' column with NaN values
# data_outcome_level_with_pred['group_HMM'] = np.full(n_sb*n_seq*n_tri, np.nan)

# def hmm_group_prediction(subject):
#     data = data_outcome_level_with_pred[data_outcome_level_with_pred['subject'] == subject]
#     subject_indices = data.index.tolist()
#     preds = np.full(len(subject_indices), np.nan)
#     for n in range(nb_test_seqs):
#         test_sequences = data.iloc[n_test_seqs*n_tri*n: n_test_seqs*n_tri*(n + 1)]
        
#         # Découper test_sequences['outcome'] en une liste de np.array de taille 75
        
#         test_sequences_outcomes = [
#             np.array(test_sequences['outcome'].iloc[i:i + n_tri])
#             for i in range(0, len(test_sequences), n_tri)
#         ]

#         # Filter out from the full data any sequence whose outcome vector matches one in test_sequences_outcomes
#         train_sequences = []
#         data_outcomes = data_outcome_level_with_pred['outcome'].tolist()
#         for i in range(0, len(data_outcomes), n_tri):
#             candidate_outcome = np.array(data_outcomes[i:i + n_tri])
#             # Only keep if not in test_sequences_outcomes
#             if not any(np.array_equal(candidate_outcome, test_outcome) for test_outcome in test_sequences_outcomes):
#                 train_sequences.append(data_outcome_level_with_pred.iloc[i:i + n_tri])
        
#         # Concatenate the kept sequences back into a DataFrame
#         train_sequences = pd.concat(train_sequences, ignore_index=True)

#         # Randomly select 12 sequences of length n_tri from the candidate sequences
#         sequence_starts = list(range(0, len(train_sequences), n_tri))
#         selected_starts = np.random.choice(sequence_starts, size=12, replace=False)
#         train_sequences = pd.concat(
#             [train_sequences.iloc[start:start + n_tri] for start in selected_starts],
#             ignore_index=True
#         )

#         print(f"Length of test_sequences: {len(test_sequences)}")
#         print(f"Length of train_sequences: {len(train_sequences)}")

#         HMM_group = predict_sequences_with_HMM(train_sequences, test_sequences, task, int(1 / (1 - train_size_ratio)))
#         start_idx = n_test_seqs*n_tri*n
#         preds[start_idx:start_idx + len(HMM_group)] = HMM_group
#     return subject, subject_indices, preds

# results = Parallel(n_jobs=n_jobs)(
#     delayed(hmm_group_prediction)(subject) for subject in subjects_nb.keys()
# )

# for subject, subject_indices, preds in results:
#     data_outcome_level_with_pred.loc[subject_indices, 'group_HMM'] = preds

# # Save the final DataFrame with all predictions
# data_outcome_level_with_pred.to_csv(
#     data_outcome_level_with_pred,
#     index=False
# )
# print("Saved all predictions to CSV.")

# #%%
# mse_group_hmm = mean_squared_error(
#     data_outcome_level_with_pred['estimate'],
#     data_outcome_level_with_pred['big_group_GRU']
# )
# evf_group_hmm = explained_variance_score(
#     data_outcome_level_with_pred['estimate'],
#     data_outcome_level_with_pred['big_group_GRU']
# )

# print(f"MSE between 'estimate' and 'big_group_GRU': {mse_group_hmm}")
# print(f"Explained variance between 'estimate' and 'big_group_GRU': {evf_group_hmm}")

# #%%
# data_outcome_level_with_pred = pd.read_csv(data_outcome_level_with_pred_path)
# data_outcome_level_with_pred = data_outcome_level_with_pred.rename(columns={
#     'group_GRU': 'group_GRU_512',
#     'subject_GRU': 'subject_GRU_512',
#     'big_group_GRU': 'big_group_GRU_32'
# })
# data_outcome_level_with_pred.to_csv(data_outcome_level_with_pred_path, index=False)

#%%
# Load data
data_outcome_level_with_pred = pd.read_csv(data_outcome_level_with_pred_path)

metrics = ['mean_estimate', 'subject_GRU_512', 'subject_GRU_1024', 'group_GRU_512', 'big_group_GRU_32', 'big_group_GRU_512', 'subject_HMM', 'optimal_HMM']
results = []

for col in metrics:
    y_true = data_outcome_level_with_pred['estimate']
    y_pred = data_outcome_level_with_pred[col]
    mse = mean_squared_error(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    results.append({'model': col, 'mse': mse, 'evs': evs})

results_df = pd.DataFrame(results)
results_df.to_csv(data_outcome_level_preprocessed_path.replace('.csv', '_ada-prob_with_predictions_mse_evs_vs_estimate.csv'), index=False)
print(results_df)

# #%%
# Extract only the first row of each sequence (assuming n_tri = 75)
n_tri = length  # already defined above
first_rows = data_outcome_level_with_pred.iloc[::n_tri].copy()

metrics = ['estimate'] + metrics

def compute_stats(df, metrics):
    stats = []
    for col in metrics:
        vals = df[col]
        stats.append({
            'model': col,
            'mean': np.nanmean(vals),
            'std': np.nanstd(vals)
        })
    return pd.DataFrame(stats)

# No conditioning
overall_stats = compute_stats(first_rows, metrics)
print("Overall stats (no conditioning):")
print(overall_stats)

# Conditioned on outcome == 1
stats_outcome_1 = compute_stats(first_rows[first_rows['outcome'] == 1], metrics)
print("\nStats conditioned on outcome == 1:")
print(stats_outcome_1)

# Conditioned on outcome == 0
stats_outcome_0 = compute_stats(first_rows[first_rows['outcome'] == 0], metrics)
print("\nStats conditioned on outcome == 0:")
print(stats_outcome_0)

# Save the overall_stats, stats_outcome_1, and stats_outcome_0 to CSV
all_stats = pd.concat([
    overall_stats.assign(condition='all'),
    stats_outcome_1.assign(condition='outcome==1'),
    stats_outcome_0.assign(condition='outcome==0')
], ignore_index=True)
all_stats.to_csv(data_outcome_level_preprocessed_path.replace('.csv', '_ada-prob_with_predictions_first_outcome_stats.csv'), index=False)

# #%%

# # Compute MSE and EVF between all pairs of metrics and display as square matrices
# metrics = ['estimate', 'mean_estimate', 'subject_GRU_512', 'group_GRU_512', 'big_group_GRU_32', 'big_group_GRU_512', 'subject_HMM', 'optimal_HMM']


# mse_matrix = pd.DataFrame(index=metrics, columns=metrics, dtype=float)
# evf_matrix = pd.DataFrame(index=metrics, columns=metrics, dtype=float)

# for m1 in metrics:
#     for m2 in metrics:
#         y1 = data_outcome_level_with_pred[m1]
#         y2 = data_outcome_level_with_pred[m2]
#         mse_matrix.loc[m1, m2] = mean_squared_error(y1, y2)
#         evf_matrix.loc[m1, m2] = explained_variance_score(y1, y2)

# mse_matrix.to_csv(data_outcome_level_preprocessed_path.replace('.csv', '_ada-prob_with_predictions_mse_matrix.csv'))
# evf_matrix.to_csv(data_outcome_level_preprocessed_path.replace('.csv', '_ada-prob_with_predictions_evs_matrix.csv'))
# print("MSE matrix:\n", mse_matrix)
# print("\nEVF matrix:\n", evf_matrix)

# #%%
data_outcome_level_with_pred = pd.read_csv(data_outcome_level_with_pred_path)
models = metrics
reference = 'hidden_parameter'

# Compute linear regression parameters for each model vs reference and store in a DataFrame
regression_results = []

for model in models:
    if model == reference:
        # For the reference variable, linregress against itself
        slope, intercept, r_value, p_value, std_err = 1.0, 0.0, 1.0, 0.0, 0.
    else:
        df = data_outcome_level_with_pred[[reference, model]].dropna()
        if len(df) < 2:
            slope, intercept, r_value, p_value, std_err = [np.nan]*5
        else:
            slope, intercept, r_value, p_value, std_err = linregress(df[reference], df[model])
    regression_results.append({
        'model': model,
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err
    })

regression_df = pd.DataFrame(regression_results)
print(regression_df)
regression_df.to_csv(
    data_outcome_level_preprocessed_path.replace(
        '.csv', '_ada-prob_with_predictions_regression_vs_hidden_parameter_results.csv'
    ),
    index=False
)

# Plotting (sampled, to avoid memory issues)
n_cols = 2
n_rows = int(np.ceil(len(models) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7*n_rows))  # Make figure larger for square axes
axes = axes.flatten()

for idx, model in enumerate(models):
    ax = axes[idx]
    # Plot model vs reference as before
    if model == reference:
        df = data_outcome_level_with_pred[[reference]].dropna()
        if len(df) > 200:
            df_sample = df.sample(n=200, random_state=42)
        else:
            df_sample = df
        ax.scatter(df_sample[reference], df_sample[reference], alpha=0.5)
        ax.plot(df_sample[reference], df_sample[reference], color='red', label='y=x')
        ax.set_title(f'{reference} vs {reference}\n$y = x$')
        ax.set_xlabel(reference)
        ax.set_ylabel(reference)
        ax.set_aspect('equal', adjustable='datalim')
        # Make axes square
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        continue
    df = data_outcome_level_with_pred[[reference, model]].dropna()
    if len(df) > 200:
        df_sample = df.sample(n=200, random_state=42)
    else:
        df_sample = df
    reg_row = regression_df[regression_df['model'] == model]
    slope = reg_row['slope'].values[0]
    intercept = reg_row['intercept'].values[0]
    sns.regplot(
        data=df_sample,
        x=reference,
        y=model,
        scatter=True,
        line_kws={'color': 'red'},
        ax=ax
    )
    ax.set_title(f'{model} vs {reference}\n$y = {slope:.3f}x + {intercept:.3f}$')
    ax.set_xlabel(reference)
    ax.set_ylabel(model)
    ax.set_aspect('equal', adjustable='datalim')
    # Make axes square
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.set_xlim(lims)
    ax.set_ylim(lims)

# Hide unused subplots
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# #%%
### Figure 5
data_outcome_level_with_pred = pd.read_csv(data_outcome_level_with_pred_path)

# Pour chaque séquence (75 lignes), détecter les change points et faire la régression demandée

n_tri = length  # nombre de trials par séquence
window_before = -5
window_after = 15

windows_by_position = {k: [] for k in range(window_before, window_after + 1)}                

n_seq_total = len(data_outcome_level_with_pred) // n_tri

def process_sequence(seq_idx):
    print(f"Processing sequence {seq_idx}")
    seq_start = seq_idx*n_tri
    seq_end = seq_start + n_tri
    sequence_df = data_outcome_level_with_pred.iloc[seq_start:seq_end].copy()
    sequence_df.reset_index(drop=True, inplace=True)
    
    change_points = sequence_df.index[
        (sequence_df['did_change_point_occur'] == True) & (sequence_df.index > 0)
    ].tolist()
    
    rows_by_position = {k: [] for k in range(window_before, window_after + 1)}
    for cp_idx in change_points:
        for rel_pos in range(window_before, window_after + 1):
            abs_idx = cp_idx + rel_pos
            if 0 <= abs_idx < n_tri - 1:
                row = sequence_df.iloc[abs_idx].copy()
                row['relative_position'] = rel_pos
                row['hidden_parameter_before'] = sequence_df.iloc[cp_idx - 1]['hidden_parameter'] 
                row['hidden_parameter_after'] = sequence_df.iloc[cp_idx]['hidden_parameter']
                rows_by_position[rel_pos].append(row)
    return rows_by_position

# Parallelize over all sequences
all_rows_by_position = Parallel(n_jobs=50)(
    delayed(process_sequence)(seq_idx) for seq_idx in range(n_seq_total)
)

# Aggregate results into windows_by_position
for rows_by_position in all_rows_by_position:
    for rel_pos, rows in rows_by_position.items():
        windows_by_position[rel_pos].extend(rows)
    
regression_by_position = {k: [] for k in windows_by_position.keys()}

def regression_for_position(rel_pos, rows):
    print(f"Running multilinear regression for relative position {rel_pos} with {len(rows)} rows")
    results = []
    if not rows:
        return rel_pos, results
    df = pd.DataFrame(rows)
    for model in models:
        subdf = df[[model, 'hidden_parameter_before', 'hidden_parameter_after']].dropna()
        if len(subdf) < 3:
            results.append({
                'model': model,
                'n': len(subdf),
                'a': np.nan,
                'b': np.nan,
                'c': np.nan,
                'r2': np.nan
            })
            continue
        X = subdf[['hidden_parameter_before', 'hidden_parameter_after']].values
        y = subdf[model].values
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        a, b = reg.coef_
        #c = reg.intercept_
        r2 = reg.score(X, y)
        results.append({
            'model': model,
            'n': len(subdf),
            'a': a,
            'b': b,
            #'c': c,
            'r2': r2
        })
    return rel_pos, results

parallel_results = Parallel(n_jobs=50)(
    delayed(regression_for_position)(rel_pos, rows)
    for rel_pos, rows in windows_by_position.items()
)

for rel_pos, results in parallel_results:
    regression_by_position[rel_pos] = results

    # Print regression results by relative position
    for rel_pos in sorted(regression_by_position.keys()):
        print(f"Relative position {rel_pos}:")
        for res in regression_by_position[rel_pos]:
            print(res)
        print("-"*40)

# Save regression results to CSV
# Save regression results to CSV for each relative position
all_regression_rows = []
for rel_pos, results in regression_by_position.items():
    for res in results:
        row = {'relative_position': rel_pos}
        row.update(res)
        all_regression_rows.append(row)
regression_by_position_df = pd.DataFrame(all_regression_rows)
regression_by_position_df.to_csv(
    data_outcome_level_preprocessed_path.replace(
        '.csv', '_ada-prob_with_predictions_figure_5_results.csv'
    ),
    index=False
)

#%%
# Plot 'a' (dashed) and 'b' (solid) coefficients vs relative position for each model
# Use the same color for both lines of the same model

def plot_regression_coefficients(regression_by_position, models, display_models=None):
    """
    Plot regression coefficients 'a' (dashed) and 'b' (solid) vs relative position for each model.

    Args:
        regression_by_position: dict mapping relative position to list of regression result dicts.
        models: list of model names.
        display_models: list of model names to display (default: all in models).
    """
    positions = sorted(regression_by_position.keys())
    model_names = models
    if display_models is None:
        display_models = model_names

    # Assign a unique color to each model
    cmap = plt.get_cmap('tab10')
    color_map = {model: cmap(i % 10) for i, model in enumerate(model_names)}

    plt.figure(figsize=(16, 12))
    for model in model_names:
        if model not in display_models:
            continue
        a_values = []
        b_values = []
        for pos in positions:
            res_list = regression_by_position[pos]
            a = np.nan
            b = np.nan
            for res in res_list:
                if res['model'] == model:
                    a = res['a']
                    b = res['b']
                    break
            a_values.append(a)
            b_values.append(b)
        color = color_map[model]
        plt.plot(positions, a_values, marker='o', linestyle='--', color=color, label=f"{model} (hidden_param_bef, dashed)")
        plt.plot(positions, b_values, marker='s', linestyle='-', color=color, label=f"{model} (hidden_param_aft, solid)")

    plt.xlabel('Relative Position')
    plt.ylabel("Regression coefficients 'hidden_param_bef' (dashed), 'hidden_param_aft' (solid)")
    plt.title("Coefficients 'hidden_param_bef' (dashed) and 'hidden_param_aft' (solid) vs Relative Position for each model")
    plt.axvline(x=0, color='red', linestyle='--', label='Change-point')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

# Example usage:
plot_regression_coefficients(regression_by_position, models, display_models= None)
plot_regression_coefficients(regression_by_position, models, display_models= ['estimate', 'subject_GRU_512', 'subject_HMM'])

#%%
### HMM Parameter Recovery
simulated_outcomes = pd.read_csv(data_outcome_level_simulated_path)
task = 'ada-prob'
model = 'HMM'

simulated_outcomes = simulated_outcomes[simulated_outcomes['task'] == task]
simulated_outcomes = simulated_outcomes[['subject', 'session_idx', 'outcome']]

subjects = simulated_outcomes['subject'].unique()

generative_p_c = {subject: np.random.uniform(0.53, 0.57) for subject in subjects}

fitted_p_c = {subject: None for subject in subjects}

def fit_subject_p_c(i, subject):
    subject_outcomes = simulated_outcomes[simulated_outcomes['subject'] == subject]
    predictions = HMM_prediction(generative_p_c[subject], subject_outcomes, task)
    sessions_idx = subject_outcomes['session_idx'].unique()
    return subject, fit_model(expID, model, i, sessions_idx)[1]

results = Parallel(n_jobs=n_jobs)(
    delayed(fit_subject_p_c)(i, subject) for i, subject in enumerate(subjects)
)
for subject, fit_val in results:
    fitted_p_c[subject] = fit_val

gen_p_c = np.array([generative_p_c[subject] for subject in subjects])
fit_p_c = np.array([fitted_p_c[subject] for subject in subjects])

mse_p_c = mean_squared_error(gen_p_c, fit_p_c)
print(f"MSE between generative_p_c and fitted_p_c: {mse_p_c}")

# #%%
# ### Fit NN to HMM to approximate decision function
# data = pd.read_csv(data_outcome_level_with_pred_path)
# task = 'ada-prob'
# nb_units = 512

# data = data[data['task'] == task]
# subjects = data['subject'].unique()
# data['augmented_HMM'] = np.full(n_sb*n_seq*n_tri, np.nan)

# def predict_decision_function(train_data, test_data, nb_units):
#     # Prepare input and output from training data
#     X_train = train_data['subject_HMM'].values.reshape(-1,1)
#     y_train = train_data['estimate'].values

#     # Prepare input from test data
#     X_test = test_data['subject_HMM'].values.reshape(-1,1)

#     # Define and fit the MLP regressor
#     mlp = MLPRegressor(hidden_layer_sizes=(nb_units,), activation='relu', max_iter=500, random_state=42)
#     mlp.fit(X_train, y_train)

#     # Predict outcomes for the test data
#     y_pred = mlp.predict(X_test)

#     return y_pred

# # def augmented_subject(i, subject):
# for i, subject in enumerate(subjects):
#     subject_data = data[data['subject'] == subject]
#     sbj_bound = i*n_seq*n_tri
#     for n in range(nb_test_seqs):
#         start = n*n_test_seqs*n_tri 
#         end = (n + 1)*n_test_seqs*n_tri
#         test_data = subject_data.iloc[start:end]
#         train_data = subject_data.drop(test_data.index)
#         augmented_HMM = predict_decision_function(train_data, test_data, nb_units)
#         abs_start = sbj_bound + start
#         abs_end = sbj_bound + end 
#         data.loc[abs_start:abs_end-1, 'augmented_HMM'] = augmented_HMM
#     print(f'Done for subject {subject}')

# # Parallel(n_jobs=n_jobs)(
# #     delayed(augmented_subject)(i, subject) for i, subject in enumerate(subjects)
# # )

# data.to_csv(data_outcome_level_with_pred)
# print("saved")

# mse = mean_squared_error(data['estimate'], data['augmented_HMM'])
# evs = explained_variance_score(data['estimate'], data['augmented_HMM'])
# print(mse)
# print(evs)

# #%%
# ### Fit NN to HMM to approximate decision function - subject level

# data = pd.read_csv(data_outcome_level_with_pred_path)
# task = 'ada-prob'
# nb_units = 512
# window_size = 5 
# max_iter = 500
# random_state = 42
# data = data[data['task'] == task]
# subjects = data['subject'].unique()

# # Define the prediction function
# def predict_decision_function(train_data, test_data, nb_units):
#     train_seqs = [train_data[n*n_tri:(n + 1)*n_tri] for n in range(int(len(train_data)/n_tri))]
#     train_tuples_x = [seq[t + 1-window_size:t + 1]['subject_HMM'].reshape(-1,1) for t in range(window_size, n_tri) for seq in train_seqs]
#     train_tuples_y = [seq[t + 1-window_size:t + 1]['estimate'] for t in range(window_size, n_tri) for seq in train_seqs]

#     test_seqs = [test_data[n*n_tri:(n + 1)*n_tri] for n in range(int(len(test_data)/n_tri))]   
#     test_tuples_x = [seq[t + 1-window_size:t + 1].reshape(-1,1) for t in range(window_size, n_tri) for seq in test_seqs]

#     mlp = MLPRegressor(hidden_layer_sizes=(nb_units,), activation='relu', max_iter=max_iter, random_state=random_state)
    
#     # train
#     mlp.fit(train_tuples_x, train_tuples_y)

#     # test
#     y_pred = mlp.predict(test_tuples_x)
#     return y_pred

# for i, subject in enumerate(subjects):
#     subject_data = data[data['subject'] == subject]
#     sbj_bound = i*n_seq*n_tri
#     for n in range(nb_test_seqs):
#         start = n*n_test_seqs*n_tri
#         end = (n + 1)*n_test_seqs*n_tri
#         test_data = subject_data.iloc[start:end]
#         train_data = subject_data.drop(test_data.index)
#         subject_augmented_HMM = predict_decision_function(train_data, test_data, nb_units)
#         abs_start = sbj_bound + start
#         abs_end = sbj_bound + end
#         data.loc[abs_start:abs_end - 1, 'subject_augmented_HMM'] = subject_augmented_HMM
#     print(f'Done for subject {subject}')

# # Save the updated data
# data.to_csv(data_outcome_level_with_pred)
# print("Saved")

# # Calculate metrics
# mse = mean_squared_error(data['estimate'], data['subject_augmented_HMM'])
# evs = explained_variance_score(data['estimate'], data['subject_augmented_HMM'])
# print(f"MSE: {mse}")
# print(f"Explained Variance Score: {evs}")

#%%
### Fit NN to HMM to approximate decision function - subject level

data = pd.read_csv(data_outcome_level_with_pred_path)
task = 'ada-prob'
nb_units = 512
window_size = 1
max_iter = 1000
random_state = 42

data = data[data['task'] == task]
subjects = data['subject'].unique()

# Initialize column
data['subject_augmented_HMM'] = np.nan

def predict_decision_function(train_data, test_data, nb_units):
    # Prepare training sequences
    train_seqs = [train_data.iloc[n*n_tri:(n + 1)*n_tri] for n in range(int(len(train_data) // n_tri))]
    train_x = []
    train_y = []

    for seq in train_seqs:
        for t in range(window_size, n_tri):
            x = seq.iloc[t + 1 - window_size : t + 1]['subject_HMM'].values.reshape(-1)
            y = seq.iloc[t]['estimate']
            train_x.append(x)
            train_y.append(y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # Prepare test sequences
    test_seqs = [test_data.iloc[n*n_tri:(n + 1)*n_tri] for n in range(int(len(test_data) // n_tri))]
    test_x = []

    for seq in test_seqs:
        for t in range(window_size, n_tri):
            x = seq.iloc[t + 1 - window_size : t + 1]['subject_HMM'].values.reshape(-1)
            test_x.append(x)

    test_x = np.array(test_x)

    # Train and predict
    mlp = MLPRegressor(hidden_layer_sizes=(nb_units,), activation='logistic', solver='adam', max_iter=max_iter, random_state=random_state)
    mlp.fit(train_x, train_y)
    preds = mlp.predict(test_x)

    return preds

# Process each subject
for i, subject in enumerate(subjects):
    subject_data = data[data['subject'] == subject]
    sbj_bound = i*n_seq*n_tri

    for n in range(nb_test_seqs):
        start = n*n_test_seqs*n_tri
        end = (n + 1)*n_test_seqs*n_tri
        test_data = subject_data.iloc[start:end]
        train_data = subject_data.drop(test_data.index)

        preds = predict_decision_function(train_data, test_data, nb_units)

        abs_start = sbj_bound + start
        abs_end = sbj_bound + end

        for m in range(n_test_seqs):
            data.loc[abs_start + m*n_tri:abs_start + m*n_tri + window_size - 1, 'subject_augmented_HMM'] = \
                data.loc[abs_start + m*n_tri:abs_start + m*n_tri + window_size - 1, 'subject_HMM']
            data.loc[abs_start + m*n_tri + window_size:abs_start + (m+1)*n_tri - 1, 'subject_augmented_HMM'] = \
                preds[m*(n_tri - window_size):(m + 1)*(n_tri - window_size)]

    print(f"Done for subject {subject}")

# Save updated data
data.to_csv(data_outcome_level_with_pred_path, index=False)
print("Saved")

mse = mean_squared_error(data['estimate'], data['subject_augmented_HMM'])
evs = explained_variance_score(data['estimate'], data['subject_augmented_HMM'])

print(f"MSE: {mse}")
print(f"Explained Variance Score: {evs}")

#%%
### Fit NN to HMM to approximate decision function - group level

data = pd.read_csv(data_outcome_level_with_pred_path)
task = 'ada-prob'
nb_units = 1024
window_size = 10
max_iter = 1000
random_state = 42

data = data[data['task'] == task]
subjects = data['subject'].unique()

# Initialize column
data['group_augmented_HMM'] = np.nan

def predict_decision_function(train_data, test_data, nb_units):
    # Prepare training sequences
    train_seqs = [train_data.iloc[n*n_tri:(n + 1)*n_tri] for n in range(int(len(train_data) // n_tri))]
    train_x = []
    train_y = []

    for seq in train_seqs:
        for t in range(window_size, n_tri):
            x = seq.iloc[t + 1 - window_size : t + 1]['subject_HMM'].values.reshape(-1)
            y = seq.iloc[t]['estimate']
            train_x.append(x)
            train_y.append(y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # Prepare test sequences
    test_seqs = [test_data.iloc[n*n_tri:(n + 1)*n_tri] for n in range(int(len(test_data) // n_tri))]
    test_x = []

    for seq in test_seqs:
        for t in range(window_size, n_tri):
            x = seq.iloc[t + 1 - window_size : t + 1]['subject_HMM'].values.reshape(-1)
            test_x.append(x)

    test_x = np.array(test_x)

    # Train and predict
    mlp = MLPRegressor(hidden_layer_sizes=(nb_units,), activation='logistic', solver='adam', max_iter=max_iter, random_state=random_state)
    mlp.fit(train_x, train_y)
    preds = mlp.predict(test_x)

    return preds

# Process each subject
# for i, subject in enumerate(subjects):
def process_subject_group_augmented_HMM(i, subject): 
    print(f'Subject {subject} processed')
    subject_data = data[data['subject'] == subject]
    sbj_bound = i*n_seq*n_tri

    for n in range(nb_test_seqs):
        start = n*n_test_seqs*n_tri
        end = (n + 1)*n_test_seqs*n_tri
        test_data = subject_data.iloc[start:end]
        
        group_data = data[data['subject'] != subject]
        test_sequences = [test_data.loc[m*n_tri:(m+1)*n_tri - 1, 'outcome'] for m in range(int(len(test_data)/n_tri))]
        
        # Remove sequences from group_data that match any in test_sequences
        sequences_to_remove = []
        for m in range(int(len(group_data) / n_tri)):
            seq_start = m * n_tri
            seq_end = (m + 1) * n_tri
            current_seq = group_data.loc[seq_start:seq_end - 1, 'outcome'].values
            if any(np.array_equal(current_seq, ts) for ts in test_sequences):
                sequences_to_remove.extend(range(seq_start, seq_end))

        # Drop the identified sequences using iloc
        train_data = group_data.drop(group_data.index[sequences_to_remove]).reset_index(drop=True)

        preds = predict_decision_function(train_data, test_data, nb_units)

        abs_start = sbj_bound + start
        abs_end = sbj_bound + end

        for m in range(n_test_seqs):
            data.loc[abs_start + m*n_tri:abs_start + m*n_tri + window_size - 1, 'group_augmented_HMM'] = \
                data.loc[abs_start + m*n_tri:abs_start + m*n_tri + window_size - 1, 'subject_HMM']
            data.loc[abs_start + m*n_tri + window_size:abs_start + (m+1)*n_tri - 1, 'group_augmented_HMM'] = \
                preds[m*(n_tri - window_size):(m + 1)*(n_tri - window_size)]

    print(f"Done for subject {subject}")

Parallel(n_jobs=n_jobs)(
    delayed(process_subject_group_augmented_HMM)(i, subject) for i, subject in enumerate(subjects)
)

# Save updated data
data.to_csv(data_outcome_level_with_pred_path, index=False)
print("Saved")

mse = mean_squared_error(data['estimate'], data['group_augmented_HMM'])
evs = explained_variance_score(data['estimate'], data['group_augmented_HMM'])

print(f"MSE: {mse}")
print(f"Explained Variance Score: {evs}")
#%%
