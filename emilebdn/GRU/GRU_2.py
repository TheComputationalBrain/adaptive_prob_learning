"""
GRU Sequence Prediction Pipeline
--------------------------------
This script runs a GRU-based recurrent neural network pipeline for human learning studies.
It includes hyperparameter tuning, subject-wise and group-level evaluation on simulated and real data.

Author: @emilebdn
Created date: 2025-06-06
"""
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
    computed_data_emile_path,
    data_outcome_level_preprocessed_path,
    data_outcome_level_simulated_path
)

data_outcome_level_with_pred_path = data_outcome_level_preprocessed_path.replace('.csv', '_ada-prob_with_predictions.csv')

from emilebdn.config.variables import (
    n_jobs,
    random_state,
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

##########################################################################################
#%% Section 4 — Train Subject-wise GRU 
print("\n=== Section 4: Subject-wise GRU Training ===")

model = 'GRU'
model_config = 'subject'
hidden_size = 1024
data = 'experiment'
data_path = data_outcome_level_preprocessed_path
task = 'ada-prob'
content = 'mse_evf_scores'

print("model:", model)
print("model_config:", model_config)
print("hidden_size:", hidden_size)
print("data:", data)
print("data_path:", data_path)
print("task_types:", task_types)
print("content:", content)

path = data_outcome_level_preprocessed_path

all_flat_results = {}

sequences = format_sequences(path, task)

subject_ids = sorted(set(seq[3] for seq in sequences))

subjects_sequences = {subject_id: [seq for seq in sequences if seq[3] == subject_id] for subject_id in subject_ids}

results = Parallel(n_jobs=n_jobs)(delayed(train_and_evaluate_subject_gru)(subjects_sequences[subject_id], subject_id, data, task, train_size_ratio, \
                                                            hidden_size) for subject_id in subject_ids)

for subject_id, result in results:
    all_flat_results[(task, subject_id)] = result

results_df = pd.DataFrame.from_dict(all_flat_results, orient='index').reset_index(drop=True)
# results_df = results_df[['data', 'task', 'hidden_size', 'subject_id', 'training_time', 'test_loss', 'explained_variance']]

results_path = op.join(computed_data_emile_path, model, model_config, data, \
                                 f"{today}_{model}_{model_config}_{data}_{task}_{content}.csv")
#results_df.to_csv(results_path, index=False)
print(f"Saved subject-wise GRU results to: {results_path}")

date = '20250604'
results_path = op.join(computed_data_emile_path, model, model_config, data, \
                                 f"{date}_{model}_{model_config}_{data}_{task}_{content}.csv")
results_df = pd.read_csv(results_path)

evf_col = 'evf'
mse_col = 'mse'

# Compute stats for EVF and MSE
evfs = results_df[evf_col]
mses = results_df[mse_col]

stats = pd.DataFrame({
    'min': [mses.min(), evfs.min()],
    'max': [mses.max(), evfs.max()],
    'mean': [mses.mean(), evfs.mean()],
    'std': [mses.std(), evfs.std()]
}, index=['mse', 'evf']).reset_index().rename(columns={'index': 'metric'})

evf_stats_path = results_path.replace('.csv', '_stats.csv')
#stats.to_csv(evf_stats_path, index=False)
print(f"Saved explained variance and mse stats to: {evf_stats_path}")

##########################################################################################
#%% Section 5 — Train Subject-wise GRU with various hidden layer sizes 
print("\n=== Section 5: Subject-wise GRU Training with various hidden layer sizes ===")

model = 'GRU'
model_config = 'subject'
data = 'simulation'
data_path = data_outcome_level_preprocessed_path
task_types = ['ada-prob']
content = 'mse_evf_scores'

print("model:", model)
print("model_config:", model_config)
print("data:", data)
print("data_path:", data_path)
print("task_types:", task_types)
print("content:", content)

path = data_outcome_level_preprocessed_path
hidden_sizes = [2**i for i in range(0, 12)]  # [1, 2, 4, ..., 2048]

all_stats_by_hidden_size = []

for hidden_size in hidden_sizes:
    print(f"\n=== Training subject GRU with hidden size: {hidden_size} ===")
    
    all_flat_results = {}

    for task in task_types:
        sequences = format_sequences(path, task)
        subject_ids = sorted(set(seq[3] for seq in sequences))
        subjects_sequences = {subject_id: [seq for seq in sequences if seq[3] == subject_id] for subject_id in subject_ids}
        
        results = Parallel(n_jobs=n_jobs)(delayed(train_and_evaluate_subject_gru)(
            subjects_sequences[subject_id], subject_id, data, task, train_size_ratio,
            hidden_size
        ) for subject_id in subject_ids)
        
        for subject_id, result in results:
            all_flat_results[(task, subject_id)] = result

    # Create DataFrame
    results_df = pd.DataFrame.from_dict(all_flat_results, orient='index').reset_index(drop=True)

    # Save full results (optional per hidden size)
    results_path = op.join(computed_data_emile_path, model, model_config, data,
        f"{today}_{model}_{model_config}_{data}_{content}_hs_{hidden_size}.csv")
    #results_df.to_csv(results_path, index=False)
    print(f"Saved subject GRU results (hidden size {hidden_size}) to: {results_path}")

    # Compute stats
    evf_col = 'evf'
    mse_col = 'mse'
    for task in results_df['task'].unique():
        evfs = results_df[results_df['task'] == task][evf_col]
        mses = results_df[results_df['task'] == task][mse_col]
        evf_stats = {
            'hidden_size': hidden_size,
            'task': task,
            'min_evf': evfs.min(),
            'max_evf': evfs.max(),
            'mean_evf': evfs.mean(),
            'std_evf': evfs.std(),
            'min_mse': mses.min(),
            'max_mse': mses.max(),
            'mean_mse': mses.mean(),
            'std_mse': mses.std()
        }
        all_stats_by_hidden_size.append(evf_stats)

# Save the aggregated stats
all_stats_df = pd.DataFrame(all_stats_by_hidden_size)
stats_summary_path = op.join(computed_data_emile_path, model, model_config, data,
                             f"{today}_{model}_{model_config}_{data}_{content}_various_hidden_sizes_all_stats.csv")
#all_stats_df.to_csv(stats_summary_path, index=False)
print(f"\nSaved all hidden size stats to: {stats_summary_path}")

# Plot mean MSE and mean EVF for each hidden size
fig, ax1 = plt.subplots(figsize=(8, 5))

# Prepare data
mean_mse = all_stats_df.groupby('hidden_size')['mean_mse'].mean()
mean_evf = all_stats_df.groupby('hidden_size')['mean_evf'].mean()
hidden_sizes = mean_mse.index

color = 'tab:blue'
ax1.set_xlabel('Hidden Size')
ax1.set_xscale('log', base=2)
ax1.set_ylabel('Mean MSE', color=color)
ax1.plot(hidden_sizes, mean_mse, marker='o', color=color, label='Mean MSE')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Mean EVF', color=color)
ax2.plot(hidden_sizes, mean_evf, marker='s', color=color, label='Mean EVF')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Mean MSE and Mean EVF vs Hidden Size')
fig.tight_layout()
ax1.set_ylim(0, 1)
ax2.set_ylim(-1, 1)
plt.show()

##########################################################################################
#%% Section 6 — Plot Subject-wise GRU 
model = 'GRU'
model_config = 'subject'
hidden_sizes = {
    'group': 512,
    'subject': 512
}
data = 'experiment'
path = data_outcome_level_preprocessed_path
task = 'ada-prob'
content = 'mse_evf_scores'

print("model:", model)
print("model_config:", model_config)
print("data:", data)
print("data path:", path)
print("task:", task)
print("content:", content)

all_flat_results = {}
all_sequences = format_sequences(path, task)
subject_ids = sorted(set(seq[3] for seq in all_sequences))

# Choose one subject and one sequence randomly
random_subject = random.choice(subject_ids)

# Split into train and test
train_seq_gr, test_seq_gr = train_test_split(all_sequences, test_size=1-train_size_ratio, random_state=random_state)

train_seq_sb = [seq for seq in train_seq_gr if seq[3] == random_subject]
test_seq_sb = [seq for seq in test_seq_gr if seq[3] == random_subject]

train_seq_gr = random.sample(train_seq_gr, len(train_seq_sb))

# Choose randomly one sequence from test_seq_sb
if len(test_seq_sb) > 0:
    tested_sequence = random.choice(test_seq_sb)
else:
    raise RuntimeError("No test sequences found for the selected subject.")

print(f"Randomly selected subject: {random_subject}")
print(f"Train sequences group-level: {len(train_seq_gr)}, Test sequences group-level: {len(test_seq_gr)}")
print(f"Train sequences subject-level: {len(train_seq_sb)}, Test sequences subject-level: {len(test_seq_sb)}")

GRU_group_level_prediction = train_and_evaluate_gru(train_seq_gr, test_seq_gr, hidden_size=hidden_sizes['group'], return_pred=True)['sequence_pairs']
GRU_subject_level_prediction = train_and_evaluate_gru(train_seq_sb, test_seq_sb, hidden_size=hidden_sizes['subject'], return_pred=True)['sequence_pairs']

# Flatten the train and test sequences for HMM prediction
train_seq_gr_HMM = flatten_sequences(train_seq_gr)
test_seq_gr_HMM = flatten_sequences(test_seq_gr, tested_sequence=tested_sequence)
train_seq_sb_HMM = flatten_sequences(train_seq_sb)
test_seq_sb_HMM = flatten_sequences(test_seq_sb, tested_sequence=tested_sequence)

# To match the number of sequences, randomly select the same number of group-level sequences as subject-level
n_seq_sb = len(train_seq_sb_HMM) // length  # Number of subject-level sequences
all_seq_ids = train_seq_gr_HMM['sequence_id_2'].unique()
selected_seq_ids = np.random.choice(all_seq_ids, size=n_seq_sb, replace=False)
train_seq_gr_HMM_matched = train_seq_gr_HMM[train_seq_gr_HMM['sequence_id_2'].isin(selected_seq_ids)]

train_seq_gr_HMM = train_seq_gr_HMM_matched

HMM_group_level_prediction = predict_sequences_with_HMM(train_seq_gr_HMM, test_seq_gr_HMM, task, int(1/(1-train_size_ratio)))
HMM_subject_level_prediction = predict_sequences_with_HMM(train_seq_sb_HMM, test_seq_sb_HMM, task, int(1/(1-train_size_ratio)))

# Only select the tested sequence
hmm_sb_predicted_seq = HMM_subject_level_prediction[:length - 1]
hmm_gr_predicted_seq = HMM_group_level_prediction[:length - 1]

# Compute mean subject behavior
# Filter subject_ids to find those who did the task with the tested_sequence
filtered_subject_ids = [
    seq[3] for seq in all_sequences if np.array_equal(seq[1], tested_sequence[1])
]

print(f"Number of filtered subject IDs: {len(filtered_subject_ids)}")

# Extract subject behavior for these filtered subjects
filtered_subject_behaviors = [
    seq[2] for seq in all_sequences if seq[3] in filtered_subject_ids and np.array_equal(seq[1], tested_sequence[1])
]

# Compute the mean subject behavior
mean_subject_behavior = np.mean(filtered_subject_behaviors, axis=0)

print(mean_subject_behavior)

for i in range(len(GRU_subject_level_prediction)):
    # Compare values by converting both to numpy arrays
    left = GRU_subject_level_prediction[i][1]
    right = tested_sequence[1]
    if np.allclose(np.asarray(left), np.asarray(right)):
        hidden_params, outcome_seq, subject_seq, gru_sb_predicted_seq = GRU_subject_level_prediction[i]


for i in range(len(GRU_group_level_prediction)):
    # Compare values by converting both to numpy arrays
    left = GRU_group_level_prediction[i][1]
    right = tested_sequence[1]
    if np.allclose(np.asarray(left), np.asarray(right)):
        _, __, ___, gru_gr_predicted_seq = GRU_group_level_prediction[i]

plot_subject_sequence(
    outcome_seq, 
    hidden_params,
    len(train_seq_sb),
    len(train_seq_gr), 
    random_subject,
    hidden_sizes,
    subject_seq=subject_seq,
    mean_subject_behavior=None, #mean_subject_behavior,
    gru_sb_predicted_seq=gru_sb_predicted_seq,
    gru_gr_predicted_seq=gru_gr_predicted_seq,
    hmm_sb_predicted_seq=hmm_sb_predicted_seq,
    hmm_gr_predicted_seq=hmm_gr_predicted_seq
)

##########################################################################################
#%% 7 - Plot simulation results for group-wise GRU for various hidden sizes

date = '20250523'
model = 'GRU'
model_config = 'group_without_subject_embedding'
data = 'simulation'
task = 'ada-pos'
content = 'mse_and_evf_scores_various_hidden_layer_sizes_all_stats'
all_stats_path = op.join(computed_data_emile_path, model, model_config, data, \
                             f"{date}_{model}_{model_config}_{data}_{content}.csv")

all_stats_df = pd.read_csv(all_stats_path)
all_stats_df = all_stats_df[all_stats_df['task'] == task]

print(all_stats_df.columns)

all_stats_df = all_stats_df.rename(columns={
    'Hidden Size': 'hidden_size',
    'Test Loss': 'mean_mse',
    'Explained Variance Fraction': 'mean_evf'
})

# Plot mean MSE and mean EVF for each hidden size
fig, ax1 = plt.subplots(figsize=(8, 5))

# Prepare data
mean_mse = all_stats_df.groupby('hidden_size')['mean_mse'].mean()
mean_evf = all_stats_df.groupby('hidden_size')['mean_evf'].mean()
hidden_sizes = mean_mse.index

color = 'tab:blue'
ax1.set_xlabel('Hidden Size')
ax1.set_xscale('log', base=2)
ax1.set_ylabel('Mean MSE', color=color)
ax1.plot(hidden_sizes, mean_mse, marker='o', color=color, label='Mean MSE')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Mean EVF', color=color)
ax2.plot(hidden_sizes, mean_evf, marker='s', color=color, label='Mean EVF')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Mean MSE and Mean EVF vs Hidden Size')
fig.tight_layout()
ax1.set_ylim(0, 1)
ax2.set_ylim(-1, 1)
plt.show()
# %%
