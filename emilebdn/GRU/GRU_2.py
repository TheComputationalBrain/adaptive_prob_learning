"""
GRU Sequence Prediction Pipeline
--------------------------------
This script runs a GRU-based recurrent neural network pipeline for human learning studies.
It includes hyperparameter tuning, subject-wise and group-level evaluation on simulated and real data.

Author: @emilebdn
Created date: 2025-06-06
"""
# %%
import datetime
import random
import sys

import matplotlib.pyplot as plt
import os.path as op
import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

# Add the root of the repository to sys.path
sys.path.append(op.dirname(op.dirname(op.dirname(op.abspath(__file__)))))

from emilebdn.config.paths import (
    computed_data_emile_path,
    data_outcome_level_preprocessed_path
)
from emilebdn.config.variables import (
    n_jobs,
    task_types,
    train_size_ratio,
    input_size
)
from emilebdn.GRU.GRU_functions_2 import (
    import_sequences,
    train_and_evaluate_gru,
    train_and_evaluate_subject_gru
)

today = datetime.datetime.now().strftime("%Y%m%d")

##########################################################################################
# %% Section 4 — Train Subject-wise GRU 
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

sequences = import_sequences(data, path, task)

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
results_df.to_csv(results_path, index=False)
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
stats.to_csv(evf_stats_path, index=False)
print(f"Saved explained variance and mse stats to: {evf_stats_path}")

##########################################################################################
# %% Section 5 — Train Subject-wise GRU with various hidden layer sizes 
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
        sequences = import_sequences(data, path, task)
        subject_ids = sorted(set(seq[3] for seq in sequences))
        subjects_sequences = {subject_id: [seq for seq in sequences if seq[3] == subject_id] for subject_id in subject_ids}
        
        results = Parallel(n_jobs=n_jobs)(delayed(train_and_evaluate_subject_gru)(
            subjects_sequences[subject_id], subject_id, data, task, train_size_ratio,
            input_size, hidden_size
        ) for subject_id in subject_ids)
        
        for subject_id, result in results:
            all_flat_results[(task, subject_id)] = result

    # Create DataFrame
    results_df = pd.DataFrame.from_dict(all_flat_results, orient='index').reset_index(drop=True)

    # Save full results (optional per hidden size)
    results_path = op.join(computed_data_emile_path, model, model_config, data,
        f"{today}_{model}_{model_config}_{data}_{content}_hs_{hidden_size}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved subject GRU results (hidden size {hidden_size}) to: {results_path}")

    # Compute stats
    evf_col = 'explained_variance_fraction'
    mse_col = 'test_loss'
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
all_stats_df.to_csv(stats_summary_path, index=False)
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
# %% Section 6 — Plot Subject-wise GRU 
model = 'GRU'
model_config = 'subject'
hidden_sizes = {
    'group': 1024,
    'subject': 1024
}
data = 'experiment'
data_path = data_outcome_level_preprocessed_path
task = 'ada-prob'
content = 'mse_evf_scores'

print("model:", model)
print("model_config:", model_config)
print("data:", data)
print("data_path:", data_path)
print("task:", task)
print("content:", content)

path = data_outcome_level_preprocessed_path

all_flat_results = {}

all_sequences = import_sequences(data, path, task)

# Get all subject IDs
subject_ids = sorted(set(seq[3] for seq in all_sequences))

# Choose one subject and one sequence randomly
random_subject = random.choice(subject_ids)

# Split into train and test (80% train, 20% test)
train_seq_gr, test_seq_gr = train_test_split(all_sequences, test_size=0.2, random_state=42)

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

group_level_prediction = train_and_evaluate_gru(train_seq_gr, test_seq_gr, hidden_size=hidden_sizes['group'], return_pred=True)['sequence_pairs']
subject_level_prediction = train_and_evaluate_gru(train_seq_sb, test_seq_sb, hidden_size=hidden_sizes['subject'], return_pred=True)['sequence_pairs']

for i in range(len(subject_level_prediction)):
    # Compare values by converting both to numpy arrays
    left = subject_level_prediction[i][1]
    right = tested_sequence[1]
    if np.allclose(np.asarray(left), np.asarray(right)):
        hidden_parms, outcome_seq, subject_seq, subject_model_seq = subject_level_prediction[i]

for i in range(len(group_level_prediction)):
    # Compare values by converting both to numpy arrays
    left = group_level_prediction[i][1]
    right = tested_sequence[1]
    if np.allclose(np.asarray(left), np.asarray(right)):
        _, __, ___, group_model_seq = group_level_prediction[i]
    
#%%
plt.figure(figsize=(14, 7))  # Augmenter la hauteur ici (de 4 à 7)
ax1 = plt.subplot(1, 1, 1)
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
# Plot subject estimate as light blue line
ax1.plot(subject_seq, label='Subject Behavior', color='deepskyblue', alpha=0.8, linewidth=2)
# Plot subject-level model estimate as red line
ax1.plot(subject_model_seq, label=f"Subject-level GRU (nb units: {hidden_sizes['subject']})", color='red', alpha=0.8, linewidth=2)
# Plot group-level model estimate as pink line
ax1.plot(group_model_seq, label=f"Group-level GRU (nb units: {hidden_sizes['group']})", color='pink', alpha=0.8, linewidth=2)
ax1.set_ylabel(r'$\mathbb{P}(\mathrm{blue})$')
ax1.set_ylim(0, 1)
ax1.set_xlabel('Time step')
ax1.set_title(f'Sequence 1 (Subject {random_subject})')
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
ax2.set_ylim(0, 1)
ax2.tick_params(axis='y', labelcolor='black')  # couleur noire pour l'axe de droite
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# Légende légèrement plus haute et un peu plus à gauche
# Combine all legend entries and add training sequence info as a separate legend below

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
        f"Train seq (subject): {len(train_seq_sb)}",
        f"Train seq (group): {len(train_seq_gr)}"
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
# %%
