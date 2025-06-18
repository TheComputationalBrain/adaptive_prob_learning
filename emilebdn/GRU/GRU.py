"""
GRU Sequence Prediction Pipeline
--------------------------------
This script runs a GRU-based recurrent neural network pipeline for human learning studies.
It includes hyperparameter tuning, subject-wise and group-level evaluation on simulated and real data.

Author: @emilebdn
Created date: 2025-04-15
"""
# %%
import datetime
import sys

import os.path as op
import pandas as pd

from joblib import Parallel, delayed

# Add the root of the repository to sys.path
sys.path.append(op.dirname(op.dirname(op.dirname(op.abspath(__file__)))))

from emilebdn.config.paths import (
    computed_data_emile_path,
    data_outcome_level_preprocessed_path,
    data_outcome_level_simulated_path,
)
from emilebdn.config.variables import (
    n_jobs,
    task_types,
    train_size_ratio,
    best_GRU_hidden_layer_sizes,
    subject_embedding_dim
)
from emilebdn.GRU.GRU_functions import (
    find_best_GRU_hidden_layer_size,
    import_sequences,
    process_subject,
    split_sequences_for_cv,
    train_and_evaluate_gru,
)

today = datetime.datetime.now().strftime("%Y%m%d")

##########################################################################################
# %% Section 1 — Grid Search on Simulated Data (Group-level)
print("=== Section 1: Group-level Grid Search for hidden_layer_size ===")

model = 'GRU'
use_subject_embedding = False
model_config = 'group_with_subject_embedding' if use_subject_embedding else 'group_without_subject_embedding'
subject_embedding_dim_1 = subject_embedding_dim
data = 'simulation'
data_path = data_outcome_level_simulated_path
task_types = ['ada-pos', 'ada-prob']
content = 'fit'

print("model:", model)
print("use_subject_embedding:", use_subject_embedding)
print("model_config:", model_config)
if use_subject_embedding:
    print("subject_embedding_dim:", subject_embedding_dim_1)
print("data:", data)
print("data_path:", data_path)
print("task_types:", task_types)
print("content:", content)

all_results = []

for task in task_types:
    print(f"\nRunning task: {task}")
    sequences = import_sequences(data, data_path, task, use_subject_embedding)[0]
    results_df = find_best_GRU_hidden_layer_size(sequences, use_subject_embedding=use_subject_embedding)
    results_df['model'] = model
    results_df['model_config'] = model_config
    if use_subject_embedding:
        results_df['subject_embedding_dim'] = subject_embedding_dim
    results_df['data'] = data
    results_df['task'] = task
    all_results.append(results_df)

combined_df = pd.concat(all_results, ignore_index=True)

summary_path = op.join(computed_data_emile_path, model, model_config, data, \
                        f"{today}_{model}_{model_config}_{data}_{content}.csv")
combined_df.to_csv(summary_path, index=False)
print(f"\nSaved group-level results to: {summary_path}")

# Best per-subject hidden sizes
best_hidden_sizes = combined_df.loc[combined_df.groupby('task')['Explained Variance Fraction'].idxmax()]
best_hidden_sizes_path = summary_path.replace('.csv', '_best_hidden_sizes.csv')
best_hidden_sizes.to_csv(best_hidden_sizes_path, index=False)
print(f"Saved best hidden sizes to: {best_hidden_sizes_path}")

##########################################################################################
# %% Section 2 — Subject-wise Grid Search on Simulated Data
print("\n=== Section 2: Subject-wise Grid Search for hidden_layer_size ===")

data = 'simulation'
path = data_outcome_level_simulated_path
GRU_type = 'subjectwise'
use_subject_embedding = True
task_types = ['ada-pos', 'ada-prob']

print("data:", data)
print("path:", path)
print("GRU_type:", GRU_type)
print("use_subject_embedding:", use_subject_embedding)
print("task_types:", task_types)

all_results = []

for task in task_types:
    sequences = import_sequences(data, path, task, use_subject_embedding)[0]
    subject_ids = sorted(set(seq[2] for seq in sequences))

    task_results = Parallel(n_jobs=n_jobs)(delayed(process_subject)(sequences, subject_id, data, \
                                                                    task, use_subject_embedding) for subject_id in subject_ids)
    all_results.extend(task_results)

combined_df = pd.concat(all_results, ignore_index=True)
combined_df = combined_df[['data', 'use_subject_embedding', 'task', 'subject_id', 'Hidden Size', 'Test Loss', 'Explained Variance Fraction']]

subjectwise_path = op.join(computed_data_emile_path, f"{today}_{data}_GRU_hidden_size_subjectwise_results_with_subject_embedding.csv")
combined_df.to_csv(subjectwise_path, index=False)
print(f"\nSaved subject-wise results to: {subjectwise_path}")

# Best per-subject hidden sizes
best_hidden_sizes = combined_df.loc[combined_df.groupby(['task', 'subject_id'])['Explained Variance Fraction'].idxmax()]
best_hidden_sizes_path = subjectwise_path.replace('.csv', '_best_hidden_sizes.csv')
best_hidden_sizes.to_csv(best_hidden_sizes_path, index=False)

# Summary stats
summary_stats_df = pd.DataFrame({
    'Metric': ['Mean Explained Variance Fraction', 'Mean Test Loss'],
    'Value': [best_hidden_sizes['Explained Variance Fraction'].mean(), best_hidden_sizes['Test Loss'].mean()]
})
summary_stats_path = best_hidden_sizes_path.replace('.csv', '_summary_stats.csv')
summary_stats_df.to_csv(summary_stats_path, index=False)

print(f"Saved best hidden sizes and summary stats for subject-wise results.")

##########################################################################################
# %% Section 3 — Train Group GRU with or without subject embedding
print("\n=== Section 3: Group-level GRU Training ===")

model = 'GRU'
use_subject_embedding = False
model_config = 'group_with_subject_embedding' if use_subject_embedding else 'group_without_subject_embedding'
subject_embedding_dim_3 = subject_embedding_dim*4
data = 'experiment'
data_path = data_outcome_level_preprocessed_path
task_types = ['ada-pos', 'ada-prob']
content = 'fit'

print("model:", model)
print("use_subject_embedding:", use_subject_embedding)
print("model_config:", model_config)
print("subject_embedding_dim:", subject_embedding_dim_3)
print("data:", data)
print("data_path:", data_path)
print("task_types:", task_types)
print("content:", content)

for task in task_types:
    hidden_size = best_GRU_hidden_layer_sizes[task]
    sequences = import_sequences(data, data_path, task, use_subject_embedding)[0]
    n_splits = int(1 / (1 - train_size_ratio))
    cv_splits = split_sequences_for_cv(sequences, n_splits)
    cv_results = []

    for train_seq, test_seq in cv_splits:
        results = train_and_evaluate_gru(train_seq, test_seq, hidden_size=hidden_size, use_subject_embedding=use_subject_embedding)
        cv_results.append(results)

    # Compute mean of results across CV splits for each key,
    # except 'training_time' for which we take the sum
    mean_results = {}
    keys = cv_results[0].keys()
    for key in keys:
        if key == 'training_time':
            mean_results['total_training_time'] = sum(res[key] for res in cv_results)
        else:
            mean_results['mean_' + key] = sum(res[key] for res in cv_results) / len(cv_results)
    
    mean_results['model'] = model
    mean_results['model_config'] = model_config
    if use_subject_embedding:
        mean_results['subject_embedding_dim'] = subject_embedding_dim_3
    mean_results['data'] = data
    mean_results['task'] = task
    mean_results['hidden_size'] = hidden_size

    results_df = pd.DataFrame([mean_results])
    results_filename = f"{today}_{model}_{model_config}_{data}_{task}_{content}.csv"
    results_path = op.join(computed_data_emile_path, model, model_config, data, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"Saved group GRU fit results to: {results_path}")
