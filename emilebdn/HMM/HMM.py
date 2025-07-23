"""
This script computes Explained Variance fractions (EVF) for HMM fits on real data.

Author: @emilebdn  
Created date: 2025-05-20
"""
#%%
import random
import sys

import numpy as np
import os.path as op
import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from sklearn.metrics import explained_variance_score

# Add the root of the repository to sys.path
sys.path.append(op.dirname(op.dirname(op.dirname(__file__))))

from emilebdn.config.paths import (
    data_outcome_level_preprocessed_path,
    computed_data_emile_path
)
from emilebdn.config.variables import (
    n_jobs,
    task_types,
    b_0,
    tau_0,
    expID,
    model, 
    std_dev_pos
)
from emilebdn.HMM.HMM_functions import (
    compute_mse_evf_for_all_subjects,
    HMM_prediction,
    predict_sequences_with_HMM
)

today = datetime.now().strftime('%Y%m%d')

#%%
model = 'HMM'
model_config = ''
data = 'experiment'
content = 'mse_evf_scores'
task_types = ['ada-prob']

# Load data
data_outcome_level = pd.read_csv(data_outcome_level_preprocessed_path)

for task in task_types:
    print(f"Computing MSE and EVF for task: {task}")
    mse_scores, evf_scores = compute_mse_evf_for_all_subjects(data_outcome_level, task)

    mean_evf = np.mean(list(evf_scores.values()))
    mean_mse = np.mean(list(mse_scores.values()))
    print(f"Mean MSE across subjects ({task}):", mean_mse)
    print(f"Mean EVF across subjects ({task}):", mean_evf)

    # Save to CSV
    results = []
    for subject in mse_scores.keys():
        results.append({
            'subject': subject,
            'mse score': mse_scores[subject],
            'evf score': evf_scores.get(subject, np.nan)
        })
    results_df = pd.DataFrame(results, columns=['subject', 'mse score', 'evf score'])

    scores_path = op.join(computed_data_emile_path, model, data,
                          f'{today}_{model}{model_config}_{data}_{task}_{content}.csv')
    #results_df.to_csv(scores_path, index=False)
    print(f"Saved MSE and EVF scores to {scores_path}")

#%%
date = '20250604'
model = 'HMM'
model_config = ''
data = 'experiment'
task = 'ada-prob'
content = 'mse_evf_scores_stats'

scores = pd.read_csv(op.join(computed_data_emile_path, model, data, \
                             f'{date}_{model}_{data}_{task}_{content}'.replace('_stats', '.csv')))

evf_stats = scores['evf'].agg(['min', 'max', 'mean', 'std'])
mse_stats = scores['mse'].agg(['min', 'max', 'mean', 'std'])

stats_df = pd.DataFrame({
    'metric': ['evf', 'mse'],
    'min': [evf_stats['min'], mse_stats['min']],
    'max': [evf_stats['max'], mse_stats['max']],
    'mean': [evf_stats['mean'], mse_stats['mean']],
    'std': [evf_stats['std'], mse_stats['std']]
})

stats_path = op.join(computed_data_emile_path, model, data,
                     f'{date}_{model}_{data}_{task}_{content}.csv')
#stats_df.to_csv(stats_path, index=False)
print(f"Saved stats to {stats_path}")

#%%
data = pd.read_csv(data_outcome_level_preprocessed_path.replace('.csv', '_ada-prob_with_predictions.csv'))
subjects = data['subject'].unique()
task = 'ada-prob'

n_sb = 94
n_seq = 15
n_tri = 75

subject_hmm_array = np.full(n_sb*n_seq*n_tri, np.nan)

def process_subject(subject):
    subject_data = data[data['subject'] == subject]
    train_data, test_data = train_test_split(subject_data, test_size=0.2, random_state=42)
    hmm_model = predict_sequences_with_HMM(train_data, test_data, task)
    print(f"Predicted sequences for subject {subject} with task {task}.")
    print(f"Type of hmm_model: {type(hmm_model)}")
    if hasattr(hmm_model, 'shape'):
        print(f"Shape of hmm_model: {hmm_model.shape}")
    return subject, hmm_model

results = Parallel(n_jobs=50)(delayed(process_subject)(subject) for subject in subjects)

#%%
task = 'ada-prob'
# Import 'HMM_pc' column from results.optimisation.ada_opt_results.csv
base_dir = op.dirname(op.dirname(op.dirname(__file__)))
ada_opt_results_path = op.join(base_dir, 'results', 'optimisation', 'ada_opt_results.csv')
ada_opt_results = pd.read_csv(ada_opt_results_path)
hmm_pc_column = ada_opt_results['HMM_pc']
print("Imported 'HMM_pc' column:")
print(hmm_pc_column.head())

# Run HMM_prediction() for each subject using hmm_pc_column
data = pd.read_csv(data_outcome_level_preprocessed_path)
data = data[data['task'] == task]
subjects = data['subject'].unique()
hmm_predictions = {}

def run_hmm_prediction(subject, hmm_pc):
    print(f'HMM prediction for subject {subject}')
    subject_data = data[data['subject'] == subject]
    prediction = HMM_prediction(hmm_pc, subject_data, task, expID, b_0, tau_0, std_dev_pos)
    return subject, prediction

results = Parallel(n_jobs=n_jobs)(
    delayed(run_hmm_prediction)(subject, hmm_pc) for subject, hmm_pc in zip(subjects, hmm_pc_column)
)

hmm_predictions = dict(results)

print("HMM predictions for each subject using hmm_pc_column:")

# Compute EVS (Explained Variance Score) between 'estimate' and HMM prediction for each subject

evs_scores = {}
for subject, prediction in list(hmm_predictions.items())[:5]:
    subject_data = data[data['subject'] == subject]
    y_true = subject_data['estimate'].values
    y_pred = np.array(prediction)
    # Ensure y_true and y_pred have the same length
    min_len = min(len(y_true), len(y_pred))
    evs = explained_variance_score(y_true[:min_len], y_pred[:min_len])
    evs_scores[subject] = evs
    print(f"Subject: {subject}, EVS: {evs}, Prediction: {prediction}")
for subject, prediction in list(hmm_predictions.items())[:5]:
    print(f"Subject: {subject}, Prediction: {prediction}")

mean_evs = np.mean(list(evs_scores.values()))
print(f"Mean EVS across subjects: {mean_evs}")
mean_mse = np.mean([
    np.mean((data[data['subject'] == subject]['estimate'].values[:min(len(data[data['subject'] == subject]['estimate'].values), len(np.array(prediction)))] - np.array(prediction)[:min(len(data[data['subject'] == subject]['estimate'].values), len(np.array(prediction)))]) ** 2)
    for subject, prediction in hmm_predictions.items()
])
print(f"Mean EVS across subjects: {mean_evs}")
print(f"Mean MSE across subjects: {mean_mse}")

#%%
