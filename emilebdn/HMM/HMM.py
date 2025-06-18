"""
This script computes Explained Variance fractions (EVF) for HMM fits on real data.

Author: @emilebdn  
Created date: 2025-05-20
"""
#%%
import sys
import numpy as np
import os.path as op
import pandas as pd

from datetime import datetime

# Add the root of the repository to sys.path
sys.path.append(op.dirname(op.dirname(op.dirname(op.abspath(__file__)))))

from emilebdn.config.paths import (
    data_outcome_level_preprocessed_path,
    computed_data_emile_path
)
from emilebdn.config.variables import (
    model,
    task_types,
)
from emilebdn.HMM.HMM_functions import compute_mse_evf_for_all_subjects

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
    results_df.to_csv(scores_path, index=False)
    print(f"Saved MSE and EVF scores to {scores_path}")

# %%
date = '20250604'
model = 'HMM'
model_config = ''
data = 'experiment'
task = 'ada-prob'
content = 'mse_evf_scores_stats'

scores = pd.read_csv(op.join(computed_data_emile_path, model, data, \
                             f'{date}_{model}_{data}_{task}_{content}'.replace('_stats', '.csv')))

evf_stats = scores['evf score'].agg(['min', 'max', 'mean', 'std'])
mse_stats = scores['mse score'].agg(['min', 'max', 'mean', 'std'])

stats_df = pd.DataFrame({
    'metric': ['evf', 'mse'],
    'min': [evf_stats['min'], mse_stats['min']],
    'max': [evf_stats['max'], mse_stats['max']],
    'mean': [evf_stats['mean'], mse_stats['mean']],
    'std': [evf_stats['std'], mse_stats['std']]
})

stats_path = op.join(computed_data_emile_path, model, data,
                     f'{date}_{model}_{data}_{task}_{content}.csv')
stats_df.to_csv(stats_path, index=False)
print(f"Saved stats to {stats_path}")

# %%
