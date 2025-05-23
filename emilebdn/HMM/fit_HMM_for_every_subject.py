"""
This script fits a Hidden Markov Model (HMM) for every subject and task type using preprocessed outcome-level data.
It loads the data, fits the HMM, and saves the fitted parameters in both pickle and CSV formats.

Author: @emilebdn  
Created date: 2025-05-22
"""
#%%
import sys

import os.path as op
import pandas as pd

from datetime import datetime

# Add the root of the repository to sys.path
sys.path.append(op.dirname(op.dirname(op.dirname(op.abspath(__file__)))))

from emilebdn.config.paths import (
    computed_data_emile_path,
    data_outcome_level_preprocessed_path
)
from emilebdn.config.variables import task_types
from emilebdn.HMM.HMM_functions import fit_HMM_for_every_subject

today = datetime.now().strftime('%Y%m%d')

# Fit HMM for every subject
model = 'HMM'
model_config = ''
data = 'experiment'
content = 'fitted_p_c_for_every_subject'

#%%
for task in task_types:
    data_outcome_level_preprocessed = pd.read_csv(op.join(data_outcome_level_preprocessed_path))
    p_c_fitted = fit_HMM_for_every_subject(data_outcome_level_preprocessed, task)

    # Save the fitted p_c values in pickle format
    p_c_fitted_path = op.join(computed_data_emile_path, model, data, content, \
                          f'{today}_{model}{model_config}_{data}_{task}_{content}.pkl')
    pd.to_pickle(p_c_fitted, p_c_fitted_path)
    print(f'Fitted p_c values saved to {p_c_fitted_path}')
    
    # Save the fitted p_c values in CSV format
    p_c_fitted_path_csv = p_c_fitted_path.replace('.pkl', '.csv')
    p_c_fitted_df = pd.DataFrame(list(p_c_fitted.items()), columns=['subject', 'p_c'])
    p_c_fitted_df.to_csv(p_c_fitted_path_csv, index=False)
    print(f'Fitted p_c values for {task} saved to {p_c_fitted_path_csv}')

# %%
stats = []

for task in task_types:
    p_c_fitted_path = op.join(computed_data_emile_path, model, data, content, f'20250522_{model}{model_config}_{data}_{task}_{content}.pkl')
    p_c_fitted = pd.read_pickle(p_c_fitted_path)
    p_c_values = list(p_c_fitted.values())
    stats.append({
        'task': task,
        'min': min(p_c_values),
        'max': max(p_c_values),
        'mean': sum(p_c_values) / len(p_c_values),
        'std': pd.Series(p_c_values).std()
    })

stats_df = pd.DataFrame(stats)
print(stats_df)

# After computing stats_df
p_c_fitted_stats_path = op.join(computed_data_emile_path, model, data, content, \
                                f'20250522_{model}{model_config}_{data}_{content}_stats.csv')
stats_df.to_csv(p_c_fitted_stats_path, index=False)
print(f'Stats saved to {p_c_fitted_stats_path}')