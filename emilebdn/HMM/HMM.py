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
from emilebdn.HMM.HMM_functions import compute_evf_for_all_subjects

today = datetime.now().strftime('%Y%m%d')

#%%
model = 'HMM'
model_config = ''
data = 'experiment'
content = 'evf_scores'

# Load data
data_outcome_level = pd.read_csv(data_outcome_level_preprocessed_path)

for task in task_types:
    print(f"Computing EVF for task: {task}")
    evf_scores = compute_evf_for_all_subjects(data_outcome_level, task)
    print(evf_scores)

    mean_evf = np.mean(list(evf_scores.values()))
    print(f"Mean EVF across subjects ({task}):", mean_evf)

    # Save to CSV
    evf_scores_path = op.join(computed_data_emile_path, model, data, \
                              f'{today}_{model}{model_config}_{data}_{task}_{content}.csv')
    evf_scores_df = pd.DataFrame(list(evf_scores.items()), columns=['subject', 'evf score'])
    evf_scores_df.to_csv(evf_scores_path, index=False)
    print(f"Saved EVF scores to {op.join(computed_data_emile_path, model, data, f'{today}_{model}{model_config}_{data}_{task}_{content}.csv')}")
# %%
