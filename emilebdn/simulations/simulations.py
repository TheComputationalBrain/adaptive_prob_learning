"""
This script simulates subjects' behavior using a Hidden Markov Model (HMM) for sequence prediction tasks. 
It includes loading pre-fitted HMM parameters, assigning sequences, simulating behavior, and saving the results 
to a CSV file.

Author: @emilebdn  
Created date: 2025-03-22
"""

#%%
import sys

import os.path as op
import pandas as pd

from datetime import datetime

# Add the root of the repository to sys.path
sys.path.append(op.dirname(op.dirname(op.dirname(__file__))))

from emilebdn.config.paths import computed_data_emile_path
from emilebdn.config.variables import (
    task_types,
    p_c_bounds,
    do_inference_on_current_trial, resol,
    n_sequences_for_each_subject
)
from emilebdn.simulations.simulation_functions import (
    assign_sequences_to_sims, 
    simulate_subjects_behavior_with_HMM
)

today = datetime.now().strftime('%Y%m%d')
#%%
options = {
    'resol': resol,
    #'p_c': to be adapted for each subject,
    'p1_min': p_c_bounds['ada-prob']['min'],
    'p1_max': p_c_bounds['ada-prob']['max'],
    'do_inference_on_current_trial': do_inference_on_current_trial
}

#%%
date = '20250522'
model = 'HMM'
model_config = ''
data = 'experiment'
content = 'fitted_p_c_for_every_subject'

simulated_behaviors = pd.DataFrame()

for task in task_types:
    # Load the fitted p_c values
    p_c_fitted_path = op.join(computed_data_emile_path, model, data, content, \
                         f'{date}_{model}{model_config}_{data}_{task}_{content}.pkl')
    p_c_fitted = pd.read_pickle(p_c_fitted_path)
    print(f'Loaded p_c values from {p_c_fitted_path}')

    # Extract subject IDs from the keys of the loaded dictionary
    subject_ids = list(p_c_fitted.keys())

    # Pass subject_ids to assign_sequences_to_sims
    indexed_sequences, selected_sequences = assign_sequences_to_sims(task, subject_ids, n_sequences_for_each_subject)

    results_df = simulate_subjects_behavior_with_HMM(task, indexed_sequences, selected_sequences, p_c_fitted, options)

    simulated_behaviors = pd.concat([simulated_behaviors, results_df], axis=0, ignore_index=True)

simulated_behaviors_path = op.join(computed_data_emile_path, 'data_outcome_level', \
                    f"{today}_data_outcome_level_simulated.csv")

simulated_behaviors = simulated_behaviors.sort_values(
    by=['subject', 'task', 'session_idx', 'outcome_idx'],
    key=lambda col: (
        col.map({subject: i for i, subject in enumerate(subject_ids)}) if col.name == 'subject'
        else col.map({'ada-pos': 0, 'ada-prob': 1}) if col.name == 'task'
        else col
    )
)
simulated_behaviors.to_csv(simulated_behaviors_path, index=False)

print(f'Saved CSV to: {simulated_behaviors_path}')