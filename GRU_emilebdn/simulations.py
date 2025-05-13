# created on 2025/03/22

#%%
import sys

import numpy as np
import os.path as op
import pandas as pd

from datetime import datetime

sys.path.append(op.dirname(op.dirname(op.abspath(__file__))))

from functions.simulation_functions import assign_sequences_to_sims, fit_HMM_for_every_subject, simulate_subjects_behavior_with_HMM
from config.paths import data_path
from config.variables import task_types, do_inference_on_current_trial, p1_max, p1_min, resol

#%%

options = {
    'resol': resol,
    #'p_c': to be adapted for each subject,
    'p1_min': p1_min,
    'p1_max': p1_max,
    'do_inference_on_current_trial': do_inference_on_current_trial
}

# Dynamically generate today's date
today_date = datetime.now().strftime('%Y%m%d')

simulated_behaviors = pd.DataFrame()

for task in task_types:
    # p_c_fitted = fit_HMM_for_every_subject(task)

    # # Save the fitted p_c values in pickle format
    # output_file = op.join(data_path, 'computed_data_emile', f'{today_date}_{task}_fitted_p_c_value_for_each_subject.pkl')
    # pd.to_pickle(p_c_fitted, output_file)
    # print(f'Fitted p_c values saved to {output_file}')

    # Load the fitted p_c values from the specified path
    load_date = '20250513'
    input_file = op.join(data_path, 'computed_data_emile', f'{load_date}_{task}_fitted_p_c_value_for_each_subject.pkl')
    p_c_fitted = pd.read_pickle(input_file)
    print(f'Loaded p_c values from {input_file}')

    # Extract subject IDs from the keys of the fitted dictionary
    subject_ids = list(p_c_fitted.keys())

    # Pass subject_ids to assign_sequences_to_sims
    indexed_sequences, selected_sequences = assign_sequences_to_sims(task, subject_ids)

    results_df = simulate_subjects_behavior_with_HMM(task, indexed_sequences, selected_sequences, p_c_fitted, options)

    simulated_behaviors = pd.concat([simulated_behaviors, results_df], axis=0, ignore_index=True)

file_path = op.join(data_path, 'computed_data_emile', f'{today_date}_data_outcome_level_simulated.csv')
simulated_behaviors = simulated_behaviors.sort_values(by='subject')
simulated_behaviors.to_csv(file_path, index=False)

print(f'Saved CSV to: {file_path}')
# %%
