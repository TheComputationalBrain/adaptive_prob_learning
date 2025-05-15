"""
This script provides functions for generating sequences, fitting HMM models to real subjects' data, 
and simulating subjects' behavior in adaptive learning tasks. It includes utilities for both 
magnitude learning and probability learning tasks.

Author: @emilebdn  
Created date: 2025-04-30
"""

#%%
import sys

import numpy as np
import os.path as op
import pandas as pd

from joblib import Parallel, delayed

sys.path.append(op.dirname(op.dirname(op.abspath(__file__))))

import models.IdealObserver as IO
import model_learner_pos as MP

from data_analysis_utils import fit_model
from config.paths import data_outcome_level_preprocessed_path
from config.variables import n_jobs, b_0, tau_0, expID, model, length, n_sequences_pos, change_prob_pos, freeze_duration_pos, std_dev_pos, \
    n_sequences_for_each_subject_pos, n_sequences_prob, change_prob_prob, freeze_duration_prob, min_val_prob, max_val_prob, \
    odds_change_threshold_prob, n_sequences_for_each_subject_prob

# -------------------------------
# Generate magnitude learning tast sequences
# -------------------------------

def generate_magnitude_sequence(length, change_prob, freeze_duration, std_dev):
    sequence = {'outcome': [], 'did_change_point_occur': [], 'hidden_parameter': []}
    current_mean = np.random.uniform(0, 1)
    freeze_counter = freeze_duration

    for _ in range(length):
        change_occurred = False
        if freeze_counter <= 0 and np.random.rand() < change_prob:
            current_mean = np.random.uniform(0, 1)
            freeze_counter = freeze_duration
            change_occurred = True
        else:
            freeze_counter -= 1

        outcome = np.random.normal(loc=current_mean, scale=std_dev)
        sequence['outcome'].append(np.clip(outcome, 0, 1))  # Outcome, clipped to [0, 1]
        sequence['did_change_point_occur'].append(change_occurred)  # Change point occurrence
        sequence['hidden_parameter'].append(current_mean)  # Hidden parameter (mean)

    return sequence

def generate_magnitude_sequences(length=length, n_sequences=n_sequences_pos, change_prob=change_prob_pos, freeze_duration=freeze_duration_pos, std_dev=std_dev_pos):
    return [generate_magnitude_sequence(length, change_prob, freeze_duration, std_dev) for _ in range(n_sequences)]

# -------------------------------
# Generate probability learning task sequences
# -------------------------------

def sample_new_prob(p_old, min_val, max_val, odds_change_threshold):
    while True:
        p_new = np.random.uniform(min_val, max_val)
        if p_old is None:
            return p_new
        old_odds = p_old / (1 - p_old)
        new_odds = p_new / (1 - p_new)
        if (new_odds / old_odds >= odds_change_threshold) or (old_odds / new_odds >= odds_change_threshold):
            return p_new

def generate_probability_sequence(length, change_prob, freeze_duration,
                                  min_val, max_val, odds_change_threshold):
    sequence = {'outcome': [], 'did_change_point_occur': [], 'hidden_parameter': []}
    current_prob = sample_new_prob(None, min_val, max_val, odds_change_threshold)
    freeze_counter = freeze_duration
    
    for _ in range(length):
        change_occurred = False
        if freeze_counter <= 0 and np.random.rand() < change_prob:
            current_prob = sample_new_prob(current_prob, min_val, max_val, odds_change_threshold)
            freeze_counter = freeze_duration
            change_occurred = True
        else:
            freeze_counter -= 1
        
        outcome = np.random.binomial(1, current_prob)
        sequence['outcome'].append(outcome)  # Outcome
        sequence['did_change_point_occur'].append(change_occurred)  # Change point occurrence
        sequence['hidden_parameter'].append(current_prob)  # Hidden parameter (prob)

    return sequence

def generate_probability_sequences(length=length, n_sequences=n_sequences_prob, change_prob=change_prob_prob,
                                   freeze_duration=freeze_duration_prob, min_val=min_val_prob, max_val=max_val_prob, odds_change_threshold=odds_change_threshold_prob):
    return [generate_probability_sequence(length, change_prob, freeze_duration, min_val, max_val, \
                                          odds_change_threshold) for _ in range(n_sequences)]

# -------------------------------
# Fit HMM to real subjects' data
# -------------------------------

def fit_HMM_for_each_subject(subject, subj_idx, sessions, expID, model):
    # Fitting p_c (HMM model parameter) for one subject data
    return fit_model(expID, model, subj_idx, sessions[subject])[1]  # Extract the second value (fitted p_c)

data_outcome_level = pd.read_csv(data_outcome_level_preprocessed_path)

def fit_HMM_for_every_subject(task, data_outcome_level=data_outcome_level, n_jobs=n_jobs, expID=expID, model=model):
    # Fitting p_c (HMM model parameter) with real subjects' data
    subjects = data_outcome_level['subject'].unique()

    filtered_data = data_outcome_level[data_outcome_level['task'] == task] 

    sessions = {
    subject: filtered_data.loc[filtered_data['subject'] == subject, 'session_idx'].unique()
    for subject in subjects
    }

    p_c_fitted = Parallel(n_jobs=n_jobs)(
        delayed(fit_HMM_for_each_subject)(subject, subj_idx, sessions, expID, model)
        for subj_idx, subject in enumerate(subjects)
    )
    return dict(zip(subjects, p_c_fitted))

# -------------------------------
# Simulate subjects' behavior
# -------------------------------

def assign_sequences_to_sims(task, subject_ids):
    if task == 'ada-pos':
        sequences = generate_magnitude_sequences()
        n_sequences_for_each_subject = n_sequences_for_each_subject_pos
        indexed_sequences = {seq_id: seq for seq_id, seq in enumerate(sequences[:n_sequences_pos], start=1)}
    
    elif task == 'ada-prob':
        sequences = generate_probability_sequences()
        n_sequences_for_each_subject = n_sequences_for_each_subject_prob
        indexed_sequences = {seq_id: seq for seq_id, seq in enumerate(sequences[:n_sequences_prob], start=1)}
    
    else:
        raise ValueError("Task must be 'ada-pos' or 'ada-prob'.")

    # Assign real subject IDs (from fit_HMM)
    selected_sequences = {
        subject: np.random.choice(list(indexed_sequences.keys()), n_sequences_for_each_subject, replace=False)
        for subject in subject_ids
    }

    return indexed_sequences, selected_sequences

def simulate_subjects_behavior_with_HMM(task, indexed_sequences, selected_sequences, p_c_fitted, options):
    # Prepare containers
    sequence_ids = []
    subjects_col = []
    positions = []
    simulated_outcomes = []
    fitted_estimates = []
    hidden_parameters = []
    change_point_flags = []
    tasks = []
    
    # Run inference and collect data
    for subject_key, seq_ids in selected_sequences.items():
        options['p_c'] = p_c_fitted[subject_key]  # Update p_c for each subject
        for sequence_id in seq_ids:
            prob_seq = indexed_sequences[sequence_id]
            
            # Run inference
            if task == 'ada-pos':
                prob_seq['outcome'] = np.array(prob_seq['outcome'])
                inference_out = MP.run_inference(prob_seq['outcome'], p_c=options['p_c'], std_gen=std_dev_pos, b_0=b_0, tau_0=tau_0)
                mod_est = inference_out['mean']

            if task == 'ada-prob':
                inference_out = IO.run_inference(prob_seq['outcome'], options=options)
                mod_est = inference_out[1,]['mean']


            for position, (obs_val, fit_val, hidden_val, change_flag) in enumerate(
                zip(prob_seq['outcome'], mod_est, prob_seq['hidden_parameter'], prob_seq['did_change_point_occur'])
            ):
                sequence_ids.append(sequence_id)
                subjects_col.append(subject_key)
                positions.append(position)
                simulated_outcomes.append(obs_val)
                fitted_estimates.append(fit_val)
                hidden_parameters.append(hidden_val)
                change_point_flags.append(change_flag)
                tasks.append(task)

    # Create DataFrame
    probability_and_fitted_sequences_flattened = pd.DataFrame({
        'subject': subjects_col,
        'task': tasks,
        'outcome_idx': positions,
        'outcome': simulated_outcomes,
        'estimate': fitted_estimates,
        'hidden_parameter': hidden_parameters,
        'did_change_point_occur': change_point_flags,
        'sequence_id': sequence_ids,
        'p_c_fitted': [p_c_fitted[subject][0] for subject in subjects_col]
    })

    return probability_and_fitted_sequences_flattened
