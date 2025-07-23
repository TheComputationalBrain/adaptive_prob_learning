"""
This script provides functions for generating sequences and simulating subjects' behavior in adaptive learning tasks. 
It includes utilities for both magnitude learning and probability learning tasks.

Author: @emilebdn  
Created date: 2025-04-30
"""

#%%
import sys

import numpy as np
import os.path as op
import pandas as pd

# Add the root of the repository to sys.path
sys.path.append(op.dirname(op.dirname(op.dirname(__file__))))

import emilebdn.simulations.model_learner_pos as MP
import models.IdealObserver as IO

from emilebdn.config.variables import (
    b_0, tau_0,
    length, n_sequences_for_each_subject,
    n_sequences_pos, change_prob_pos, freeze_duration_pos, std_dev_pos,
    n_sequences_prob, change_prob_prob, freeze_duration_prob, min_val_prob, max_val_prob, \
    odds_change_threshold_prob
)

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
# Simulate subjects' behavior
# -------------------------------

def assign_sequences_to_sims(task, subject_ids, n_sequences_for_each_subject=n_sequences_for_each_subject):
    if task == 'ada-pos':
        sequences = generate_magnitude_sequences()
        indexed_sequences = {seq_id: seq for seq_id, seq in enumerate(sequences, start=1)}
    
    elif task == 'ada-prob':
        sequences = generate_probability_sequences()
        indexed_sequences = {seq_id: seq for seq_id, seq in enumerate(sequences, start=1)}
    
    else:
        raise ValueError("Task must be 'ada-pos' or 'ada-prob'.")

    n_sequences_for_each_subject = n_sequences_for_each_subject[task]

    selected_sequences = {
        subject: {
            'sequence_ids': np.random.choice(list(indexed_sequences.keys()), n_sequences_for_each_subject, replace=False),
            'session_idxs': np.arange(n_sequences_for_each_subject)
        }
        for subject in subject_ids
    }

    return indexed_sequences, selected_sequences

def simulate_subjects_behavior_with_HMM(task, indexed_sequences, selected_sequences, p_c_fitted, options):
    # Prepare containers for collected data
    data = {
        'subject': [],
        'task': [],
        'session_idx': [],
        'outcome_idx': [],
        'outcome': [],
        'estimate': [],
        'hidden_parameter': [],
        'did_change_point_occur': [],
        'sequence_id': [],
        'p_c_fitted': []
    }

    # Iterate through each subject and their associated sequences
    for subject_id, seq_info in selected_sequences.items():
        sequence_ids = seq_info['sequence_ids']
        session_idxs = seq_info['session_idxs']

        options['p_c'] = p_c_fitted[subject_id]  # Update model parameters for current subject

        for sequence_id, session_idx in zip(sequence_ids, session_idxs):
            prob_seq = indexed_sequences[sequence_id]

            # Run the appropriate inference function based on the task
            if task == 'ada-pos':
                prob_seq['outcome'] = np.array(prob_seq['outcome'])
                inference_result = MP.run_inference(
                    prob_seq['outcome'], 
                    p_c=options['p_c'], 
                    std_gen=std_dev_pos, 
                    b_0=b_0, 
                    tau_0=tau_0
                )
                estimates = inference_result['mean']

            elif task == 'ada-prob':
                inference_result = IO.run_inference(prob_seq['outcome'], options=options)
                estimates = inference_result[1,]['mean']

            else:
                raise ValueError(f"Unsupported task type: {task}")

            # Record each timepoint within the sequence
            for position, (obs, est, hidden, change_flag) in enumerate(
                zip(prob_seq['outcome'], estimates, prob_seq['hidden_parameter'], prob_seq['did_change_point_occur'])
            ):
                data['subject'].append(subject_id)
                data['task'].append(task)
                data['session_idx'].append(session_idx)
                data['outcome_idx'].append(position)
                data['outcome'].append(obs)
                data['estimate'].append(est)
                data['hidden_parameter'].append(hidden)
                data['did_change_point_occur'].append(change_flag)
                data['sequence_id'].append(sequence_id)
                data['p_c_fitted'].append(p_c_fitted[subject_id])

    # Convert collected data into a DataFrame
    df = pd.DataFrame(data)
    return df