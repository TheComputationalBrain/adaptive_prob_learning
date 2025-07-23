"""
This script defines paths for accessing and preprocessing outcome-level data, 
accessing already completed analyses and simulation data.

Author: @emilebdn  
Created date: 2025-03-22
"""
#%%
import os
import pandas as pd
import os.path as op

root_dir = op.dirname(op.dirname(op.dirname(__file__)))

data_outcome_level_path = op.join(root_dir, 'data', 'Foucault_Meyniel_2024', \
                    'structured-dataset', 'ada-learn_study', 'data_outcome-level.csv')

computed_data_emile_path = os.path.normpath(
    os.path.join(
        root_dir, '..', '..', 'ebayondenoyer', 'nasShare', \
        'projects', 'protocols', 'AdaptiveProbLearning_ChungMeyniel_2025', \
        'data', 'Foucault_Meyniel_2024', 'computed_data_emile'
    )
)

data_outcome_level_preprocessed_path = op.join(computed_data_emile_path, 'data_outcome_level', \
                                               '20250515_data_outcome_level_preprocessed.csv')

data_outcome_level_simulated_path = op.join(computed_data_emile_path, 'data_outcome_level', \
                                            '20250522_data_outcome_level_simulated.csv')

#%%
# # Read the preprocessed outcome-level data
# df = pd.read_csv(data_outcome_level_preprocessed_path)

# # Check if there are multiple sessions per subject
# sessions_per_subject = df.groupby('subject')['session_idx'].nunique()
# multiple_sessions_per_subject = sessions_per_subject.gt(1).any()

# # Check if there are multiple subjects per session
# subjects_per_session = df.groupby('session_idx')['subject'].nunique()
# multiple_subjects_per_session = subjects_per_session.gt(1).any()

# print("Multiple sessions per subject:", multiple_sessions_per_subject)
# print("Multiple subjects per session:", multiple_subjects_per_session)

# # Check if for fixed session_idx and task, the sequence of outcomes is always the same across subjects
# # Step 1: Group by (session_idx, task, subject) and get the outcome sequence
# grouped = df.groupby(['session_idx', 'task', 'subject'])['outcome'].apply(list).reset_index(name='outcome_sequence')

# # Step 2: Convert lists to tuples to make them hashable
# grouped['outcome_sequence'] = grouped['outcome_sequence'].apply(tuple)

# # Step 3: Group by (session_idx, task) and count unique sequences
# unique_sequences_per_group = grouped.groupby(['session_idx', 'task'])['outcome_sequence'].nunique()

# # Step 4: Check if there's any (session_idx, task) with more than one unique outcome sequence
# inconsistent_outcomes = unique_sequences_per_group.gt(1).any()

# print("Inconsistent outcome sequences for same (session_idx, task):", inconsistent_outcomes)

#%%
