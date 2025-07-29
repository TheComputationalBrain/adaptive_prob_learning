"""
This script randomly selects one subject and one sequence performed by the subject to plot the subject behavior with several model predictions.

Author: @emilebdn
Created date: 2025-06-06
"""
#%%
import datetime
import random
import sys

import numpy as np
import os.path as op

from sklearn.model_selection import train_test_split

sys.path.append(op.dirname(op.dirname(__file__)))

from emilebdn.config.paths import data_outcome_level_preprocessed_path

data_outcome_level_with_pred_path = data_outcome_level_preprocessed_path.replace('.csv', '_ada-prob_with_predictions.csv')

from emilebdn.config.variables import (
    random_state,
    train_size_ratio,
    length, 
)
from emilebdn.GRU.GRU_functions import (
    flatten_sequences,
    format_sequences,
    plot_subject_sequence,
    train_and_evaluate_gru,
)
from emilebdn.HMM.HMM_functions import (
    predict_sequences_with_HMM
)

today = datetime.datetime.now().strftime("%Y%m%d")

model = 'GRU'
model_config = 'subject'
hidden_sizes = {
    'group': 512,
    'subject': 512
}
data = 'experiment'
path = data_outcome_level_preprocessed_path
task = 'ada-prob'
content = 'mse_evf_scores'

print("model:", model)
print("model_config:", model_config)
print("data:", data)
print("data path:", path)
print("task:", task)
print("content:", content)

all_flat_results = {}
all_sequences = format_sequences(path, task)
subject_ids = sorted(set(seq[3] for seq in all_sequences))

# Choose one subject and one sequence randomly
random_subject = random.choice(subject_ids)

# Split into train and test
train_seq_gr, test_seq_gr = train_test_split(all_sequences, test_size=1-train_size_ratio, random_state=random_state)

train_seq_sb = [seq for seq in train_seq_gr if seq[3] == random_subject]
test_seq_sb = [seq for seq in test_seq_gr if seq[3] == random_subject]

train_seq_gr = random.sample(train_seq_gr, len(train_seq_sb))

# Choose randomly one sequence from test_seq_sb
if len(test_seq_sb) > 0:
    tested_sequence = random.choice(test_seq_sb)
else:
    raise RuntimeError("No test sequences found for the selected subject.")

print(f"Randomly selected subject: {random_subject}")
print(f"Train sequences group-level: {len(train_seq_gr)}, Test sequences group-level: {len(test_seq_gr)}")
print(f"Train sequences subject-level: {len(train_seq_sb)}, Test sequences subject-level: {len(test_seq_sb)}")

GRU_group_level_prediction = train_and_evaluate_gru(train_seq_gr, test_seq_gr, hidden_size=hidden_sizes['group'], return_pred=True)['sequence_pairs']
GRU_subject_level_prediction = train_and_evaluate_gru(train_seq_sb, test_seq_sb, hidden_size=hidden_sizes['subject'], return_pred=True)['sequence_pairs']

# Flatten the train and test sequences for HMM prediction
train_seq_gr_HMM = flatten_sequences(train_seq_gr)
test_seq_gr_HMM = flatten_sequences(test_seq_gr, tested_sequence=tested_sequence)
train_seq_sb_HMM = flatten_sequences(train_seq_sb)
test_seq_sb_HMM = flatten_sequences(test_seq_sb, tested_sequence=tested_sequence)

# To match the number of sequences, randomly select the same number of group-level sequences as subject-level
n_seq_sb = len(train_seq_sb_HMM) // length  # Number of subject-level sequences
all_seq_ids = train_seq_gr_HMM['sequence_id_2'].unique()
selected_seq_ids = np.random.choice(all_seq_ids, size=n_seq_sb, replace=False)
train_seq_gr_HMM_matched = train_seq_gr_HMM[train_seq_gr_HMM['sequence_id_2'].isin(selected_seq_ids)]

train_seq_gr_HMM = train_seq_gr_HMM_matched

HMM_group_level_prediction = predict_sequences_with_HMM(train_seq_gr_HMM, test_seq_gr_HMM, int(1/(1-train_size_ratio)))
HMM_subject_level_prediction = predict_sequences_with_HMM(train_seq_sb_HMM, test_seq_sb_HMM, int(1/(1-train_size_ratio)))

# Only select the tested sequence
hmm_sb_predicted_seq = HMM_subject_level_prediction[:length - 1]
hmm_gr_predicted_seq = HMM_group_level_prediction[:length - 1]

# Compute mean subject behavior
# Filter subject_ids to find those who did the task with the tested_sequence
filtered_subject_ids = [
    seq[3] for seq in all_sequences if np.array_equal(seq[1], tested_sequence[1])
]

print(f"Number of filtered subject IDs: {len(filtered_subject_ids)}")

# Extract subject behavior for these filtered subjects
filtered_subject_behaviors = [
    seq[2] for seq in all_sequences if seq[3] in filtered_subject_ids and np.array_equal(seq[1], tested_sequence[1])
]

# Compute the mean subject behavior
mean_subject_behavior = np.mean(filtered_subject_behaviors, axis=0)

print(mean_subject_behavior)

for i in range(len(GRU_subject_level_prediction)):
    # Compare values by converting both to numpy arrays
    left = GRU_subject_level_prediction[i][1]
    right = tested_sequence[1]
    if np.allclose(np.asarray(left), np.asarray(right)):
        hidden_params, outcome_seq, subject_seq, gru_sb_predicted_seq = GRU_subject_level_prediction[i]


for i in range(len(GRU_group_level_prediction)):
    # Compare values by converting both to numpy arrays
    left = GRU_group_level_prediction[i][1]
    right = tested_sequence[1]
    if np.allclose(np.asarray(left), np.asarray(right)):
        _, __, ___, gru_gr_predicted_seq = GRU_group_level_prediction[i]

plot_subject_sequence(
    outcome_seq, 
    hidden_params,
    len(train_seq_sb),
    len(train_seq_gr), 
    random_subject,
    hidden_sizes,
    subject_seq=subject_seq,
    mean_subject_behavior=None, #mean_subject_behavior,
    gru_sb_predicted_seq=gru_sb_predicted_seq,
    gru_gr_predicted_seq=gru_gr_predicted_seq,
    hmm_sb_predicted_seq=hmm_sb_predicted_seq,
    hmm_gr_predicted_seq=hmm_gr_predicted_seq
)