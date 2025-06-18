"""
(...)

Author: @emilebdn  
Created date: 2025-06-04
"""
#%%
import sys

import os.path as op
import pandas as pd

from scipy.stats import ttest_ind

# Add the root of the repository to sys.path
sys.path.append(op.dirname(op.dirname(op.abspath(__file__))))

from emilebdn.config.paths import (
    computed_data_emile_path,
    data_outcome_level_preprocessed_path
)
from emilebdn.config.variables import (
    train_size_ratio,
    subject_embedding_dim
)
from emilebdn.GRU.GRU_functions import (
    import_sequences as import_sequences_GRU,
    train_and_evaluate_gru as predict_sequences_with_GRU
)
from emilebdn.HMM.HMM_functions import predict_sequences_with_HMM

#%%
model_1 = 'HMM'
model_2 = 'GRU'
use_subject_embedding = True
model_2_config = 'subject'
subject_embedding_dim = subject_embedding_dim #not influencing the fitting
best_GRU_hidden_layer_size = 1024
data = 'experiment'
data_path = data_outcome_level_preprocessed_path
task_types = ['ada-prob']
content = 'predictions comparison'

print("model_1:", model_1)
print("model_2:", model_2)
print("model_2_config:", model_2_config)
print("best_GRU_hidden_layer_size:", best_GRU_hidden_layer_size)
print("data:", data)
print("data_path:", data_path)
print("task_types:", task_types)
print("content:", content)

#%%
data_outcome_level = pd.read_csv(data_outcome_level_preprocessed_path)

subjects = data_outcome_level['subject'].unique()

for task in task_types:
    for subject in subjects:
        data = data_outcome_level.loc[data_outcome_level['subject'] == subject]
        sequences = import_sequences_GRU(data, data_path, task, use_subject_embedding)[0]

        # Split sequences into train and test sets (e.g., 80% train, 20% test)
        split_idx = int(train_size_ratio * len(sequences))
        train_sequences = sequences[:split_idx]
        test_sequences = sequences[split_idx:]

        GRU_predictions = predict_sequences_with_GRU(train_sequences, test_sequences, return_pred=True)
        HMM_predictions = predict_sequences_with_HMM(train_sequences, test_sequences, task)

#%%
# t-test GRU vs HMM
date = '20250604'
model_dirs = {
    'HMM': 'HMM', 
    'GRU_subject': op.join('GRU', 'subject')
}
data = 'experiment'
task = 'ada-prob'
content = 'mse_evf_scores'
column_names = ['mse', 'evf']

# Load the data from CSV files for both models
hmm_path = op.join(computed_data_emile_path, model_dirs['HMM'], \
                        data, f'{date}_HMM_{data}_{task}_{content}.csv')
gru_path = op.join(computed_data_emile_path, model_dirs['GRU_subject'], \
                        data, f'{date}_GRU_subject_{data}_{task}_{content}.csv')

mse_evf_hmm = pd.read_csv(hmm_path)
mse_evf_gru = pd.read_csv(gru_path)

# Extract MSE and EVF values
hmm_mse = mse_evf_hmm['mse']
hmm_evf = mse_evf_hmm['evf']
gru_mse = mse_evf_gru['mse']
gru_evf = mse_evf_gru['evf']

# Perform t-tests
mse_t_stat, mse_p_value = ttest_ind(hmm_mse, gru_mse)
evf_t_stat, evf_p_value = ttest_ind(hmm_evf, gru_evf)

# Store t-test results in a CSV file

t_test_results = {
    'metric': ['mse', 'evf'],
    't_statistic': [mse_t_stat, evf_t_stat],
    'p_value': [mse_p_value, evf_p_value]
}
t_test_df = pd.DataFrame(t_test_results)

comparison_dir = op.join(computed_data_emile_path, 'models_comparison')
comparison_path = op.join(
    comparison_dir,
    f"{date}_{list(model_dirs.keys())[0]}_vs_{list(model_dirs.keys())[1]}_{data}_{task}_{content}_t-test.csv"
)
t_test_df.to_csv(comparison_path, index=False)
print(f"T-test results saved to {comparison_path}")
# %%
