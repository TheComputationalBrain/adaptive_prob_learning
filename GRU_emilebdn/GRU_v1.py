"""
This script implements a GRU-based recurrent neural network for sequence prediction tasks. 
It includes data preprocessing, model training, and evaluation, with a focus on predicting 
human estimates in an adaptive learning study. The model architecture is not defined in model.py.

Author: @emilebdn  
Created date: 2025-04-15
"""

#%%
import datetime

import os.path as op
import pandas as pd

from functions.GRU_functions import find_best_GRU_hidden_layer_size, import_sequences, plot_results, split_sequences, train_and_evaluate_gru
from config.paths import computed_data_emile_path
from config.variables import task_types, train_size_ratio, best_GRU_hidden_layer_sizes

# %%
data_type = 'simulation'

results = {}

for task in task_types:
    print(f"Running task: {task}")
    # Run the experiment for the specified task
    results[f"{task}'s best GRU hidden layer size"] = find_best_GRU_hidden_layer_size(task, data_type)

# %%
for task in task_types:
    results_path = op.join(computed_data_emile_path, f"20250513_{task}_simulation_results.csv")
    results_df = pd.read_csv(results_path)
    plot_results(task, results_df)

# %%
data_type = 'experiment'
all_results = {}

for task in task_types:
    hidden_size = best_GRU_hidden_layer_sizes[task]
    
    sequences = import_sequences(task, data_type)
    train_sequences, test_sequences = split_sequences(sequences, train_size_ratio)
    
    results = train_and_evaluate_gru(train_sequences, test_sequences, hidden_size=hidden_size)
    
    all_results[task] = results
    print(f"Task: {task}, Training time: {results['training_time']}, Test loss: {results['test_loss']}, Explained Variance: {results['evf']}")  

# Save all_results to a file
today_date = datetime.datetime.now().strftime("%Y%m%d")
output_path = op.join(computed_data_emile_path, f"{today_date}_group_GRU_fit.csv")

all_results_df = pd.DataFrame.from_dict(all_results, orient='index')
all_results_df.to_csv(output_path)

print(f"Results saved to {output_path}")

# %%
data_type = 'experiment'
task = 'ada-pos'
subject_labeling = True
hidden_size = best_GRU_hidden_layer_sizes[task]

sequences = import_sequences(task, data_type, subject_labeling=subject_labeling)
train_sequences, test_sequences = split_sequences(sequences, train_size_ratio)
results = train_and_evaluate_gru(train_sequences, test_sequences, hidden_size=hidden_size)
print(f"Task: {task}, Training time: {results['training_time']}, Test loss: {results['test_loss']}, Explained Variance: {results['evf']}")

# %%
