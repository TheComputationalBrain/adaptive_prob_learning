"""
This module defines configuration variables for GRU modeling, HMM modeling, 
and simulating subjects' behavior in adaptive learning tasks. 

Author: @emilebdn  
Created date: 2025-04-30
"""
n_jobs = 40 # Number of parallel jobs for data processing
nb_subjects = 94 # Number of subjects

# -------------------------------
# GRU hyperparameters
# -------------------------------

data_types = ['experiment', 'simulation']  # Data sources ('experiment' or 'simulation')
data_type = data_types[1]  # Data source to be used ('experiment' or 'simulation')
task_types = ['ada-pos', 'ada-prob']  # Task types ('ada-pos' and 'ada-prob')
task = task_types[0]  # Task to be used ('ada-pos' or 'ada-prob')
train_size_ratio = 0.8  # Ratio of training data to total data
input_size = 1  # Input size for the GRU model
hidden_size = 1024  # Hidden layer size for the GRU model
max_hidden_size = 2048 # Maximum hidden layer size for simulation
output_size = 1  # Output size for the GRU model
learning_rate = 1e-4  # Learning rate for the optimizer
num_epochs = 100  # Number of training epochs
batch_size = 16  # Batch size for data loaders
best_GRU_hidden_layer_sizes = {'ada-pos': 1024, 'ada-prob': 16}  # Best hidden layer sizes for each task

use_subject_embedding = True  # Flag to use subject embedding
subject_embedding_dim = 8  # Dimensionality of the subject embedding

# -------------------------------
# HMM hyperparameters
# -------------------------------

# Bounds for p_c (HMM model parameter)
p_c_bounds = {
    'ada-pos': {'min': 1e-4, 'max': 0.9},
    'ada-prob': {'min': 0, 'max': 1}
}  # Bounds for p_c (HMM model parameter)

# ada-pos
b_0 = 0.5  # Prior belief about the mean
tau_0 = 0.5  # Prior belief about the variance
n_initial_guesses = 100  # Number of initial guesses for the optimization

# ada-prob
expID = 3  # Experiment ID
model = 'HMM'  # Model type ('HMM')
resol= 20  # Resolution for the model
do_inference_on_current_trial = True  # Flag for inference on the current trial

nb_of_restart_for_Powell = 100  # Number of restarts for the Powell optimization method

# -------------------------------
# 'Observation sequences and sequence-generating processes of the tasks'
# -------------------------------

length = 75 # Length of each sequence
n_sequences_for_each_subject = {'ada-pos': 6, 'ada-prob': 15}  # Number of sequences for each subject

n_sequences_pos = 100 # Number of sequences for the magnitude task
change_prob_pos = 1/10 # Probability of changing the mean
freeze_duration_pos = 3 # Duration of freezing (number of outcomes)
std_dev_pos = 10/300 # Standard deviation of the observations

n_sequences_prob = 150 # Number of sequences for the probability task
change_prob_prob = 1/20 # Change-point probability
freeze_duration_prob = 6 # Duration of freezing (number of outcomes)
min_val_prob = 1/10 # Minimum value for probability
max_val_prob = 9/10 # Maximum value for probability
odds_change_threshold_prob = 4 # Odds change threshold for sampling new probabilities