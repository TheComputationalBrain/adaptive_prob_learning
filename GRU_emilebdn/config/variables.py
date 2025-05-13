# created by @emilebdn on 2025/04/30

n_jobs = 40

# -------------------------------
# GRU hyperparameters
# -------------------------------

task_types = ['ada-pos', 'ada-prob']  # Task types ('ada-pos' and 'ada-prob')
task = task_types[0]  # Task to be used ('ada-pos' or 'ada-prob')
data_types = ['experiment', 'simulation']  # Data sources ('experiment' or 'simulation')
data_type = data_types[1]  # Data source to be used ('experiment' or 'simulation')
train_size_ratio = 0.8  # Ratio of training data to total data
input_size = 1  # Input size for the GRU model
hidden_size = 8  # Hidden layer size for the GRU model
max_hidden_size = 2048 # Maximum hidden layer size for simulation
output_size = 1  # Output size for the GRU model
learning_rate = 1e-4  # Learning rate for the optimizer
num_epochs = 100  # Number of training epochs
batch_size = 16  # Batch size for data loaders
best_GRU_hidden_layer_sizes = {'ada-pos': 256, 'ada-prob': 16}  # Best hidden layer sizes for each task

# -------------------------------
# HMM hyperparameters
# -------------------------------

# ada-pos
b_0 = 0.5  # Prior belief about the mean
tau_0 = 0.5  # Prior belief about the variance

# ada-prob
expID = 3  # Experiment ID
model = 'HMM'  # Model type ('HMM')
resol= 20  # Resolution for the model
p1_min = 0  # Minimum value for p1 (HMM model parameter)
p1_max = 1  # Maximum value for p1 (HMM model parameter)
do_inference_on_current_trial = True  # Flag for inference on the current trial

# -------------------------------
# 'Observation sequences and sequence-generating processes of the tasks'
# -------------------------------

length = 75 # Length of each sequence

n_sequences_pos = 100 # Number of sequences for the magnitude task
change_prob_pos = 1/10 # Probability of changing the mean
freeze_duration_pos = 3 # Duration of freezing (number of outcomes)
std_dev_pos = 10/300 # Standard deviation of the observations
n_sequences_for_each_subject_pos = 6

n_sequences_prob = 150 # Number of sequences for the probability task
change_prob_prob = 1/20 # Change-point probability
freeze_duration_prob = 6 # Duration of freezing (number of outcomes)
min_val_prob = 1/10 # Minimum value for probability
max_val_prob = 9/10 # Maximum value for probability
odds_change_threshold_prob = 4 # Odds change threshold for sampling new probabilities
n_sequences_for_each_subject_prob = 15