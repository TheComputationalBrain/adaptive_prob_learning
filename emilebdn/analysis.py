### '20250709'
#%%
### 0 - Imports and config
import sys

import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import seaborn as sns

from joblib import Parallel, delayed
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_squared_error

sys.path.append(op.dirname(op.dirname(__file__)))

from data_analysis_utils import fit_model
from emilebdn.config.paths import (
    data_outcome_level_preprocessed_path,
    data_outcome_level_simulated_path
)
from emilebdn.config.variables import (
    n_jobs,
    random_state,
    train_size_ratio,
    expID,
    length, 
    n_sequences_for_each_subject
)
from emilebdn.HMM.HMM_functions import HMM_prediction

random_state = random_state
task = 'ada-prob'
data_path = data_outcome_level_preprocessed_path.replace('.csv', f'_{task}_with_predictions.csv')
data = pd.read_csv(data_path)
subjects = data['subject'].unique().tolist()
nb_subjects = len(subjects)
subject_ids = {subject: i for i, subject in enumerate(subjects)}
n_seq = n_sequences_for_each_subject[task]
n_tri = length
n_test_seqs = int((1 - train_size_ratio)*n_seq) + 1
nb_test_seqs = int(n_seq / n_test_seqs)
models = [
    'estimate',
    'mean_estimate_without_cv', 'mean_estimate_with_cv', 
    'subject_HMM', 'optimal_HMM', 
    'subject_GRU_32', 'subject_GRU_512', 'subject_GRU_1024',
    'group_GRU_32', 'group_GRU_512', 'group_GRU_1024', 
    'big_group_GRU_32', 'big_group_GRU_512', 'big_group_GRU_1024',
    'subject_HMM_with_FNN_4', 'subject_HMM_with_FNN_8', 'subject_HMM_with_FNN_16',
    'subject_HMM_with_FNN_32', 'subject_HMM_with_FNN_512', 'subject_HMM_with_FNN_1024', 
    'big_group_HMM_with_FNN_4', 'big_group_HMM_with_FNN_8', 'big_group_HMM_with_FNN_16',
    'big_group_HMM_with_FNN_32', 'big_group_HMM_with_FNN_512', 'big_group_HMM_with_FNN_1024'
]

#%%
### 1 - Compute MSE and mean evs across subjects between subject behavior and models
results = []
reference = 'estimate'

for model in models:
    mse = mean_squared_error(data[reference], data[model])
    evs = []
    for subject in subjects:
        subject_data = data[data['subject'] == subject]
        evs.append(explained_variance_score(subject_data[reference], subject_data[model]))
    mean_evs = np.mean(evs)
    std_evs = np.std(evs)
    results.append({'model': model, 'mse': mse, 'mean_evs': mean_evs, 'std_evs': std_evs})

results_df = pd.DataFrame(results)
results_df.to_csv(data_path.replace('.csv', '_mse_mean_evs_vs_estimate.csv'), index=False)
print(results_df)

#%%
### 2 - Compare the first prediction of each sequence for each model, conditioning
### or without conditionning on outcome == 0 (yellow) or outcome == 1 (blue)

# Extract only the first row of each sequence
first_rows = data.iloc[::n_tri].copy()

def compute_stats(df, models):
    stats = []
    for model in models:
        vals = df[model]
        stats.append({
            'model': model,
            'mean': np.nanmean(vals),
            'std': np.nanstd(vals)
        })
    return pd.DataFrame(stats)

# No conditioning
overall_stats = compute_stats(first_rows, models)
print("Overall stats (no conditioning):")
print(overall_stats)

# Conditioning on outcome == 1
stats_outcome_1 = compute_stats(first_rows[first_rows['outcome'] == 1], models)
print("\nStats conditioned on outcome == 1:")
print(stats_outcome_1)

# Conditioning on outcome == 0
stats_outcome_0 = compute_stats(first_rows[first_rows['outcome'] == 0], models)
print("\nStats conditioned on outcome == 0:")
print(stats_outcome_0)

# Save the overall_stats, stats_outcome_1, and stats_outcome_0 to CSV
all_stats = pd.concat([
    overall_stats.assign(condition='all'),
    stats_outcome_1.assign(condition='outcome==1'),
    stats_outcome_0.assign(condition='outcome==0')
], ignore_index=True)
all_stats.to_csv(data_outcome_level_preprocessed_path.replace('.csv', '_ada-prob_with_predictions_first_outcome_stats.csv'), index=False)

#%%
### 3 - Compute linear regression between hidden_parameter and models,
### similarly to Figure 4 (Chung and Meyniel, 2025)
data = pd.read_csv(data_path)
reference = 'hidden_parameter'

regression_results = []

for model in models:
    if model == reference:
        # For the reference variable, linregress against itself
        slope, intercept, r_value, p_value, std_err = 1.0, 0.0, 1.0, 0.0, 0.0
    else:
        df = data[[reference, model]]
        slope, intercept, r_value, p_value, std_err = linregress(df[reference], df[model])
    regression_results.append({
        'model': model,
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err
    })

regression_df = pd.DataFrame(regression_results)
print(regression_df)
regression_df.to_csv(
    data_path.replace(
        '.csv', '_regression_vs_hidden_parameter_results.csv'
    ),
    index=False
)

# Plotting (sampled, to avoid memory issues)
sample_size = 200 
n_cols = 2
n_rows = int(np.ceil(len(models) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7*n_rows))  # Make figure larger for square axes
axes = axes.flatten()

for idx, model in enumerate(models):
    ax = axes[idx]
    # Plot model vs reference as before
    if model == reference:
        df = data[[reference]]
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=random_state)
        else:
            df_sample = df
        ax.scatter(df_sample[reference], df_sample[reference], alpha=0.5)
        ax.plot(df_sample[reference], df_sample[reference], color='red', label='y=x')
        ax.set_title(f'{reference} vs {reference}\n$y = x$')
        ax.set_xlabel(reference)
        ax.set_ylabel(reference)
        ax.set_aspect('equal', adjustable='datalim')
        # Make axes square
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        continue
    df = data[[reference, model]]
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=random_state)
    else:
        df_sample = df
    reg_row = regression_df[regression_df['model'] == model]
    slope = reg_row['slope'].values[0]
    intercept = reg_row['intercept'].values[0]
    sns.regplot(
        data=df_sample,
        x=reference,
        y=model,
        scatter=True,
        line_kws={'color': 'red'},
        ax=ax
    )
    ax.set_title(f'{model} vs {reference}\n$y = {slope:.3f}x + {intercept:.3f}$')
    ax.set_xlabel(reference)
    ax.set_ylabel(model)
    ax.set_aspect('equal', adjustable='datalim')
    # Make axes square
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.set_xlim(lims)
    ax.set_ylim(lims)

# Hide unused subplots
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#%%
### Figure 5
data = pd.read_csv(data_path)

# Pour chaque séquence (75 lignes), détecter les change points et faire la régression demandée

n_tri = length  # nombre de trials par séquence
window_before = -5
window_after = 15

windows_by_position = {k: [] for k in range(window_before, window_after + 1)}                

n_seq_total = len(data) // n_tri

def process_sequence(seq_idx):
    print(f"Processing sequence {seq_idx}")
    seq_start = seq_idx*n_tri
    seq_end = seq_start + n_tri
    sequence_df = data.iloc[seq_start:seq_end].copy()
    sequence_df.reset_index(drop=True, inplace=True)
    
    change_points = sequence_df.index[
        (sequence_df['did_change_point_occur'] == True) & (sequence_df.index > 0)
    ].tolist()
    
    rows_by_position = {k: [] for k in range(window_before, window_after + 1)}
    for cp_idx in change_points:
        for rel_pos in range(window_before, window_after + 1):
            abs_idx = cp_idx + rel_pos
            if 0 <= abs_idx < n_tri - 1:
                row = sequence_df.iloc[abs_idx].copy()
                row['relative_position'] = rel_pos
                row['hidden_parameter_before'] = sequence_df.iloc[cp_idx - 1]['hidden_parameter'] 
                row['hidden_parameter_after'] = sequence_df.iloc[cp_idx]['hidden_parameter']
                rows_by_position[rel_pos].append(row)
    return rows_by_position

# Parallelize over all sequences
all_rows_by_position = Parallel(n_jobs=50)(
    delayed(process_sequence)(seq_idx) for seq_idx in range(n_seq_total)
)

# Aggregate results into windows_by_position
for rows_by_position in all_rows_by_position:
    for rel_pos, rows in rows_by_position.items():
        windows_by_position[rel_pos].extend(rows)
    
regression_by_position = {k: [] for k in windows_by_position.keys()}

def regression_for_position(rel_pos, rows):
    print(f"Running multilinear regression for relative position {rel_pos} with {len(rows)} rows")
    results = []
    if not rows:
        return rel_pos, results
    df = pd.DataFrame(rows)
    for model in models:
        subdf = df[[model, 'hidden_parameter_before', 'hidden_parameter_after']]
        if len(subdf) < 3:
            results.append({
                'model': model,
                'n': len(subdf),
                'a': np.nan,
                'b': np.nan,
                'c': np.nan,
                'r2': np.nan
            })
            continue
        X = subdf[['hidden_parameter_before', 'hidden_parameter_after']].values
        y = subdf[model].values
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, y)
        a, b = reg.coef_
        c = reg.intercept_
        r2 = reg.score(X, y)
        results.append({
            'model': model,
            'n': len(subdf),
            'a': a,
            'b': b,
            'c': c,
            'r2': r2
        })
    return rel_pos, results

parallel_results = Parallel(n_jobs=50)(
    delayed(regression_for_position)(rel_pos, rows)
    for rel_pos, rows in windows_by_position.items()
)

for rel_pos, results in parallel_results:
    regression_by_position[rel_pos] = results

    # Print regression results by relative position
    for rel_pos in sorted(regression_by_position.keys()):
        print(f"Relative position {rel_pos}:")
        for res in regression_by_position[rel_pos]:
            print(res)
        print("-"*40)

# Save regression results to CSV
# Save regression results to CSV for each relative position
all_regression_rows = []
for rel_pos, results in regression_by_position.items():
    for res in results:
        row = {'relative_position': rel_pos}
        row.update(res)
        all_regression_rows.append(row)
regression_by_position_df = pd.DataFrame(all_regression_rows)
regression_by_position_df.to_csv(
    data_outcome_level_preprocessed_path.replace(
        '.csv', '_ada-prob_with_predictions_figure_5_results.csv'
    ),
    index=False
)

#%%
# Plot 'a' (dashed) and 'b' (solid) coefficients vs relative position for each model
# Use the same color for both lines of the same model

def plot_regression_coefficients(regression_by_position, models, display_models=None):
    """
    Plot regression coefficients 'a' (dashed) and 'b' (solid) vs relative position for each model.

    Args:
        regression_by_position: dict mapping relative position to list of regression result dicts.
        models: list of model names.
        display_models: list of model names to display (default: all in models).
    """
    positions = sorted(regression_by_position.keys())
    model_names = models
    if display_models is None:
        display_models = model_names

    # Assign a unique color to each model
    cmap = plt.get_cmap('tab10')
    color_map = {model: cmap(i % 10) for i, model in enumerate(model_names)}

    plt.figure(figsize=(16, 12))
    for model in model_names:
        if model not in display_models:
            continue
        a_values = []
        b_values = []
        for pos in positions:
            res_list = regression_by_position[pos]
            a = np.nan
            b = np.nan
            for res in res_list:
                if res['model'] == model:
                    a = res['a']
                    b = res['b']
                    break
            a_values.append(a)
            b_values.append(b)
        color = color_map[model]
        plt.plot(positions, a_values, marker='o', linestyle='--', color=color, label=f"{model} (hidden_param_bef, dashed)")
        plt.plot(positions, b_values, marker='s', linestyle='-', color=color, label=f"{model} (hidden_param_aft, solid)")

    plt.xlabel('Relative Position')
    plt.ylabel("Regression coefficients 'hidden_param_bef' (dashed), 'hidden_param_aft' (solid)")
    plt.title("Coefficients 'hidden_param_bef' (dashed) and 'hidden_param_aft' (solid) vs Relative Position for each model")
    plt.axvline(x=0, color='red', linestyle='--', label='Change-point')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

# Example usage:
plot_regression_coefficients(regression_by_position, models, display_models= None)
plot_regression_coefficients(regression_by_position, models, display_models= ['estimate', 'subject_GRU_512', 'subject_HMM', 'HMM_with_FNN_512'])

plot_regression_coefficients(regression_by_position, models, display_models= ['estimate', 'subject_HMM', 'subject_HMM_with_FNN_32', 'subject_HMM_with_FNN_512', 'subject_HMM_with_FNN_1024'])

#%%
### HMM Parameter Recovery
simulated_outcomes = pd.read_csv(data_outcome_level_simulated_path)
task = 'ada-prob'
model = 'HMM'

simulated_outcomes = simulated_outcomes[simulated_outcomes['task'] == task]
simulated_outcomes = simulated_outcomes[['subject', 'session_idx', 'outcome']]

subjects = simulated_outcomes['subject'].unique()

generative_p_c = {subject: np.random.uniform(0.53, 0.57) for subject in subjects}

fitted_p_c = {subject: None for subject in subjects}

def fit_subject_p_c(i, subject):
    subject_outcomes = simulated_outcomes[simulated_outcomes['subject'] == subject]
    predictions = HMM_prediction(generative_p_c[subject], subject_outcomes, task)
    sessions_idx = subject_outcomes['session_idx'].unique()
    return subject, fit_model(expID, model, i, sessions_idx)[1]

results = Parallel(n_jobs=n_jobs)(
    delayed(fit_subject_p_c)(i, subject) for i, subject in enumerate(subjects)
)
for subject, fit_val in results:
    fitted_p_c[subject] = fit_val

gen_p_c = np.array([generative_p_c[subject] for subject in subjects])
fit_p_c = np.array([fitted_p_c[subject] for subject in subjects])

mse_p_c = mean_squared_error(gen_p_c, fit_p_c)
print(f"MSE between generative_p_c and fitted_p_c: {mse_p_c}")