"""
This script performs a comprehensive analysis of model predictions against experimental data for probability learning task.

Key functionalities include:
- Importing and configuring data for analysis.
- Computing MSE and EVS for each subject and model, followed by aggregation.
- Performing paired t-tests to compare model performance.
- Visualizing mean MSE and EVS with error bars.
- Analyzing the first prediction of each sequence under different conditions.
- Conducting linear regression to assess the relationship between hidden parameters and model predictions.
- Investigating regression coefficients before and after change points.

Author: @emilebdn
Created date: 2025-07-09
"""
#%%
### 0 - Imports and config
import os
import sys

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import seaborn as sns

from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_squared_error
from scipy.stats import ttest_rel 

sys.path.append(op.dirname(op.dirname(__file__)))

from emilebdn.config.paths import data_outcome_level_preprocessed_path
from emilebdn.config.variables import (
    random_state,
    task,
    train_size_ratio,
    length, 
    n_sequences_for_each_subject
)

random_state = random_state

data_path = data_outcome_level_preprocessed_path.replace('.csv', f'_{task}_with_predictions.csv')
data = pd.read_csv(data_path)

subjects = data['subject'].unique().tolist()

n_seq = n_sequences_for_each_subject[task] # Number of sequences per subject
n_tri = length # Number of trials per sequence
n_test_seqs = int((1 - train_size_ratio)*n_seq) + 1 # Number of test sequences per test sequences' set
nb_test_seqs = int(n_seq / n_test_seqs) # Number of test sequences' sets per subject

models = [
    'estimate', 'mean_estimate_without_cv', 'mean_estimate_with_cv',
    'subject_HMM', 'optimal_HMM', 'group_HMM', 'big_group_HMM',
    'subject_RNN_32', 'subject_RNN_512', 'subject_RNN_1024', 'subject_RNN_2048',
    'group_RNN_32', 'group_RNN_512', 'group_RNN_1024', 'group_RNN_2048',
    'big_group_RNN_32', 'big_group_RNN_512', 'big_group_RNN_1024',
    'subject_GRU_32', 'subject_GRU_512', 'subject_GRU_1024', 'subject_GRU_2048',
    'group_GRU_32', 'group_GRU_512', 'group_GRU_1024', 'group_GRU_2048',
    'big_group_GRU_32', 'big_group_GRU_512', 'big_group_GRU_1024',
    'subject_HMM_with_FNN_32', 'subject_HMM_with_FNN_512', 'subject_HMM_with_FNN_1024',
    'big_group_HMM_with_FNN_4', 'big_group_HMM_with_FNN_8', 'big_group_HMM_with_FNN_16',
    'big_group_HMM_with_FNN_32', 'big_group_HMM_with_FNN_512', 'big_group_HMM_with_FNN_1024',
]

#%%
### 1 - Compute MSE and EVS for each subject and model, then aggregate
reference = 'estimate'

# Compute MSE and EVS per subject and model
subject_model_scores = [
    {
        'subject': subject,
        'model': model,
        'mse': mean_squared_error(
            data.loc[data['subject'] == subject, reference],
            data.loc[data['subject'] == subject, model]
        ),
        'evs': explained_variance_score(
            data.loc[data['subject'] == subject, reference],
            data.loc[data['subject'] == subject, model]
        )
    }
    for model in models for subject in subjects
]

scores_df = pd.DataFrame(subject_model_scores)
scores_df.to_csv(
    data_path.replace('.csv', '_mse_evs_per_subject_per_model.csv'),
    index=False
)

#%%
reference_model = 'subject_HMM'

# Compute mean MSE, mean EVS, and std EVS per model, preserving models order
agg_df = pd.DataFrame([
    {
        'model': model,
        'mean_mse': scores_df[scores_df['model'] == model]['mse'].mean(),
        'std_mse': scores_df[scores_df['model'] == model]['mse'].std(),
        'mean_evs': scores_df[scores_df['model'] == model]['evs'].mean(),
        'std_evs': scores_df[scores_df['model'] == model]['evs'].std(),
        f'mean_diff_mse_with_{reference_model}': (
            scores_df[scores_df['model'] == model]['mse'].reset_index(drop=True) -
            scores_df[scores_df['model'] == reference_model]['mse'].reset_index(drop=True)
        ).mean(),
        f'std_diff_mse_with_{reference_model}': (
            scores_df[scores_df['model'] == model]['mse'].reset_index(drop=True) -
            scores_df[scores_df['model'] == reference_model]['mse'].reset_index(drop=True)
        ).std(),
        f'mean_diff_evs_with_{reference_model}': (
            scores_df[scores_df['model'] == model]['evs'].reset_index(drop=True) -
            scores_df[scores_df['model'] == reference_model]['evs'].reset_index(drop=True)
        ).mean(),
        f'std_diff_evs_with_{reference_model}': (
            scores_df[scores_df['model'] == model]['evs'].reset_index(drop=True) -
            scores_df[scores_df['model'] == reference_model]['evs'].reset_index(drop=True)
        ).std(),
    }
    for model in models
])

agg_df.to_csv(
    data_path.replace('.csv', '_mse_mean_evs_vs_estimate.csv'),
    index=False
)
print(agg_df)

#%%
### 2 - Operate t-test on MSE and EVS for every pair of models
scores_df = pd.read_csv(data_path.replace('.csv', '_mse_evs_per_subject_per_model.csv'))

# Prepare to store t-test results
t_test_results = []

# For each pair of models, perform paired t-test on MSE and EVS across subjects
for i, model1 in enumerate(models):
    for j, model2 in enumerate(models):
        if j <= i:
            continue  # Avoid duplicate pairs and self-comparison
        # Get subject-wise MSE and EVS for both models
        mse1 = scores_df[scores_df['model'] == model1]['mse'].values
        mse2 = scores_df[scores_df['model'] == model2]['mse'].values
        evs1 = scores_df[scores_df['model'] == model1]['evs'].values
        evs2 = scores_df[scores_df['model'] == model2]['evs'].values

        if len(mse1) != len(mse2) or len(evs1) != len(evs2):
            print(f"Length mismatch for models: {model1} ({len(mse1)},{len(evs1)}) vs {model2} ({len(mse2)},{len(evs2)})")

        # Paired t-test (subjects are matched)
        t_mse, p_mse = ttest_rel(mse1, mse2)
        t_evs, p_evs = ttest_rel(evs1, evs2)

        t_test_results.append({
            'model1': model1,
            'model2': model2,
            't_mse': t_mse,
            'p_mse': p_mse,
            't_evs': t_evs,
            'p_evs': p_evs
        })

t_test_df = pd.DataFrame(t_test_results)
t_test_df.to_csv(
    data_path.replace('.csv', '_mse_evs_ttest_results.csv'),
    index=False
)
print(t_test_df)

#%%
### 3 - Plot mean MSE and mean EVS for each model, with error bars for std
agg_df = pd.read_csv(data_path.replace('.csv', '_mse_mean_evs_vs_estimate.csv'))
    
def plot_mse_evs_barplot(agg_df, sort_by=None, diff_with_reference_model=None, show_model_legend=True):
    """
    Plot Mean MSE (bar) and Mean EVS (point ± std) for each model.
    If diff_with_reference_model is not None (str), plot difference with reference model.

    Args:
        agg_df (pd.DataFrame): DataFrame with columns for mean/std MSE/EVS and optionally diff columns.
        sort_by (str or None): None (no sorting), 'mse'/'evs' (sort by mse/evs)
        diff_with_reference_model (str or None): Reference model name for difference columns, or None.
        show_model_legend (bool): Whether to display the model number-to-name mapping below the plot.

    Returns:
        pd.DataFrame: DataFrame mapping model index `i` to `modelname_i`
    """
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'font.size': 12,
    })

    df = agg_df.copy()
    if diff_with_reference_model is not None:
        mse_col = f'mean_diff_mse_with_{diff_with_reference_model}'
        evs_col = f'mean_diff_evs_with_{diff_with_reference_model}'
        std_evs_col = f'std_diff_evs_with_{diff_with_reference_model}'
    else:
        mse_col = 'mean_mse'
        evs_col = 'mean_evs'
        std_evs_col = 'std_evs'

    if sort_by == 'mse':
        df = df.sort_values(mse_col, ascending=True)
    elif sort_by == 'evs':
        df = df.sort_values(evs_col, ascending=False)

    model_names = df['model'].tolist()
    model_numbers = list(range(1, len(model_names) + 1))
    model_number_map = {name: num for name, num in zip(model_names, model_numbers)}

    fig, ax1 = plt.subplots(figsize=(12, 7))

    bar_width = 0.5
    bar_color = sns.color_palette("Blues", n_colors=3)[2]
    evs_color = sns.color_palette("Oranges", n_colors=3)[2]

    bars = ax1.bar(
        model_numbers,
        df[mse_col],
        width=bar_width,
        color=bar_color,
        alpha=0.85,
        label='Mean MSE' if diff_with_reference_model is None else f'Mean ΔMSE vs {diff_with_reference_model}',
        edgecolor='black'
    )
    ax1.set_ylabel(
        'Mean MSE' if diff_with_reference_model is None else f'Mean ΔMSE vs {diff_with_reference_model}',
        color=bar_color, fontsize=13
    )
    ax1.tick_params(axis='y', labelcolor=bar_color)
    ax1.set_xticks(model_numbers)
    ax1.set_xticklabels(model_numbers, rotation=0, ha='center', fontsize=12)
    ax1.set_xlabel('Model (see legend below)' if show_model_legend else 'Model', fontsize=13)

    ax2 = ax1.twinx()
    evs = ax2.errorbar(
        model_numbers,
        df[evs_col],
        yerr=df[std_evs_col],
        fmt='o',
        markersize=10,
        color=evs_color,
        ecolor='gray',
        elinewidth=2,
        capsize=7,
        label='Mean EVS ± STD' if diff_with_reference_model is None else f'Mean ΔEVS ± STD vs {diff_with_reference_model}',
        zorder=10
    )
    ax2.set_ylabel(
        'Mean EVS' if diff_with_reference_model is None else f'Mean ΔEVS vs {diff_with_reference_model}',
        color=evs_color, fontsize=13
    )
    ax2.tick_params(axis='y', labelcolor=evs_color)
    ax2.set_ylim(-1, 1)

    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    fig.patch.set_facecolor('white')
    ax1.set_axisbelow(True)

    # Custom legend (compact)
    handles = [
        mpatches.Patch(
            color=bar_color,
            label='Mean MSE' if diff_with_reference_model is None else f'Mean ΔMSE vs {diff_with_reference_model}'
        ),
        plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor=evs_color, markersize=10,
            label='Mean EVS ± STD' if diff_with_reference_model is None else f'Mean ΔEVS ± STD vs {diff_with_reference_model}',
            markeredgecolor='gray'
        )
    ]
    ax1.legend(handles=handles, loc='upper right', fontsize=10, frameon=True)

    plt.title(
        f"Model Comparison: {'Mean MSE (bar) and Mean EVS (point ± std)' if diff_with_reference_model is None else f'Difference vs {diff_with_reference_model}'}"
        f", sort_by: {sort_by}",
        fontsize=15, pad=15
    )
    plt.tight_layout()

    if show_model_legend:
        mapping_text = ", ".join([f"{num}: {name}" for num, name in zip(model_numbers, model_names)])
        plt.figtext(
            0.5, -0.08, f"Model mapping: {mapping_text}",
            ha='center', va='top', fontsize=10, wrap=True
        )
        plt.subplots_adjust(bottom=0.2)
    else:
        plt.subplots_adjust(bottom=0.1)

    plt.show()

    plt.savefig(
        data_path.replace('.csv', f'_models_comparison_sort_by_{sort_by}_diff_with_{diff_with_reference_model}.png'),
    )
    print(f"Saved plot to {data_path.replace('.csv', f'_models_comparison_sort_by_{sort_by}_diff_with_{diff_with_reference_model}.png')}")

    pd.DataFrame({'Index': model_numbers, 'Model name': model_names}).to_csv(
        data_path.replace('.csv', f'_models_comparison_sort_by_{sort_by}_diff_with_{diff_with_reference_model}.csv'),
        index=False
    )
    print(f"Model mapping saved to CSV: {data_path.replace('.csv', f'_models_comparison_sort_by_{sort_by}_diff_with_{diff_with_reference_model}.csv')}")

plot_mse_evs_barplot(agg_df, sort_by='evs', diff_with_reference_model=reference_model, show_model_legend=False)
plot_mse_evs_barplot(agg_df, sort_by='mse', diff_with_reference_model=reference_model, show_model_legend=False)
plot_mse_evs_barplot(agg_df, sort_by=None, diff_with_reference_model=reference_model, show_model_legend=False)
plot_mse_evs_barplot(agg_df, sort_by='evs', diff_with_reference_model=None, show_model_legend=False)
plot_mse_evs_barplot(agg_df, sort_by='mse', diff_with_reference_model=None, show_model_legend=False)
plot_mse_evs_barplot(agg_df, sort_by=None, diff_with_reference_model=None, show_model_legend=False)

#%%
### 4 - Plot MSE and EVS for each subject and a chosen model, with bars for mean ± std
scores_df = pd.read_csv(data_path.replace('.csv', '_mse_evs_per_subject_per_model.csv'))

def plot_subjectwise_comparison(scores_df, model1, model2):
    """
    Plot subject-wise comparison between two models for both MSE and EVS.
    For each model and metric, show a horizontal bar from mean-std to mean+std,
    with a large dot at the mean value.
    Also, connect subject values with lines.
    """
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'font.size': 12,
    })

    metrics = ['mse', 'evs']
    metric_labels = {'mse': 'MSE', 'evs': 'EVS'}

    # Set beautiful seaborn style
    sns.set(style="whitegrid")
    mse_color = sns.color_palette("Blues", n_colors=3)[2]
    evs_color = sns.color_palette("Oranges", n_colors=3)[2]
    bar_colors = [mse_color, evs_color]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        df1 = scores_df[scores_df['model'] == model1].sort_values('subject')
        df2 = scores_df[scores_df['model'] == model2].sort_values('subject')
        subjects = df1['subject'].values
        vals1 = df1[metric].values
        vals2 = df2[metric].values
        mean1, std1 = np.mean(vals1), np.std(vals1)
        mean2, std2 = np.mean(vals2), np.std(vals2)

        x = np.array([0, 1])

        # Draw thick horizontal bars for mean ± std
        for i, (mean, std, color) in enumerate(zip([mean1, mean2], [std1, std2], bar_colors)):
            ax.plot([i - 0.15, i + 0.15], [mean - std, mean - std], color=color, linewidth=8, solid_capstyle='round', alpha=0.8, zorder=2)
            ax.plot([i - 0.15, i + 0.15], [mean + std, mean + std], color=color, linewidth=8, solid_capstyle='round', alpha=0.8, zorder=2)
            ax.plot([i, i], [mean - std, mean + std], color=color, linewidth=8, alpha=0.3, zorder=1)
            ax.scatter(i, mean, color=color, s=200, edgecolor='black', zorder=3)

        # Connect subject-wise values with slightly shortened horizontal segments
        delta = 0.2
        for v1, v2 in zip(vals1, vals2):
            ax.plot([0 + delta, 1 - delta], [v1, v2], color='gray', alpha=0.4, linewidth=1, zorder=0)
            ax.scatter([0 + delta, 1 - delta], [v1, v2], color='black', s=20, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels([model1, model2], rotation=15, fontsize=13)
        ax.set_ylabel(metric_labels[metric], fontsize=14)
        ax.set_title(f'Subject-wise {metric_labels[metric]}', fontsize=16)
        ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

        if metric == 'evs':
            ax.set_ylim(-1, 1)
            ax.axhline(0, color='black', linestyle=':', linewidth=1, zorder=1)

        # Add a custom legend below the plot
        mean_dot = Line2D([0], [0], marker='o', color='w', label='Mean value',
                          markerfacecolor='gray', markeredgecolor='black', markersize=12)
        std_bar = Line2D([0], [0], color='gray', linewidth=8, alpha=0.3, label='± 1 std. range')
        subject_line = Line2D([0], [0], color='gray', linewidth=1, alpha=0.4, label='Subject values')
        fig.legend(handles=[mean_dot, std_bar, subject_line],
                   loc='lower center', ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.15))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)

        # Define the save path
        save_dir = op.join(op.dirname(data_path), '20250515_data_outcome_level_preprocessed_ada-prob_with_predictions_subjectwise_models_comparison')
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        save_path = op.join(save_dir, f'20250515_data_outcome_level_preprocessed_ada-prob_with_predictions_{model1}_vs_{model2}_subjectwise_comparison_{metric}.png')

        # Save the figure
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

        plt.show()
        plt.close()

# Define the specific pairs you want to compare
specific_pairs = [
    ('subject_HMM', model) for model in models if model != 'subject_HMM'
] + [
    ('group_HMM', model) for model in models if model.startswith('group_RNN_')
] + [
    ('group_HMM', model) for model in models if model.startswith('group_GRU_')
] + [
    ('big_group_HMM', model) for model in models if model.startswith('big_group_RNN_')
] + [
    ('big_group_HMM', model) for model in models if model.startswith('big_group_GRU_')
]

# Iterate over the specific pairs and plot comparisons
for model1, model2 in specific_pairs:
    print(f"Plotting comparison between {model1} and {model2}")
    # plot_subjectwise_comparison(scores_df, model1, model2)

#%%
### 5 - Compare the first prediction of each sequence for each model, conditioning
### or without conditionning on outcome == 0 (yellow) or outcome == 1 (blue)
data = pd.read_csv(data_path)

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
### 6 - Compute linear regression between hidden_parameter and models,
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
        '.csv', f'_regression_vs_{reference}_results.csv'
    ),
    index=False
)

# # Plotting (sampled, to avoid memory issues)
# sample_size = 200 
# n_cols = 2
# n_rows = int(np.ceil(len(models) / n_cols))
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7*n_rows))  # Make figure larger for square axes
# axes = axes.flatten()

# for idx, model in enumerate(models):
#     ax = axes[idx]
#     # Plot model vs reference as before
#     if model == reference:
#         df = data[[reference]]
#         if len(df) > sample_size:
#             df_sample = df.sample(n=sample_size, random_state=random_state)
#         else:
#             df_sample = df
#         ax.scatter(df_sample[reference], df_sample[reference], alpha=0.5)
#         ax.plot(df_sample[reference], df_sample[reference], color='red', label='y=x')
#         ax.set_title(f'{reference} vs {reference}\n$y = x$')
#         ax.set_xlabel(reference)
#         ax.set_ylabel(reference)
#         ax.set_aspect('equal', adjustable='datalim')
#         # Make axes square
#         lims = [
#             np.min([ax.get_xlim(), ax.get_ylim()]),
#             np.max([ax.get_xlim(), ax.get_ylim()]),
#         ]
#         ax.set_xlim(lims)
#         ax.set_ylim(lims)
#         continue
#     df = data[[reference, model]]
#     if len(df) > sample_size:
#         df_sample = df.sample(n=sample_size, random_state=random_state)
#     else:
#         df_sample = df
#     reg_row = regression_df[regression_df['model'] == model]
#     slope = reg_row['slope'].values[0]
#     intercept = reg_row['intercept'].values[0]
#     sns.regplot(
#         data=df_sample,
#         x=reference,
#         y=model,
#         scatter=True,
#         line_kws={'color': 'red'},
#         ax=ax
#     )
#     ax.set_title(f'{model} vs {reference}\n$y = {slope:.3f}x + {intercept:.3f}$')
#     ax.set_xlabel(reference)
#     ax.set_ylabel(model)
#     ax.set_aspect('equal', adjustable='datalim')
#     # Make axes square
#     lims = [
#         np.min([ax.get_xlim(), ax.get_ylim()]),
#         np.max([ax.get_xlim(), ax.get_ylim()]),
#     ]
#     ax.set_xlim(lims)
#     ax.set_ylim(lims)

# # Hide unused subplots
# for j in range(idx + 1, len(axes)):
#     fig.delaxes(axes[j])

# plt.tight_layout()
# plt.show()
# plt.savefig(
#     data_path.replace('.csv', f'_regression_vs_{reference}_results.png'),
#     bbox_inches='tight'
# )

# Plotting
sample_size = 200
random_state = 42

# Iterate over pairs of models
for i in range(0, len(models), 2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))  # Create a figure with two subplots
    for j, model in enumerate(models[i:i+2]):
        ax = axes[j]
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
        else:
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
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    plt.tight_layout()
    plt.show()
    plt.savefig(
        data_path.replace('.csv', f'_regression_vs_{reference}_results_{i//2}.png'),
        bbox_inches='tight'
    )

#%%
### 7 - Compute linear regression to obtain weights attributed to hidden parameters before
### and after change points, for each model and relative position,
### similarly to Figure 5 (Chung and Meyniel, 2025)
data = pd.read_csv(data_path)

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

def plot_regression_coefficients(regression_by_position, models, display_models=None):
    """
    Plot regression coefficients 'a' (hidden_param_bef, dashed) and 'b' (hidden_param_aft, solid) vs relative position for each model.

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

    if len(display_models) <= 3:
        models_prefix = '_vs_'.join(display_models)
    else:
        models_prefix = '_vs_'.join(display_models[:3]) + '_and-so-on'

    # Utiliser models_prefix dans le chemin de fichier
    file_path = data_outcome_level_preprocessed_path.replace(
        '.csv', f'_ada-prob_with_predictions_figure_5_results_{models_prefix}.png'
    )

    plt.savefig(
        file_path,
        bbox_inches='tight'
    )

plot_regression_coefficients(regression_by_position, models, display_models=['estimate', 'subject_HMM', 'optimal_HMM', 'group_HMM', 'big_group_HMM'])
plot_regression_coefficients(regression_by_position, models, display_models= ['estimate', 'subject_HMM', 'subject_RNN_2048', 'subject_GRU_2048'])
plot_regression_coefficients(regression_by_position, models, display_models= ['estimate', 'group_HMM', 'group_RNN_2048', 'group_GRU_2048'])
plot_regression_coefficients(regression_by_position, models, display_models= ['estimate', 'big_group_HMM', 'big_group_RNN_1024', 'big_group_GRU_1024'])
plot_regression_coefficients(regression_by_position, models, display_models= ['estimate', 'big_group_HMM', 'big_group_RNN_512', 'big_group_GRU_512'])
plot_regression_coefficients(regression_by_position, models, display_models= ['estimate', 'big_group_HMM', 'big_group_RNN_32', 'big_group_GRU_32'])
plot_regression_coefficients(regression_by_position, models, display_models= ['estimate', 'subject_HMM', 'subject_HMM_with_FNN_32', 'subject_HMM_with_FNN_512', 'subject_HMM_with_FNN_1024'])
plot_regression_coefficients(regression_by_position, models, display_models= ['estimate', 'group_HMM', 'big_group_HMM_with_FNN_4', 'big_group_HMM_with_FNN_8', 'big_group_HMM_with_FNN_16'])
plot_regression_coefficients(regression_by_position, models, display_models= ['estimate', 'big_group_HMM', 'big_group_HMM_with_FNN_32', 'big_group_HMM_with_FNN_512', 'big_group_HMM_with_FNN_1024'])

# %%
