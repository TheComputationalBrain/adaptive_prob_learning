

import os
import sys
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import itertools
from operator import mul
from functools import reduce
from scipy.stats import dirichlet
from scipy.stats import norm
import scipy.io

from time import time
import scipy.optimize as so

import data_analysis_utils as da

from scipy.stats import ttest_rel
from models.HMM_fit import HMM
from scipy.optimize import curve_fit

#%% Figure 1

G_dat = da.load_data("Gallistel")
Khaw_dat = da.load_data("Khaw")
FM_dat = da.load_data("FM")


G_subj_n = 10
G_all_sess = list(range(10))

K_subj_n = 11
K_all_sess = list(range(10))

FM_subj_n = 96
FM_all_sess = list(range(15))


fig, axs = plt.subplots(2, 1, figsize=(13, 8)) 
plt.rcParams.update({
    "font.size": 16,        # General font size
    "axes.titlesize": 21,   # Title size
    "axes.labelsize": 19,   # X and Y label size
    "xtick.labelsize": 19,  # X-axis tick label size
    "ytick.labelsize": 19,  # Y-axis tick label size
    "legend.fontsize": 20   # Legend font size
})

# Plot 1
sub = 0
session = 3
hidden_p = G_dat['true_p'][sub][session]
outcome = G_dat['outcome'][sub][session]
sub_estimate = G_dat['sub_est'][sub][session]

axs[0].plot(hidden_p, linestyle="-", color="grey", label="Hidden probability",linewidth=3)
axs[0].plot(sub_estimate, linestyle="-", color="blue", label="subject's estimate",linewidth=3)
axs[0].scatter(range(len(outcome)), outcome, color="red", alpha=0.8, s=0.5)
axs[0].set_ylim([-0.05, 1.05])
axs[0].set_xlabel("Observations")  # smaller font size for x-axis label
axs[0].set_ylabel("Probability")  # smaller font size for y-axis label
axs[0].legend(loc="upper left", bbox_to_anchor=(0, 0.95), frameon=False)  # smaller font size for legend
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].set_title("Gallistel et al. (2014)")

# Plot 2
sub = 0
session = 0
hidden_p = FM_dat['true_p'][sub][session]
outcome = FM_dat['outcome'][sub][session]
sub_estimate = FM_dat['sub_est'][sub][session]

axs[1].plot(hidden_p, linestyle="-", color="grey",linewidth=3)
axs[1].plot(sub_estimate, linestyle="-", color="blue",linewidth=3)
axs[1].scatter(range(len(outcome)), outcome, color="red", alpha=0.8, s=0.5)
axs[1].set_ylim([-0.05, 1.05])
axs[1].set_xlabel("Observations")  # smaller font size for x-axis label
axs[1].set_ylabel("Probability")  # smaller font size for y-axis label
axs[1].legend(loc="upper left", bbox_to_anchor=(0, 0.95), frameon=False)  # smaller font size for legend
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].set_title("Foucault & Meyniel (2024)")

# Adjust layout for tight spacing
plt.tight_layout()



plt.savefig("figures/Figure1.png", dpi=300)

plt.show()


#%% Figure 3


mod_names = ["HMM", "HGF", "VKF", "Mixed_delta", "reduced_bayes", "reduced_bayes_lamda","changepoint", "PID", "p_hall","delta_rule"]


# Load all CSV files into a single DataFrame
data_list = []
for model in mod_names:
    file_path = f"results/mod_recovery/cv_test/{model}_cv_test_sum.csv"  
    df = pd.read_csv(file_path)
    df['generated_model'] = model  # Add a column to track the generating model
    data_list.append(df)

# Combine all data into one DataFrame
df_all = pd.concat(data_list, ignore_index=True)

# Find the best-fitting model for each simulation
best_fit = df_all.loc[df_all.groupby(["generated_model", "simu_idx"])["cv_test_sum"].idxmin()]

# Count how often each model was the best fit
best_fit_counts = best_fit.groupby("generated_model")["mod"].value_counts().unstack(fill_value=0)

# Reorder rows based on the desired order in `mod_names` for the 'generated_model' column
best_fit_counts = best_fit_counts.loc[mod_names]

# Reorder columns based on the desired order in `mod_names` for the 'mod' column
best_fit_counts = best_fit_counts[mod_names]

rename_dict = {
    "HMM": "HMM",
    "HGF": "HGF",
    "VKF": "VKF",
    "Mixed_delta": "Mixture of delta rules",
    "reduced_bayes": "Reduced Bayesian model",
    "reduced_bayes_lamda": "Reduced Bayesian model\n(with under-weighted likelihood)",
    "changepoint": "Change Point model",
    "PID": "PID",
    "p_hall": "Pearce-Hall model",
    "delta_rule": "Delta Rule"
}

# Rename the 'generated_model' and 'mod' based on the dictionary
best_fit_counts = best_fit_counts.rename(index=rename_dict, columns=rename_dict)

# Convert counts to proportions
best_fit_proportions = best_fit_counts.div(best_fit_counts.sum(axis=1), axis=0)


data_matrix = best_fit_proportions.values 

# Define figure and axis
fig, ax = plt.subplots(figsize=(17, 10))


# Increase font size for all elements
plt.rcParams.update({
    "font.size": 18,        # General font size
    "axes.titlesize": 24,   # Title size
    "axes.labelsize": 24,   # X and Y label size
    "xtick.labelsize": 23,  # X-axis tick label size
    "ytick.labelsize": 23,  # Y-axis tick label size
    "legend.fontsize": 22,  # Legend font size
})

# Plot using imshow
cax = ax.imshow(data_matrix, cmap="Blues", aspect="auto")

# Add colorbar
cbar = plt.colorbar(cax)
cbar.set_label("Proportion of Best Fit")
cbar.ax.yaxis.set_label_position('left')

# Extract the correct model order from the DataFrame
x_labels = best_fit_proportions.columns.tolist()  # Generating models (x-axis)
y_labels = best_fit_proportions.index.tolist()   # Fitted models (y-axis)

x_labels = ["1", "2", "3", "4", "5", "6" ,"7", "8", "9", "10"]  # Example of manual labels for x-axis
y_labels = ["HMM(1)", "HGF(2)", "VKF(3)", "Mixture of delta rules(4)",
            "Reduced Bayesian model(5)", "Reduced Bayesian model(6)\n(with under-weighted likelihood)",
            "Change Point model(7)", "PID(8)", "Pearce-Hall model(9)","Delta Rule(10)" ]  # Example of manual labels for y-axis


# Annotate with values
num_x = len(x_labels)  # Number of generating models
num_y = len(y_labels)  # Number of fitted models

for i in range(num_y):  # Fitted model (y-axis)
    for j in range(num_x):  # Generating model (x-axis)
        ax.text(j, i, f"{data_matrix[i, j]:.2f}", ha="center", va="center", color="black")

# Set tick labels using the correct order
ax.set_xticks(np.arange(num_x))
ax.set_yticks(np.arange(num_y))
ax.set_xticklabels(x_labels, rotation=0, ha="center")  
ax.set_yticklabels(y_labels)  

# Labels
ax.set_ylabel("Generative Model")
ax.set_xlabel("Fitted Model")
ax.set_title("Model Recovery Matrix")

# Show plot
plt.tight_layout()
plt.savefig("figures/Figure3.png", dpi=300)

plt.show()

#%% Figure 4




## get ideal HMM esitamte 

mod_names = ["HMM"]
G_dat['mod_est'] =  {key: [] for key in mod_names}
Khaw_dat['mod_est'] = {key: [] for key in mod_names}
FM_dat['mod_est']= {key: [] for key in mod_names}


for key in mod_names:
    
    for subji in range(G_subj_n):
        
        
        subj_model_estimate = []
    
        data = {} 
    
        data['outcomes'] = np.concatenate(G_dat['outcome'][subji])
        expID = 1
                
        if key == "HMM":
            # p_c = G_paras[key][subji][0]
            p_c = .005
            model_est = HMM(p_c,data['outcomes'],expID)
            
    
        n_trials_per_session = 1000  # Number of trials per session
        n_sessions = len(model_est) // n_trials_per_session  
    
    
        model_est_sessions = [model_est[i * n_trials_per_session:(i + 1) * n_trials_per_session] for i in range(n_sessions)]
    
        subj_model_estimate = model_est_sessions
    
    
        G_dat['mod_est'][key].append(np.array(subj_model_estimate))
       
    
    
    
    for subji in range(K_subj_n):
        
        subj_model_estimate = []
    
        data = {} 
    
        data['outcomes'] = np.concatenate(Khaw_dat['outcome'][subji])
        expID = 2
        
    
                
        if key == "HMM":
            # p_c = K_paras[key][subji][0]
            p_c = .005
            model_est = HMM(p_c,data['outcomes'],expID)
        
    
        n_trials_per_session = 999  # Number of trials per session
        n_sessions = len(model_est) // n_trials_per_session  
    
    
        model_est_sessions = [model_est[i * n_trials_per_session:(i + 1) * n_trials_per_session] for i in range(n_sessions)]
    
        subj_model_estimate = model_est_sessions
    
    
        Khaw_dat['mod_est'][key].append(np.array(subj_model_estimate))
    
      
    
    for subji in range(FM_subj_n):
        
        subj_model_estimate = []
    
        data = {} 
    
        data['outcomes'] = np.concatenate(FM_dat['outcome'][subji]) 
        expID = 3
    
      
        if key == "HMM":
            # p_c = ada_paras[key][subji][0]
            p_c = .05
            model_est = HMM(p_c,data['outcomes'],expID)
            
        
    
        n_trials_per_session = 75  # Number of trials per session
        n_sessions = len(model_est) // n_trials_per_session  
    
    
        model_est_sessions = [model_est[i * n_trials_per_session:(i + 1) * n_trials_per_session] for i in range(n_sessions)]
    
        subj_model_estimate = model_est_sessions
    
    
        FM_dat['mod_est'][key].append(np.array(subj_model_estimate))
        

## correlation 


## FM_dat

n_subjects = FM_subj_n

name_refvals = {
        "pearsonr": [0],
        "slope": [0, 1],
        "yval-at-0p5": [],
    }



subs_linregress_data_FM = {}

for name in name_refvals:
        subs_linregress_data_FM[name] = np.empty(n_subjects)

for subji in range(n_subjects): 
    
    sub_estimate = FM_dat["sub_est"][subji]
    model_estimate = FM_dat["mod_est"]["HMM"][subji]
    
    linreg_res = scipy.stats.linregress(  # linear regression between subj and model estimate
    sub_estimate.flatten(),
    model_estimate.flatten())
    
    subs_linregress_data_FM["pearsonr"][subji] = linreg_res.rvalue
    subs_linregress_data_FM["slope"][subji] = linreg_res.slope
    subs_linregress_data_FM["yval-at-0p5"][subji] = linreg_res.intercept + linreg_res.slope * 0.5


# Compute group-level statistics on subject vs model estimate


for name, refvals in name_refvals.items():
    subs_vals = subs_linregress_data_FM[name]
    stat_data = {
            f"Mean of {name}": np.mean(subs_vals),
            f"S.e.m. of {name}": scipy.stats.sem(subs_vals),
            f"Standard deviation of {name}": np.std(subs_vals),
            }


    for refval in refvals:
        testres = scipy.stats.ttest_1samp(subs_vals, refval)
        stat_data[f"t-statistic of t-test against {refval}"] = testres.statistic
        stat_data[f"p-value of t-test against {refval}"] = testres.pvalue

    stat_data["dof"] = len(subs_vals) - 1
    stat_data = pd.Series(stat_data)
    print(stat_data)


## prepare for plotting 


subs_data = {}
subs_data["estimate"] = [None for _ in range(n_subjects)]
subs_data["model_estimate"] = [None for _ in range(n_subjects)]
nbins = 8

for subji in range(n_subjects):

    subject_data_sub_est = FM_dat["sub_est"][subji]
    subject_data_mod_est = FM_dat["mod_est"]["HMM"][subji]
    # Initialize empty lists to store sequence data for this subject
    estimates = []
    model_estimates = []
    
    for sessi in range(15):
        # Append sequence estimates to the lists
        estimates.append(subject_data_sub_est[sessi])
        model_estimates.append(subject_data_mod_est[sessi])
        
    # Concatenate all sequences into a single array per subject
    subs_data["estimate"][subji] = estimates
    subs_data["model_estimate"][subji] = model_estimates
    
    
all_model_estimate = np.concatenate(subs_data["model_estimate"]).flatten()
_, bin_edges = pd.qcut(all_model_estimate, nbins, retbins=True)

def get_bin_mask(x, bin_edges, i_bin):
    n_bins = len(bin_edges) - 1
    if i_bin < (n_bins - 1):
        return ((x >= bin_edges[i_bin]) & (x < bin_edges[i_bin+1]))
    else:
        return ((x >= bin_edges[i_bin]) & (x <= bin_edges[i_bin+1]))


# Compute average per bin over trials for each subject
for key in ["estimate", "model_estimate"]:
    subs_data[f"{key}_avg_per_bin"] = np.empty((n_subjects, nbins))
    
for i_sub in range(n_subjects):
        for i_bin in range(nbins):
            bin_mask = get_bin_mask(subs_data["model_estimate"][i_sub],
                bin_edges, i_bin)
             
            for key in ["estimate", "model_estimate"]:
                subdat = np.array(subs_data[key][i_sub])
                subs_data[f"{key}_avg_per_bin"][i_sub, i_bin] = (
                    np.mean(subdat[bin_mask]))

        # Compute average over subjects
group_data_FM = {}
for key in ["estimate", "model_estimate"]:
    group_data_FM[f"{key}_avg_per_bin"] = np.mean(
            subs_data[f"{key}_avg_per_bin"], axis=0)
    group_data_FM[f"{key}_sem_per_bin"] = scipy.stats.sem(
        subs_data[f"{key}_avg_per_bin"], axis=0)
  
        



#### G_dat

n_subjects = G_subj_n

name_refvals = {
        "pearsonr": [0],
        "slope": [0, 1],
        "yval-at-0p5": [],
    }



subs_linregress_data_G = {}

for name in name_refvals:
        subs_linregress_data_G[name] = np.empty(n_subjects)

for subji in range(n_subjects): 
    
    sub_estimate = G_dat["sub_est"][subji]
    model_estimate = G_dat["mod_est"]["HMM"][subji]
    
    linreg_res = scipy.stats.linregress(  # linear regression between subj and model estimate
    sub_estimate.flatten(),
    model_estimate.flatten())
    
    subs_linregress_data_G["pearsonr"][subji] = linreg_res.rvalue
    subs_linregress_data_G["slope"][subji] = linreg_res.slope
    subs_linregress_data_G["yval-at-0p5"][subji] = linreg_res.intercept + linreg_res.slope * 0.5


# Compute group-level statistics on subject vs model estimate

for name, refvals in name_refvals.items():
    subs_vals = subs_linregress_data_G[name]
    stat_data = {
            f"Mean of {name}": np.mean(subs_vals),
            f"S.e.m. of {name}": scipy.stats.sem(subs_vals),
            f"Standard deviation of {name}": np.std(subs_vals),
            }


    for refval in refvals:
        testres = scipy.stats.ttest_1samp(subs_vals, refval)
        stat_data[f"t-statistic of t-test against {refval}"] = testres.statistic
        stat_data[f"p-value of t-test against {refval}"] = testres.pvalue

    stat_data["dof"] = len(subs_vals) - 1
    stat_data = pd.Series(stat_data)
    print(stat_data)


### prepare for plotting 


subs_data = {}
subs_data["estimate"] = [None for _ in range(n_subjects)]
subs_data["model_estimate"] = [None for _ in range(n_subjects)]
nbins = 8

for subji in range(n_subjects):

    subject_data_sub_est = G_dat["sub_est"][subji]
    subject_data_mod_est = G_dat["mod_est"]["HMM"][subji]
    # Initialize empty lists to store sequence data for this subject
    estimates = []
    model_estimates = []
    
    for sessi in range(10):
        # Append sequence estimates to the lists
        estimates.append(subject_data_sub_est[sessi])
        model_estimates.append(subject_data_mod_est[sessi])
        
    # Concatenate all sequences into a single array per subject
    subs_data["estimate"][subji] = estimates
    subs_data["model_estimate"][subji] = model_estimates
    
    
all_model_estimate = np.concatenate(subs_data["model_estimate"]).flatten()
_, bin_edges = pd.qcut(all_model_estimate, nbins, retbins=True)



# Compute average per bin over trials for each subject
for key in ["estimate", "model_estimate"]:
    subs_data[f"{key}_avg_per_bin"] = np.empty((n_subjects, nbins))
    
for i_sub in range(n_subjects):
        for i_bin in range(nbins):
            bin_mask = get_bin_mask(subs_data["model_estimate"][i_sub],
                bin_edges, i_bin)
             
            for key in ["estimate", "model_estimate"]:
                subdat = np.array(subs_data[key][i_sub])
                subs_data[f"{key}_avg_per_bin"][i_sub, i_bin] = (
                    np.mean(subdat[bin_mask]))

        # Compute average over subjects
group_data_G = {}
for key in ["estimate", "model_estimate"]:
    group_data_G[f"{key}_avg_per_bin"] = np.mean(
            subs_data[f"{key}_avg_per_bin"], axis=0)
    group_data_G[f"{key}_sem_per_bin"] = scipy.stats.sem(
        subs_data[f"{key}_avg_per_bin"], axis=0)
    
    
    
#### Khaw_dat

n_subjects = K_subj_n

name_refvals = {
        "pearsonr": [0],
        "slope": [0, 1],
        "yval-at-0p5": [],
    }



subs_linregress_data_Khaw = {}

for name in name_refvals:
        subs_linregress_data_Khaw[name] = np.empty(n_subjects)

for subji in range(n_subjects): 
    
    sub_estimate = Khaw_dat["sub_est"][subji]
    model_estimate = Khaw_dat["mod_est"]["HMM"][subji]
    
    linreg_res = scipy.stats.linregress(  # linear regression between subj and model estimate
    sub_estimate.flatten(),
    model_estimate.flatten())
    
    subs_linregress_data_Khaw["pearsonr"][subji] = linreg_res.rvalue
    subs_linregress_data_Khaw["slope"][subji] = linreg_res.slope
    subs_linregress_data_Khaw["yval-at-0p5"][subji] = linreg_res.intercept + linreg_res.slope * 0.5


# Compute group-level statistics on subject vs model estimate


for name, refvals in name_refvals.items():
    subs_vals = subs_linregress_data_Khaw[name]
    stat_data = {
            f"Mean of {name}": np.mean(subs_vals),
            f"S.e.m. of {name}": scipy.stats.sem(subs_vals),
            f"Standard deviation of {name}": np.std(subs_vals),
            }


    for refval in refvals:
        testres = scipy.stats.ttest_1samp(subs_vals, refval)
        stat_data[f"t-statistic of t-test against {refval}"] = testres.statistic
        stat_data[f"p-value of t-test against {refval}"] = testres.pvalue

    stat_data["dof"] = len(subs_vals) - 1
    stat_data = pd.Series(stat_data)
    print(stat_data)


### prepare for plotting 


subs_data = {}
subs_data["estimate"] = [None for _ in range(n_subjects)]
subs_data["model_estimate"] = [None for _ in range(n_subjects)]
nbins = 8

for subji in range(n_subjects):

    subject_data_sub_est = Khaw_dat["sub_est"][subji]
    subject_data_mod_est = Khaw_dat["mod_est"]["HMM"][subji]
    # Initialize empty lists to store sequence data for this subject
    estimates = []
    model_estimates = []
    
    for sessi in range(10):
        # Append sequence estimates to the lists
        estimates.append(subject_data_sub_est[sessi])
        model_estimates.append(subject_data_mod_est[sessi])
        
    # Concatenate all sequences into a single array per subject
    subs_data["estimate"][subji] = estimates
    subs_data["model_estimate"][subji] = model_estimates
    
    
all_model_estimate = np.concatenate(subs_data["model_estimate"]).flatten()
_, bin_edges = pd.qcut(all_model_estimate, nbins, retbins=True)



# Compute average per bin over trials for each subject
for key in ["estimate", "model_estimate"]:
    subs_data[f"{key}_avg_per_bin"] = np.empty((n_subjects, nbins))
    
for i_sub in range(n_subjects):
        for i_bin in range(nbins):
            bin_mask = get_bin_mask(subs_data["model_estimate"][i_sub],
                bin_edges, i_bin)
             
            for key in ["estimate", "model_estimate"]:
                subdat = np.array(subs_data[key][i_sub])
                subs_data[f"{key}_avg_per_bin"][i_sub, i_bin] = (
                    np.mean(subdat[bin_mask]))

        # Compute average over subjects
group_data_Khaw = {}
for key in ["estimate", "model_estimate"]:
    group_data_Khaw[f"{key}_avg_per_bin"] = np.mean(
            subs_data[f"{key}_avg_per_bin"], axis=0)
    group_data_Khaw[f"{key}_sem_per_bin"] = scipy.stats.sem(
        subs_data[f"{key}_avg_per_bin"], axis=0)




## prob. weight function 

## FM dat



num_subjects = len(FM_dat["sub_est"])

p_values = FM_dat["mod_est"]["HMM"] # ideal observer probabilities
p_values  = [subject_data.flatten() for subject_data in FM_dat["mod_est"]["HMM"]]



w_values_subjects = FM_dat["sub_est"]
w_values_subjects = [subject_data.flatten() for subject_data in FM_dat["sub_est"]]


# Store fitted parameters
delta_values = []
gamma_values = []


# Loop through each subject
for subji in range(num_subjects):
    
    # Apply log-odds transformation
    p_values_sub = p_values[subji]
    w_values = w_values_subjects[subji]

    try:
        # Fit the model
        initial_guess = [1.0, 1.0]
        params, _  = curve_fit(da.llo_function, p_values_sub, w_values, p0=initial_guess, bounds=([0, 0], [np.inf, np.inf]))
        delta_values.append(params[0])  # Convert log(δ) back to δ
        gamma_values.append(params[1])
    except RuntimeError:
        print("Fit failed for one subject")



# Compute median values
delta_median_FM = np.median(delta_values)
gamma_median_FM = np.median(gamma_values)


## G dat

num_subjects = len(G_dat["sub_est"])

p_values = G_dat["mod_est"]["HMM"] # ideal observer probabilities
p_values  = [subject_data.flatten() for subject_data in G_dat["mod_est"]["HMM"]]



w_values_subjects = G_dat["sub_est"]
w_values_subjects = [subject_data.flatten() for subject_data in G_dat["sub_est"]]


# Store fitted parameters
delta_values = []
gamma_values = []


# Loop through each subject
for subji in range(num_subjects):
    
    # Apply log-odds transformation
    p_values_sub = p_values[subji]
    w_values = w_values_subjects[subji]

    try:
        # Fit the model
        initial_guess = [1.0, 1.0]
        params, _  = curve_fit(da.llo_function, p_values_sub, w_values, p0=initial_guess, bounds=([0, 0], [np.inf, np.inf]))
        delta_values.append(params[0])  # Convert log(δ) back to δ
        gamma_values.append(params[1])
    except RuntimeError:
        print("Fit failed for one subject")



# Compute median values
delta_median_G = np.median(delta_values)
gamma_median_G = np.median(gamma_values)


## Khaw dat


# Simulated data (replace with actual data)
num_subjects = len(Khaw_dat["sub_est"])

p_values = Khaw_dat["mod_est"]["HMM"] # ideal observer probabilities
p_values  = [subject_data.flatten() for subject_data in Khaw_dat["mod_est"]["HMM"]]



# Generate simulated subject estimates (replace with real data)
w_values_subjects = Khaw_dat["sub_est"]
w_values_subjects = [subject_data.flatten() for subject_data in Khaw_dat["sub_est"]]


# Store fitted parameters
delta_values = []
gamma_values = []



# Loop through each subject
for subji in range(num_subjects):
    
    # Apply log-odds transformation
    p_values_sub = p_values[subji]
    w_values = w_values_subjects[subji]

    try:
        # Fit the model
        initial_guess = [1.0, 1.0]
        params, _  = curve_fit(da.llo_function, p_values_sub, w_values, p0=initial_guess, bounds=([0, 0], [np.inf, np.inf]))
        delta_values.append(params[0])  # Convert log(δ) back to δ
        gamma_values.append(params[1])
    except RuntimeError:
        print("Fit failed for one subject")



# Compute median values
delta_median_Khaw = np.median(delta_values)
gamma_median_Khaw = np.median(gamma_values)


## plot


# Data and titles for first row
datasets = [
    ("Foucault & Meyniel (2024)", group_data_FM, subs_linregress_data_FM),
    ("Gallistel et al. (2014)", group_data_G, subs_linregress_data_G),
    ("Khaw et al. (2017)", group_data_Khaw, subs_linregress_data_Khaw),
]

# Store blue dot data for overlaying in the second row
blue_dots_x = []
blue_dots_y = []
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

# First row: Scatter plots with error bars
for ax, (title, group_data, subs_linregress_data) in zip(axes[0], datasets):
    x_vals = []
    y_vals = []

    for i_bin in range(nbins):
        x = group_data["model_estimate_avg_per_bin"][i_bin]
        y = group_data["estimate_avg_per_bin"][i_bin]
        err = group_data["estimate_sem_per_bin"][i_bin]
        
        ax.errorbar(x, y, err, fmt='o', ms=8, color='blue', ecolor='grey')
        
        x_vals.append(x)
        y_vals.append(y)

    # Store for second row
    blue_dots_x.append(x_vals)
    blue_dots_y.append(y_vals)





# Set global font size
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})

# Create a single row of three columns (for second-row plots)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Titles for each model
titles = [
    "Foucault & Meyniel (2024)",
    "Gallistel et al. (2014)",
    "Khaw et al. (2017)"
]

# Data for LLO function plots
p_fit = np.linspace(0.01, 0.99, 100)
weighting_data = [
    (delta_median_FM, gamma_median_FM, axes[0], blue_dots_x[0], blue_dots_y[0], subs_linregress_data_FM, titles[0]),
    (delta_median_G, gamma_median_G, axes[1], blue_dots_x[1], blue_dots_y[1], subs_linregress_data_G, titles[1]),
    (delta_median_Khaw, gamma_median_Khaw, axes[2], blue_dots_x[2], blue_dots_y[2], subs_linregress_data_Khaw, titles[2])
]

for delta, gamma, ax, x_scatter, y_scatter, subs_linregress_data, title in weighting_data:
    # Plot LLO function
    w_fit = da.llo_function(p_fit, delta, gamma)
    # Overlay blue scatter points
    ax.scatter(x_scatter, y_scatter, color="blue", s=50, alpha=0.7,  zorder=3)
    ax.plot(p_fit, w_fit, color="orange", lw=2, label=f"δ={delta:.2f}, γ={gamma:.2f}", zorder=2)
    ax.plot(p_fit, p_fit, color="k", linestyle="--", lw=1,zorder=1)

   

    # Compute and display statistics
    pearsonr_mean = round(np.mean(subs_linregress_data["pearsonr"]), 2)
    pearsonr_std = round(np.std(subs_linregress_data["pearsonr"]), 2)
    slope_mean = round(np.mean(subs_linregress_data["slope"]), 2)
    slope_std = round(np.std(subs_linregress_data["slope"]), 2)

    stats_text = (f"Pearson's r = {pearsonr_mean} (SD = {pearsonr_std})\n"
                  f"Slope β = {slope_mean} (SD = {slope_std})")

    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.1, edgecolor='none'),
            fontsize=16)  # Stats text

    # Labels and aesthetics
    ax.set_xlabel("Ideal Observer Probability", fontsize=18)
    ax.set_ylabel("Subjective Probability", fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add the title above each plot
    ax.set_title(title, fontsize=20)

# Adjust layout
plt.tight_layout()
plt.savefig("figures/Figure4.png", dpi=300)
plt.show()
            
#%% Figure 5

## pairwise comparsions


# Define datasets and model names
datasets = {
    "G": "results/cv_MSE/G_cv_test_sum_{}.csv",
    "K": "results/cv_MSE/K_cv_test_sum_{}.csv",
    "FM": "results/cv_MSE/FM_cv_test_sum_{}.csv",
}

mod_names = ["HMM", "changepoint", "delta_rule", "p_hall", "reduced_bayes",
             "reduced_bayes_lamda", "PID", "Mixed_delta", "VKF", "HGF"]

# Dictionary to store reloaded cv_test_sum values
cv_test_sum_loaded = {dataset: {} for dataset in datasets}

# Load cv_test_sum from CSV files
for dataset, path_template in datasets.items():
    for model in mod_names:
        filename = path_template.format(model)
        
        # Load CSV file
        df = pd.read_csv(filename)
        
        # Extract values as a list
        cv_test_sum_loaded[dataset][model] = df[model].tolist()

 


## FM_dat

FM_mean_per_model = cv_test_sum_loaded["FM"]

# Initialize empty DataFrames for t-statistics and p-values
FM_t_stat_matrix = pd.DataFrame(np.nan, index=mod_names, columns=mod_names)
FM_p_value_matrix = pd.DataFrame(np.nan, index=mod_names, columns=mod_names)

# Loop over all unique model pairs
for model1, model2 in itertools.combinations(mod_names, 2):
    t_stat, p_value = ttest_rel(FM_mean_per_model[model1], FM_mean_per_model[model2])
    
    # Store results symmetrically in the matrices
    FM_t_stat_matrix.loc[model1, model2] = t_stat
    FM_t_stat_matrix.loc[model2, model1] = t_stat  # Mirror value

    FM_p_value_matrix.loc[model1, model2] = p_value
    FM_p_value_matrix.loc[model2, model1] = p_value  # Mirror value

# Fill diagonal with zeros (self-comparison)
np.fill_diagonal(FM_t_stat_matrix.values, 0)
np.fill_diagonal(FM_p_value_matrix.values, 1)  



# Initialize an empty DataFrame for mean differences
FM_mean_diff_matrix = pd.DataFrame(np.nan, index=mod_names, columns=mod_names)

# Loop over all unique model pairs
for model1, model2 in itertools.combinations(mod_names, 2):
    mean_diff =  np.mean(np.array(FM_mean_per_model[model1]) - np.array(FM_mean_per_model[model2]))

    # Store results symmetrically in the matrix
    FM_mean_diff_matrix.loc[model1, model2] = mean_diff
    FM_mean_diff_matrix.loc[model2, model1] = -mean_diff  # Mirror value with sign flipped

# Fill diagonal with zeros (self-comparison)
np.fill_diagonal(FM_mean_diff_matrix.values, 0)



custom_order = ["HMM", "HGF", "VKF", "Mixed_delta", "reduced_bayes", "reduced_bayes_lamda","changepoint", "PID", "p_hall","delta_rule"]


# Reorder the DataFrames
FM_t_stat_matrix = FM_t_stat_matrix.reindex(index=custom_order, columns=custom_order)
FM_p_value_matrix = FM_p_value_matrix.reindex(index=custom_order, columns=custom_order)
FM_mean_diff_matrix = FM_mean_diff_matrix .reindex(index=custom_order, columns=custom_order)




## G_dat

G_mean_per_model = cv_test_sum_loaded["G"]


# Initialize empty DataFrames for t-statistics and p-values
G_t_stat_matrix = pd.DataFrame(np.nan, index=mod_names, columns=mod_names)
G_p_value_matrix = pd.DataFrame(np.nan, index=mod_names, columns=mod_names)

# Loop over all unique model pairs
for model1, model2 in itertools.combinations(mod_names, 2):
    t_stat, p_value = ttest_rel(G_mean_per_model[model1], G_mean_per_model[model2])
    
    # Store results symmetrically in the matrices
    G_t_stat_matrix.loc[model1, model2] = t_stat
    G_t_stat_matrix.loc[model2, model1] = t_stat  # Mirror value

    G_p_value_matrix.loc[model1, model2] = p_value
    G_p_value_matrix.loc[model2, model1] = p_value  # Mirror value

# Fill diagonal with zeros (self-comparison)
np.fill_diagonal(G_t_stat_matrix.values, 0)
np.fill_diagonal(G_p_value_matrix.values, 1)  # p-value = 1 for same models




# Reorder the DataFrames
G_t_stat_matrix = G_t_stat_matrix.reindex(index=custom_order, columns=custom_order)
G_p_value_matrix = G_p_value_matrix.reindex(index=custom_order, columns=custom_order)




# Initialize an empty DataFrame for mean differences
G_mean_diff_matrix = pd.DataFrame(np.nan, index=mod_names, columns=mod_names)

# Loop over all unique model pairs
for model1, model2 in itertools.combinations(mod_names, 2):
    mean_diff =  np.mean(np.array(G_mean_per_model[model1]) - np.array(G_mean_per_model[model2]))

    # Store results symmetrically in the matrix
    G_mean_diff_matrix.loc[model1, model2] = mean_diff
    G_mean_diff_matrix.loc[model2, model1] = -mean_diff  # Mirror value with sign flipped

# Fill diagonal with zeros (self-comparison)
np.fill_diagonal(G_mean_diff_matrix.values, 0)


G_mean_diff_matrix = G_mean_diff_matrix .reindex(index=custom_order, columns=custom_order)



## K_dat

K_mean_per_model = cv_test_sum_loaded["K"]

# Initialize empty DataFrames for t-statistics and p-values
K_t_stat_matrix = pd.DataFrame(np.nan, index=mod_names, columns=mod_names)
K_p_value_matrix = pd.DataFrame(np.nan, index=mod_names, columns=mod_names)

# Loop over all unique model pairs
for model1, model2 in itertools.combinations(mod_names, 2):
    t_stat, p_value = ttest_rel(K_mean_per_model[model1], K_mean_per_model[model2])
    
    # Store results symmetrically in the matrices
    K_t_stat_matrix.loc[model1, model2] = t_stat
    K_t_stat_matrix.loc[model2, model1] = t_stat  # Mirror value

    K_p_value_matrix.loc[model1, model2] = p_value
    K_p_value_matrix.loc[model2, model1] = p_value  # Mirror value

# Fill diagonal with zeros (self-comparison)
np.fill_diagonal(K_t_stat_matrix.values, 0)
np.fill_diagonal(K_p_value_matrix.values, 1)  # p-value = 1 for same models



# Reorder the DataFrames
K_t_stat_matrix = K_t_stat_matrix.reindex(index=custom_order, columns=custom_order)
K_p_value_matrix = K_p_value_matrix.reindex(index=custom_order, columns=custom_order)


# Initialize an empty DataFrame for mean differences
K_mean_diff_matrix = pd.DataFrame(np.nan, index=mod_names, columns=mod_names)

# Loop over all unique model pairs
for model1, model2 in itertools.combinations(mod_names, 2):
    mean_diff =  np.mean(np.array(K_mean_per_model[model1]) - np.array(K_mean_per_model[model2]))

    # Store results symmetrically in the matrix
    K_mean_diff_matrix.loc[model1, model2] = mean_diff
    K_mean_diff_matrix.loc[model2, model1] = -mean_diff  # Mirror value with sign flipped

# Fill diagonal with zeros (self-comparison)
np.fill_diagonal(K_mean_diff_matrix.values, 0)

K_mean_diff_matrix = K_mean_diff_matrix .reindex(index=custom_order, columns=custom_order)


## plot


# Function to determine significance level
def get_significance_symbol(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""  # No annotation if p-value is greater than 0.05

# Function to get best-fitting models
def find_best_model(data_dict):
    best_models = []
    subjects_count = len(next(iter(data_dict.values())))  # Get number of subjects
    models = list(data_dict.keys())

    for i in range(subjects_count):
        subject_values = np.array([data_dict[model][i] for model in models])
        best_model = models[np.argmin(subject_values)]
        best_models.append(best_model)

    return best_models

def compute_percentage(best_model_list, total_subjects):
    model_counts = {model: best_model_list.count(model) for model in custom_order}
    return {model: (count / total_subjects) * 100 for model, count in model_counts.items()}

# Find best model for each subject in the three datasets
best_model_ada = find_best_model(FM_mean_per_model)  # 96 subjects
best_model_G = find_best_model(G_mean_per_model)      # 10 subjects
best_model_K = find_best_model(K_mean_per_model)      # 11 subjects



# Define figure and gridspec for 3 rows
fig = plt.figure(figsize=(21, 22))

gs = fig.add_gridspec(3, 3, height_ratios=[0.13, 0.12, 0.2])  # First row larger for matrices
gs.update(hspace=0.35) 
# Define vmin and vmax fo6 color scaling
vmin, vmax = -0.37, 0.37

# List of data matrices and titles
data_matrices = [FM_mean_diff_matrix, G_mean_diff_matrix, K_mean_diff_matrix]
p_matrices = [FM_p_value_matrix, G_p_value_matrix, K_p_value_matrix]
titles = ["Foucault & Meyniel (2024)", "Gallistel et al. (2014)", "Khaw et al. (2017)"]

# First row: P-value matrices
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

for i, (data_matrix, p_matrix, title) in enumerate(zip(data_matrices, p_matrices,titles)):
    ax = axes[i]
    
    data_matrix = data_matrix.values.copy()
    # mask = np.triu(np.ones_like(data_matrix, dtype=bool), k=1)
    # data_matrix[mask] = np.nan
    
    diagonal_indices = np.diag_indices_from(data_matrix)
    data_matrix[diagonal_indices] = np.nan
    
    # Make diagonal black
    data_matrix[diagonal_indices] = -1  # Set diagonal values to -1, so they appear black
    
    
    # cmap = plt.cm.RdBu.reversed()
    cmap = plt.cm.coolwarm
    cmap.set_under("black")  # Set the color for values below vmin
    
    cax = ax.imshow(data_matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    
    for j in range(len(p_matrix)):
        for k in range(len(p_matrix)):  # Loop over full matrix
            if j != k:  # Exclude diagonal
                symbol = get_significance_symbol(p_matrix.iloc[j, k])
                ax.text(k, j, symbol, ha="center", va="center", color="black", fontsize=16, fontweight="bold")
    
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], rotation=0, ha="center")

    if i == 0:
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels(["HMM(1)", "HGF(2)", "VKF(3)", "Mixture of delta rules(4)",
                            "Reduced Bayesian model(5)", "Reduced Bayesian model(6)\n(with under-weighted likelihood)",
                            "Change Point model(7)", "PID(8)", "Pearce-Hall model(9)", "Delta Rule(10)"],fontsize=16)
    else:
        ax.set_yticklabels([])

    ax.set_title(title)
    ax.set_xlabel("Reference Models")

# Add colorbar next to last plot
cbar_ax = fig.add_axes([0.92, 0.7, 0.015, 0.15])
cbar = plt.colorbar(cax, cax=cbar_ax, ticks=[-0.3, 0, 0.3])
cbar.set_label("mean difference",fontsize=17)
cbar.ax.yaxis.set_label_position('left')
cbar.set_ticklabels(["-0.3", "0", "0.3"])

# Second and third rows: Bar plots
ax1 = fig.add_subplot(gs[1, :])
# ax2 = fig.add_subplot(gs[2, :])
plt.subplots_adjust(hspace=0.1)  # Reduce vertical space between rows
# Define custom order and labels
custom_order = ["HMM", "HGF", "VKF", "Mixed_delta", "reduced_bayes", "reduced_bayes_lamda",
                "changepoint", "PID", "p_hall", "delta_rule"]
custom_x_labels = ["HMM", "HGF", "VKF", "Mixture of delta rules",
                   "Reduced Bayesian model", "Reduced Bayesian model\n(with under-weighted likelihood)",
                   "Change Point model", "PID", "Pearce-Hall model", "Delta Rule"]

models = custom_order

ada_percentage = compute_percentage(best_model_ada, 96)
G_percentage = compute_percentage(best_model_G, 10)
K_percentage = compute_percentage(best_model_K, 11)



ada_values = [ada_percentage[m] for m in models]
G_values = [G_percentage[m] for m in models]
K_values = [K_percentage[m] for m in models]

# Set up bar positions
bar_width = 0.3
group_spacing = 0.5
x = np.arange(len(models)) * (3 * bar_width + group_spacing)



ax1.bar(x - bar_width, ada_values, width=bar_width, label='Foucault & Meyniel (2024)', color='tab:blue', alpha=0.6)
ax1.bar(x, G_values, width=bar_width, label='Gallistel et al. (2014)', color='tab:orange', alpha=0.6)
ax1.bar(x + bar_width, K_values, width=bar_width, label='Khaw et al. (2017)', color='tab:green', alpha=0.6)
ax1.set_ylabel('Percentage of Subjects (%)')
ax1.set_title('Best-Fitting Model')
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.yaxis.grid(True, linestyle="--", alpha=0.7)
ax1.legend()

# Shared x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(custom_x_labels, rotation=45, ha="right")

plt.tight_layout()
plt.savefig("figures/Figure5.png", dpi=300, bbox_inches='tight')
plt.show()


       
