

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


from models.Gallistel_2014 import ChangePointModel 

from models.delta_learning import delta_rule,pearce_hall

from models.Nassar_et_al_2010 import reduced_bayes_lamda,reduced_bayes
from models.PID import PID
from models.Mixed_delta import mixed_delta
from models.VKF import VKF
from models.HGF import HGF


from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import gc

from scipy import stats
import scipy.stats as sps

#%% load data

G_dat = da.load_data("Gallistel")
Khaw_dat = da.load_data("Khaw")
FM_dat = da.load_data("FM")


G_subj_n = len(G_dat["sub_est"])
G_all_sess = list(range(len(G_dat["sub_est"][0])))

K_subj_n = len(Khaw_dat["sub_est"])
K_all_sess = list(range(len(Khaw_dat["sub_est"][0])))

FM_subj_n = len(FM_dat["sub_est"])
FM_all_sess = list(range(len(FM_dat["sub_est"][0])))






#%% Figure 1


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

mod_names = ["HMM_ideal"]

G_dat['mod_est'] = {}
Khaw_dat['mod_est'] = {}
FM_dat['mod_est'] = {}

for key in mod_names:
    
    G_dat['mod_est'][key] = []
    Khaw_dat['mod_est'][key] = []
    FM_dat['mod_est'][key] = []
    
    
    for subji in range(G_subj_n):
        
        
        subj_model_estimate = []
    
        data = {} 
    
        data['outcomes'] = np.concatenate(G_dat['outcome'][subji])
        expID = 1
                
        if key == "HMM_ideal":
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
        
    
                
        if key == "HMM_ideal":
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
    
      
        if key == "HMM_ideal":
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
    model_estimate = FM_dat["mod_est"]["HMM_ideal"][subji]
    
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
    subject_data_mod_est = FM_dat["mod_est"]["HMM_ideal"][subji]
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
    model_estimate = G_dat["mod_est"]["HMM_ideal"][subji]
    
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
    subject_data_mod_est = G_dat["mod_est"]["HMM_ideal"][subji]
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
    model_estimate = Khaw_dat["mod_est"]["HMM_ideal"][subji]
    
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
    subject_data_mod_est = Khaw_dat["mod_est"]["HMM_ideal"][subji]
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

p_values = FM_dat["mod_est"]["HMM_ideal"] # ideal observer probabilities
p_values  = [subject_data.flatten() for subject_data in FM_dat["mod_est"]["HMM_ideal"]]



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

p_values = G_dat["mod_est"]["HMM_ideal"] # ideal observer probabilities
p_values  = [subject_data.flatten() for subject_data in G_dat["mod_est"]["HMM_ideal"]]



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

p_values = Khaw_dat["mod_est"]["HMM_ideal"] # ideal observer probabilities
p_values  = [subject_data.flatten() for subject_data in Khaw_dat["mod_est"]["HMM_ideal"]]



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

mod_names = ["HMM", "changepoint", "delta_rule", "p_hall","reduced_bayes",
                  "reduced_bayes_lamda", "PID" , "Mixed_delta", "VKF", "HGF"]
       
         

G_dat['mod_est'] =  {key: [] for key in mod_names}
Khaw_dat['mod_est'] = {key: [] for key in mod_names}
FM_dat['mod_est']= {key: [] for key in mod_names}

# import the csv files 

file_names = ["FM_opt_results.csv","G_opt_results.csv","K_opt_results.csv"]


for filename in file_names:
    file_path = os.path.join("results/optimisation", filename)
    if os.path.isfile(file_path):  # Check if the file exists
        # Read each specified CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        if filename == "FM_opt_results.csv":
        # Convert DataFrame to the format of the original data_sets dictionary
            FM_paras = {
                'HMM': [[x] for x in df['HMM_pc']],  # Wraps each entry in a list
                'changepoint': [list(t) for t in zip(df['changepoint_T1'], df['changepoint_T2'])],  # Converts tuples to lists
                'delta_rule': [[x] for x in df['delta_lr']],
                'p_hall': [list(t) for t in zip(df['p_hall_lr'], df['p_hall_w'])],
                'reduced_bayes': [[x] for x in df['reduced_bayes_pc']],
                'reduced_bayes_lamda': [list(t) for t in zip(df['reduced_bayes_lamda_pc'], df['reduced_bayes_lamda'])],
                'PID': [list(t) for t in zip(df['PID_kp'], df['PID_ki'], df['PID_kd'], df['PID_lamda'])],
                'Mixed_delta': [list(t) for t in zip(df['mixed_delta_delta1'], df['mixed_delta_delta2'], df['mixed_delta_hrate'], df['mixed_delta_nu_p'])],
                'VKF': [list(t) for t in zip(df['VKF_lamda'], df['VKF_omega'], df['VKF_v0'])],
                'HGF': [list(t) for t in zip(df['HGF_nu'], df['HGF_kappa'], df['HGF_omega'])]
            }
            
        elif filename == "G_opt_results.csv":
            G_paras = {
                'HMM': [[x] for x in df['HMM_pc']],  # Wraps each entry in a list
                'changepoint': [list(t) for t in zip(df['changepoint_T1'], df['changepoint_T2'])],  # Converts tuples to lists
                'delta_rule': [[x] for x in df['delta_lr']],
                'p_hall': [list(t) for t in zip(df['p_hall_lr'], df['p_hall_w'])],
                'reduced_bayes': [[x] for x in df['reduced_bayes_pc']],
                'reduced_bayes_lamda': [list(t) for t in zip(df['reduced_bayes_lamda_pc'], df['reduced_bayes_lamda'])],
                'PID': [list(t) for t in zip(df['PID_kp'], df['PID_ki'], df['PID_kd'], df['PID_lamda'])],
                'Mixed_delta': [list(t) for t in zip(df['mixed_delta_delta1'], df['mixed_delta_delta2'], df['mixed_delta_hrate'], df['mixed_delta_nu_p'])],
                'VKF': [list(t) for t in zip(df['VKF_lamda'], df['VKF_omega'], df['VKF_v0'])],
                'HGF': [list(t) for t in zip(df['HGF_nu'], df['HGF_kappa'], df['HGF_omega'])]
            }
            
        elif filename == "K_opt_results.csv":
            K_paras = {
                'HMM': [[x] for x in df['HMM_pc']],  # Wraps each entry in a list
                'changepoint': [list(t) for t in zip(df['changepoint_T1'], df['changepoint_T2'])],  # Converts tuples to lists
                'delta_rule': [[x] for x in df['delta_lr']],
                'p_hall': [list(t) for t in zip(df['p_hall_lr'], df['p_hall_w'])],
                'reduced_bayes': [[x] for x in df['reduced_bayes_pc']],
                'reduced_bayes_lamda': [list(t) for t in zip(df['reduced_bayes_lamda_pc'], df['reduced_bayes_lamda'])],
                'PID': [list(t) for t in zip(df['PID_kp'], df['PID_ki'], df['PID_kd'], df['PID_lamda'])],
                'Mixed_delta': [list(t) for t in zip(df['mixed_delta_delta1'], df['mixed_delta_delta2'], df['mixed_delta_hrate'], df['mixed_delta_nu_p'])],
                'VKF': [list(t) for t in zip(df['VKF_lamda'], df['VKF_omega'], df['VKF_v0'])],
                'HGF': [list(t) for t in zip(df['HGF_nu'], df['HGF_kappa'], df['HGF_omega'])]
            }



for key in mod_names:
    
    G_dat['mod_est'][key] = []
    Khaw_dat['mod_est'][key] = []
    FM_dat['mod_est'][key] = []

    for subji in range(G_subj_n):
        
        
        
        subj_model_estimate = []
    
        data = {} 
    
        data['outcomes'] = np.concatenate(G_dat['outcome'][subji])
        expID = 1
        
        if key == "changepoint":
            T1 = G_paras[key][subji][0]
            T2 = G_paras[key][subji][1]
            model_est = ChangePointModel(T1,T2,data['outcomes'],expID)
                
        elif key == "HMM":
            p_c = G_paras[key][subji][0]
            model_est = HMM(p_c,data['outcomes'],expID)
            
        elif key == "delta_rule":
            lr = G_paras[key][subji][0]
            model_est = delta_rule(lr, data['outcomes'],expID)
            
        elif key == "p_hall":
            lr = G_paras[key][subji][0]
            weight = G_paras[key][subji][1]
            model_est = pearce_hall(lr,weight,data['outcomes'],expID)
            
        elif key == "reduced_bayes":
            p_c = G_paras[key][subji][0]
            model_est = reduced_bayes(data['outcomes'],p_c,expID)
            
        elif key == "reduced_bayes_lamda":
            p_c = G_paras[key][subji][0]
            lamda = G_paras[key][subji][1]
            model_est = reduced_bayes_lamda(data['outcomes'],p_c,lamda,expID)
        
        elif key == "PID":
            Kp = G_paras[key][subji][0]
            Ki = G_paras[key][subji][1]
            Kd = G_paras[key][subji][2]
            lamda = G_paras[key][subji][3]
            model_est = PID(data['outcomes'],Kp,Ki,Kd,lamda,expID)
            
        elif key == "Mixed_delta":
        
            delta = [G_paras[key][subji][0],G_paras[key][subji][1]]
            hrate = G_paras[key][subji][2]
            nu_p = G_paras[key][subji][3]                  
            model_est = mixed_delta(data['outcomes'],delta,hrate,nu_p,expID)   
            
        elif key == "VKF": 
            lamda = G_paras[key][subji][0]
            omega = G_paras[key][subji][1]
            v0 = G_paras[key][subji][2]
            
            model_est = VKF(data['outcomes'],lamda,omega,v0,expID) 
            
        elif key == "HGF": 
            nu = G_paras[key][subji][0]
            kappa = G_paras[key][subji][1]
            omega = G_paras[key][subji][2]
            
            model_est = HGF(data['outcomes'],nu,kappa,omega,expID)
            
    
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
        
    
        if key == "changepoint":
            T1 = K_paras[key][subji][0]
            T2 = K_paras[key][subji][1]
            model_est = ChangePointModel(T1,T2,data['outcomes'],expID)
                
        elif key == "HMM":
            p_c = K_paras[key][subji][0]
            model_est = HMM(p_c,data['outcomes'],expID)
            
        elif key == "delta_rule":
            lr = K_paras[key][subji][0]
            model_est = delta_rule(lr, data['outcomes'],expID)
            
        elif key == "p_hall":
            lr = K_paras[key][subji][0]
            weight = K_paras[key][subji][1]
            model_est = pearce_hall(lr,weight,data['outcomes'],expID)
            
        elif key == "reduced_bayes":
            p_c = K_paras[key][subji][0]
            model_est = reduced_bayes(data['outcomes'],p_c,expID)
            
        elif key == "reduced_bayes_lamda":
            p_c = K_paras[key][subji][0]
            lamda = K_paras[key][subji][1]
            model_est = reduced_bayes_lamda(data['outcomes'],p_c,lamda,expID)
        
        elif key == "PID":
            Kp = K_paras[key][subji][0]
            Ki = K_paras[key][subji][1]
            Kd = K_paras[key][subji][2]
            lamda = K_paras[key][subji][3]
            model_est = PID(data['outcomes'],Kp,Ki,Kd,lamda,expID)
            
        elif key == "Mixed_delta":
            delta = [K_paras[key][subji][0], K_paras[key][subji][1]]
            hrate = K_paras[key][subji][2]
            nu_p = K_paras[key][subji][3]                  
            model_est = mixed_delta(data['outcomes'],delta,hrate,nu_p,expID)  
            
        elif key == "VKF": 
            lamda = K_paras[key][subji][0]
            omega = K_paras[key][subji][1]
            v0 = K_paras[key][subji][2]
            
            model_est = VKF(data['outcomes'],lamda,omega,v0,expID) 
            
        elif key == "HGF": 
            nu = K_paras[key][subji][0]
            kappa = K_paras[key][subji][1]
            omega = K_paras[key][subji][2]
            
            model_est = HGF(data['outcomes'],nu,kappa,omega,expID)
            
    
    
    
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
    
        if key == "changepoint":
            T1 = FM_paras[key][subji][0]
            T2 = FM_paras[key][subji][1]
            model_est = ChangePointModel(T1,T2,data['outcomes'],expID)
                
        elif key == "HMM":
            p_c = FM_paras[key][subji][0]
            model_est = HMM(p_c,data['outcomes'],expID)
            
        elif key == "delta_rule":
            lr = FM_paras[key][subji][0]
            model_est = delta_rule(lr, data['outcomes'],expID)
            
        elif key == "p_hall":
            lr = FM_paras[key][subji][0]
            weight = FM_paras[key][subji][1]
            model_est = pearce_hall(lr,weight,data['outcomes'],expID)
            
        elif key == "reduced_bayes":
            p_c = FM_paras[key][subji][0]
            model_est = reduced_bayes(data['outcomes'],p_c,expID)
            
        elif key == "reduced_bayes_lamda":
            p_c = FM_paras[key][subji][0]
            lamda = FM_paras[key][subji][1]
            model_est = reduced_bayes_lamda(data['outcomes'],p_c,lamda,expID)
        
        elif key == "PID":
            Kp = FM_paras[key][subji][0]
            Ki = FM_paras[key][subji][1]
            Kd = FM_paras[key][subji][2]
            lamda = FM_paras[key][subji][3]
            model_est = PID(data['outcomes'],Kp,Ki,Kd,lamda,expID)
            
        elif key == "Mixed_delta":
            delta = [FM_paras[key][subji][0], FM_paras[key][subji][1]]
            hrate = FM_paras[key][subji][2]
            nu_p = FM_paras[key][subji][3]                  
            model_est = mixed_delta(data['outcomes'],delta,hrate,nu_p,expID)      
            
        elif key == "VKF": 
            lamda = FM_paras[key][subji][0]
            omega = FM_paras[key][subji][1]
            v0 = FM_paras[key][subji][2]
            
            model_est = VKF(data['outcomes'],lamda,omega,v0,expID) 
            
        elif key == "HGF": 
            nu = FM_paras[key][subji][0]
            kappa = FM_paras[key][subji][1]
            omega = FM_paras[key][subji][2]
            
            model_est = HGF(data['outcomes'],nu,kappa,omega,expID)
            
    
    
        n_trials_per_session = 75  # Number of trials per session
        n_sessions = len(model_est) // n_trials_per_session  
    
    
        model_est_sessions = [model_est[i * n_trials_per_session:(i + 1) * n_trials_per_session] for i in range(n_sessions)]
    
        subj_model_estimate = model_est_sessions
    
    
        FM_dat['mod_est'][key].append(np.array(subj_model_estimate))
        
#### identify change point 

G_dat["CP"] = []

# Loop through subjects
for subji in range(len(G_dat["true_p"])):  
    cp_subj = np.zeros_like(G_dat["true_p"][subji])  
    
    # Loop through sessions
    for sessi in range(10):  
        true_p_sess = G_dat["true_p"][subji][sessi]  # Get probability sequence
        cp_subj[sessi, 1:] = np.diff(true_p_sess) != 0  # Detect change points

    G_dat["CP"].append(cp_subj)  # Append the subject's CP matrix


Khaw_dat["CP"] = []

# Loop through subjects
for subji in range(len(Khaw_dat["true_p"])):  
    cp_subj = np.zeros_like(Khaw_dat["true_p"][subji])  
    
    # Loop through sessions
    for sessi in range(10):  # 10 sessions
        true_p_sess = Khaw_dat["true_p"][subji][sessi]  # Get probability sequence
        cp_subj[sessi, 1:] = np.diff(true_p_sess) != 0  # Detect change points

    Khaw_dat["CP"].append(cp_subj)  # Append the subject's CP matrix
    


FM_dat["CP"] = []

# Loop through subjects
for subji in range(len(FM_dat["true_p"])): 
    cp_subj = np.zeros_like(FM_dat["true_p"][subji]) 
    
    # Loop through sessions
    for sessi in range(15):  
        true_p_sess = FM_dat["true_p"][subji][sessi]  
        cp_subj[sessi, 1:] = np.diff(true_p_sess) != 0  # Detect change points

    FM_dat["CP"].append(cp_subj)  # Append the subject's CP matrix

#### extract sequences

FM_dat["CPseq_true_p"] = []

# Define the range around change points (-2 before, +15 after)
pre_cp = 5
post_cp = 15


# Loop through subjects
for subji in range(len(FM_dat["CP"])):  
    cp_sequences = []  # Store sequences for this subject
    
    # Loop through sessions
    for sessi in range(15): 
        cp_indices = np.where(FM_dat["CP"][subji][sessi] == 1)[0]  # Find CP indices
        lr_sess = FM_dat["true_p"][subji][sessi]  # Get learning rate sequence
        
        for i, cp in enumerate(cp_indices):  # Loop through each change point
            start_idx = cp - pre_cp
            end_idx = cp + post_cp + 1  # Exclusive index

            # Exclude sequence if out of trial bounds
            if start_idx < 0 or end_idx > 75:
                continue

            # Exclude sequence if another CP occurs within 15 trials
            if i < len(cp_indices) - 1:  # Check if there's a next change point
                next_cp = cp_indices[i + 1]
                if next_cp < end_idx:  # Another CP is too close
                    continue

            # Extract and store valid sequence
            seq = lr_sess[start_idx:end_idx]
            cp_sequences.append(seq)

    FM_dat["CPseq_true_p"].append(np.array(cp_sequences))  # Convert to NumPy array and store


## same for subj_est
FM_dat["CPseq_sub_est"] = []

# Define the range around change points (-2 before, +15 after)
pre_cp = 5
post_cp = 15


# Loop through subjects
for subji in range(len(FM_dat["CP"])):  
    cp_sequences = []  # Store sequences for this subject
    
    # Loop through sessions
    for sessi in range(15): 
        cp_indices = np.where(FM_dat["CP"][subji][sessi] == 1)[0]  # Find CP indices
        lr_sess = FM_dat["sub_est"][subji][sessi]  # Get learning rate sequence
        
        for i, cp in enumerate(cp_indices):  # Loop through each change point
            start_idx = cp - pre_cp
            end_idx = cp + post_cp + 1  # Exclusive index

            # Exclude sequence if out of trial bounds
            if start_idx < 0 or end_idx > 75:
                continue

            # Exclude sequence if another CP occurs within 15 trials
            if i < len(cp_indices) - 1:  # Check if there's a next change point
                next_cp = cp_indices[i + 1]
                if next_cp < end_idx:  # Another CP is too close
                    continue

            # Extract and store valid sequence
            seq = lr_sess[start_idx:end_idx]
            cp_sequences.append(seq)

    FM_dat["CPseq_sub_est"].append(np.array(cp_sequences))  # Convert to NumPy array and store



  
FM_dat["CPseq_mods"] = {}
    
# Define the range around change points (-2 before, +15 after)
pre_cp = 5
post_cp = 15

mod_names = ["HMM", "changepoint", "delta_rule", "p_hall","reduced_bayes",
                  "reduced_bayes_lamda", "PID" , "Mixed_delta", "VKF", "HGF"]
    
for key in mod_names:
    
    FM_dat["CPseq_mods"][key] = []
    
    # Loop through subjects
    for subji in range(len(FM_dat["CP"])):  
        cp_sequences = []  # Store sequences for this subject
        
        # Loop through sessions
        for sessi in range(15): 
            cp_indices = np.where(FM_dat["CP"][subji][sessi] == 1)[0]  # Find CP indices
            lr_sess = FM_dat['mod_est'][key][subji][sessi]  # Get learning rate sequence
            
            for i, cp in enumerate(cp_indices):  # Loop through each change point
                start_idx = cp - pre_cp
                end_idx = cp + post_cp + 1  # Exclusive index
    
                # Exclude sequence if out of trial bounds
                if start_idx < 0 or end_idx > 75:
                    continue
    
                # Exclude sequence if another CP occurs within 15 trials
                if i < len(cp_indices) - 1:  # Check if there's a next change point
                    next_cp = cp_indices[i + 1]
                    if next_cp < end_idx:  # Another CP is too close
                        continue
    
                # Extract and store valid sequence
                seq = lr_sess[start_idx:end_idx]
                cp_sequences.append(seq)
    
        FM_dat["CPseq_mods"][key].append(np.array(cp_sequences))  # Convert to NumPy array and store






G_dat["CPseq_true_p"] = []

# Define the range around change points (-2 before, +15 after)
pre_cp = 5
post_cp = 30


# Loop through subjects
for subji in range(len(G_dat["CP"])):  
    cp_sequences = []  # Store sequences for this subject
    
    # Loop through sessions
    for sessi in range(10): 
        cp_indices = np.where(G_dat["CP"][subji][sessi] == 1)[0]  # Find CP indices
        lr_sess = G_dat["true_p"][subji][sessi]  # Get learning rate sequence
        
        for i, cp in enumerate(cp_indices):  # Loop through each change point
            start_idx = cp - pre_cp
            end_idx = cp + post_cp + 1  # Exclusive index

            # Exclude sequence if out of trial bounds
            if start_idx < 0 or end_idx > 1000:
                continue

            # Exclude sequence if another CP occurs within 15 trials
            if i < len(cp_indices) - 1:  # Check if there's a next change point
                next_cp = cp_indices[i + 1]
                if next_cp < end_idx:  # Another CP is too close
                    continue

            # Extract and store valid sequence
            seq = lr_sess[start_idx:end_idx]
            cp_sequences.append(seq)

    G_dat["CPseq_true_p"].append(np.array(cp_sequences))  # Convert to NumPy array and store


## same for subj_est
G_dat["CPseq_sub_est"] = []

# Define the range around change points (-2 before, +15 after)
pre_cp = 5
post_cp = 30


# Loop through subjects
for subji in range(len(G_dat["CP"])):  
    cp_sequences = []  # Store sequences for this subject
    
    # Loop through sessions
    for sessi in range(10): 
        cp_indices = np.where(G_dat["CP"][subji][sessi] == 1)[0]  # Find CP indices
        lr_sess = G_dat["sub_est"][subji][sessi]  # Get learning rate sequence
        
        for i, cp in enumerate(cp_indices):  # Loop through each change point
            start_idx = cp - pre_cp
            end_idx = cp + post_cp + 1  # Exclusive index

            # Exclude sequence if out of trial bounds
            if start_idx < 0 or end_idx > 1000:
                continue

            # Exclude sequence if another CP occurs within 15 trials
            if i < len(cp_indices) - 1:  # Check if there's a next change point
                next_cp = cp_indices[i + 1]
                if next_cp < end_idx:  # Another CP is too close
                    continue

            # Extract and store valid sequence
            seq = lr_sess[start_idx:end_idx]
            cp_sequences.append(seq)

    G_dat["CPseq_sub_est"].append(np.array(cp_sequences))  # Convert to NumPy array and store


G_dat["CPseq_mods"] = {}
    
# Define the range around change points (-2 before, +15 after)
pre_cp = 5
post_cp = 30

mod_names = ["HMM", "changepoint", "delta_rule", "p_hall","reduced_bayes",
                  "reduced_bayes_lamda", "PID" , "Mixed_delta", "VKF", "HGF"]
    
for key in mod_names:
    
    G_dat["CPseq_mods"][key] = []
    
    # Loop through subjects
    for subji in range(len(G_dat["CP"])):  
        cp_sequences = []  # Store sequences for this subject
        
        # Loop through sessions
        for sessi in range(10): 
            cp_indices = np.where(G_dat["CP"][subji][sessi] == 1)[0]  # Find CP indices
            lr_sess = G_dat['mod_est'][key][subji][sessi]  # Get learning rate sequence
            
            for i, cp in enumerate(cp_indices):  # Loop through each change point
                start_idx = cp - pre_cp
                end_idx = cp + post_cp + 1  # Exclusive index
    
                # Exclude sequence if out of trial bounds
                if start_idx < 0 or end_idx > 1000:
                    continue
    
                # Exclude sequence if another CP occurs within 15 trials
                if i < len(cp_indices) - 1:  # Check if there's a next change point
                    next_cp = cp_indices[i + 1]
                    if next_cp < end_idx:  # Another CP is too close
                        continue
    
                # Extract and store valid sequence
                seq = lr_sess[start_idx:end_idx]
                cp_sequences.append(seq)
    
        G_dat["CPseq_mods"][key].append(np.array(cp_sequences))  # Convert to NumPy array and store



Khaw_dat["CPseq_true_p"] = []

# Define the range around change points (-2 before, +15 after)
pre_cp = 5
post_cp = 30


# Loop through subjects
for subji in range(len(Khaw_dat["CP"])):  
    cp_sequences = []  # Store sequences for this subject
    
    # Loop through sessions
    for sessi in range(10): 
        cp_indices = np.where(Khaw_dat["CP"][subji][sessi] == 1)[0]  # Find CP indices
        lr_sess = Khaw_dat["true_p"][subji][sessi]  # Get learning rate sequence
        
        for i, cp in enumerate(cp_indices):  # Loop through each change point
            start_idx = cp - pre_cp
            end_idx = cp + post_cp + 1  # Exclusive index

            # Exclude sequence if out of trial bounds
            if start_idx < 0 or end_idx > 999:
                continue

            # Exclude sequence if another CP occurs within 15 trials
            if i < len(cp_indices) - 1:  # Check if there's a next change point
                next_cp = cp_indices[i + 1]
                if next_cp < end_idx:  # Another CP is too close
                    continue

            # Extract and store valid sequence
            seq = lr_sess[start_idx:end_idx]
            cp_sequences.append(seq)

    Khaw_dat["CPseq_true_p"].append(np.array(cp_sequences))  # Convert to NumPy array and store


## same for subj_est
Khaw_dat["CPseq_sub_est"] = []

# Define the range around change points (-2 before, +15 after)
pre_cp = 5
post_cp = 30


# Loop through subjects
for subji in range(len(Khaw_dat["CP"])):  
    cp_sequences = []  # Store sequences for this subject
    
    # Loop through sessions
    for sessi in range(10): 
        cp_indices = np.where(Khaw_dat["CP"][subji][sessi] == 1)[0]  # Find CP indices
        lr_sess = Khaw_dat["sub_est"][subji][sessi]  # Get learning rate sequence
        
        for i, cp in enumerate(cp_indices):  # Loop through each change point
            start_idx = cp - pre_cp
            end_idx = cp + post_cp + 1  # Exclusive index

            # Exclude sequence if out of trial bounds
            if start_idx < 0 or end_idx > 999:
                continue

            # Exclude sequence if another CP occurs within 15 trials
            if i < len(cp_indices) - 1:  # Check if there's a next change point
                next_cp = cp_indices[i + 1]
                if next_cp < end_idx:  # Another CP is too close
                    continue

            # Extract and store valid sequence
            seq = lr_sess[start_idx:end_idx]
            cp_sequences.append(seq)

    Khaw_dat["CPseq_sub_est"].append(np.array(cp_sequences))  # Convert to NumPy array and store

Khaw_dat["CPseq_mods"] = {}
    
# Define the range around change points (-2 before, +15 after)
pre_cp = 5
post_cp = 30

mod_names = ["HMM", "changepoint", "delta_rule", "p_hall","reduced_bayes",
                  "reduced_bayes_lamda", "PID" , "Mixed_delta", "VKF", "HGF"]
    
for key in mod_names:
    
    Khaw_dat["CPseq_mods"][key] = []
    
    # Loop through subjects
    for subji in range(len(Khaw_dat["CP"])):  
        cp_sequences = []  # Store sequences for this subject
        
        # Loop through sessions
        for sessi in range(10): 
            cp_indices = np.where(Khaw_dat["CP"][subji][sessi] == 1)[0]  # Find CP indices
            lr_sess = Khaw_dat['mod_est'][key][subji][sessi]  # Get learning rate sequence
            
            for i, cp in enumerate(cp_indices):  # Loop through each change point
                start_idx = cp - pre_cp
                end_idx = cp + post_cp + 1  # Exclusive index
    
                # Exclude sequence if out of trial bounds
                if start_idx < 0 or end_idx > 999:
                    continue
    
                # Exclude sequence if another CP occurs within 15 trials
                if i < len(cp_indices) - 1:  # Check if there's a next change point
                    next_cp = cp_indices[i + 1]
                    if next_cp < end_idx:  # Another CP is too close
                        continue
    
                # Extract and store valid sequence
                seq = lr_sess[start_idx:end_idx]
                cp_sequences.append(seq)
    
        Khaw_dat["CPseq_mods"][key].append(np.array(cp_sequences))  # Convert to NumPy array and store



#### regression 

sub_est = np.concatenate(FM_dat["CPseq_sub_est"], axis=0)
true_p = np.concatenate(FM_dat["CPseq_true_p"], axis=0)


n_iter = sub_est.shape[1]

# Storage for intercepts and coefficients
FM_intercepts = np.zeros(n_iter)
FM_coefficients = np.zeros((n_iter, 2))  # two predictors: p_before and p_after

FM_ci_lower = np.zeros((n_iter, 2))  # Lower bounds for 95% CI
FM_ci_upper = np.zeros((n_iter, 2))  # Upper bounds for 95% CI

for i in range(n_iter):
    
    p_before = true_p[:,0]
    p_after = true_p[:,5]
    
    X = np.column_stack((p_before, p_after))
    
    y = sub_est[:,i]
    
    # Fit regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Store results
    FM_intercepts[i] = model.intercept_
    FM_coefficients[i] = model.coef_
    
    
    # Compute residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Compute the standard error of the coefficients
    n = len(y)  # Number of samples
    p = X.shape[1]  # Number of predictors (including intercept)
    
    # Variance of residuals
    residual_variance = np.var(residuals, ddof=p)
    
    # Standard error of coefficients
    X_with_intercept = np.column_stack((np.ones(n), X))  # Add intercept column
    X_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)  # Inverse of X'X
    se = np.sqrt(residual_variance * np.diagonal(X_inv))  # Standard errors
    
    # Compute the 95% confidence intervals
    t_value = stats.t.ppf(0.975, df=n - p)  # t-value for 95% confidence interval
    FM_ci_lower[i] = model.coef_ - t_value * se[1:]  # Lower bound for coefficients
    FM_ci_upper[i] = model.coef_ + t_value * se[1:]  # Upper bound for coefficients

    


FM_coefficients_mods = {}

    
for key in mod_names:
    
    FM_coefficients_mods[key] = np.zeros((n_iter, 2))  # two predictors: p_before and p_after
    
    mod_estimate = np.concatenate(FM_dat["CPseq_mods"][key], axis=0)
    
    for i in range(n_iter):
        
        p_before = true_p[:,0]
        p_after = true_p[:,5]
        
        X = np.column_stack((p_before, p_after))
        
        y = mod_estimate[:,i]
        
        # Fit regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Store results
    
        FM_coefficients_mods[key][i] = model.coef_








sub_est = np.concatenate(G_dat["CPseq_sub_est"], axis=0)
true_p = np.concatenate(G_dat["CPseq_true_p"], axis=0)

n_iter = sub_est.shape[1]

# Storage for intercepts and coefficients
G_intercepts = np.zeros(n_iter)
G_coefficients = np.zeros((n_iter, 2))  # two predictors: p_before and p_after

G_ci_lower = np.zeros((n_iter, 2))  # Lower bounds for 95% CI
G_ci_upper = np.zeros((n_iter, 2))  # Upper bounds for 95% CI

for i in range(n_iter):
    
    p_before = true_p[:,0]
    p_after = true_p[:,5]
    
    X = np.column_stack((p_before, p_after))
    
    y = sub_est[:,i]
    
    # Fit regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Store results
    G_intercepts[i] = model.intercept_
    G_coefficients[i] = model.coef_
    
    # Compute residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Compute the standard error of the coefficients
    n = len(y)  # Number of samples
    p = X.shape[1]  # Number of predictors (including intercept)
    
    # Variance of residuals
    residual_variance = np.var(residuals, ddof=p)
    
    # Standard error of coefficients
    X_with_intercept = np.column_stack((np.ones(n), X))  # Add intercept column
    X_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)  # Inverse of X'X
    se = np.sqrt(residual_variance * np.diagonal(X_inv))  # Standard errors
    
    # Compute the 95% confidence intervals
    t_value = stats.t.ppf(0.975, df=n - p)  # t-value for 95% confidence interval
    G_ci_lower[i] = model.coef_ - t_value * se[1:]  # Lower bound for coefficients
    G_ci_upper[i] = model.coef_ + t_value * se[1:]  # Upper bound for coefficients

G_coefficients_mods = {}

    
for key in mod_names:
    
    G_coefficients_mods[key] = np.zeros((n_iter, 2))  # two predictors: p_before and p_after
    
    mod_estimate = np.concatenate(G_dat["CPseq_mods"][key], axis=0)
    
    for i in range(n_iter):
        
        p_before = true_p[:,0]
        p_after = true_p[:,5]
        
        X = np.column_stack((p_before, p_after))
        
        y = mod_estimate[:,i]
        
        # Fit regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Store results
    
        G_coefficients_mods[key][i] = model.coef_
        

   

sub_est = np.concatenate(Khaw_dat["CPseq_sub_est"], axis=0)
true_p = np.concatenate(Khaw_dat["CPseq_true_p"], axis=0)

n_iter = sub_est.shape[1]

# Storage for intercepts and coefficients
Khaw_intercepts = np.zeros(n_iter)
Khaw_coefficients = np.zeros((n_iter, 2))  # two predictors: p_before and p_after

Khaw_ci_lower = np.zeros((n_iter, 2))  # Lower bounds for 95% CI
Khaw_ci_upper = np.zeros((n_iter, 2))  # Upper bounds for 95% CI


for i in range(n_iter):
    
    p_before = true_p[:,0]
    p_after = true_p[:,5]
    
    X = np.column_stack((p_before, p_after))
    
    y = sub_est[:,i]
    
    # Fit regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Store results
    Khaw_intercepts[i] = model.intercept_
    Khaw_coefficients[i] = model.coef_
    
    # Compute residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Compute the standard error of the coefficients
    n = len(y)  # Number of samples
    p = X.shape[1]  # Number of predictors (including intercept)
    
    # Variance of residuals
    residual_variance = np.var(residuals, ddof=p)
    
    # Standard error of coefficients
    X_with_intercept = np.column_stack((np.ones(n), X))  # Add intercept column
    X_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)  # Inverse of X'X
    se = np.sqrt(residual_variance * np.diagonal(X_inv))  # Standard errors
    
    # Compute the 95% confidence intervals
    t_value = stats.t.ppf(0.975, df=n - p)  # t-value for 95% confidence interval
    Khaw_ci_lower[i] = model.coef_ - t_value * se[1:]  # Lower bound for coefficients
    Khaw_ci_upper[i] = model.coef_ + t_value * se[1:]  # Upper bound for coefficients


Khaw_coefficients_mods = {}

    
for key in mod_names:
    
    Khaw_coefficients_mods[key] = np.zeros((n_iter, 2))  # two predictors: p_before and p_after
    
    mod_estimate = np.concatenate(Khaw_dat["CPseq_mods"][key], axis=0)
    
    for i in range(n_iter):
        
        p_before = true_p[:,0]
        p_after = true_p[:,5]
        
        X = np.column_stack((p_before, p_after))
        
        y = mod_estimate[:,i]
        
        # Fit regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Store results
    
        Khaw_coefficients_mods[key][i] = model.coef_

#### plot

# Extract coefficients for p_before and p_after
FM_coeff_p_before = FM_coefficients[:, 0]  # Coefficients for p_before
FM_coeff_p_after = FM_coefficients[:, 1]   # Coefficients for p_after

G_coeff_p_before = G_coefficients[:, 0]  # Coefficients for p_before
G_coeff_p_after = G_coefficients[:, 1]   # Coefficients for p_after

Khaw_coeff_p_before = Khaw_coefficients[:, 0]  # Coefficients for p_before
Khaw_coeff_p_after = Khaw_coefficients[:, 1]   # Coefficients for p_after


# Create the subplots

plt.close('all')          # Closes any open figures
plt.rcdefaults()          # Resets matplotlib rcParams to defaults
plt.figure(figsize=(10, 12))  # Sets figure size manually (forces reset)
fig, axes = plt.subplots(3, 1, num=1)  # Use the manually defined figure
# fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Set up the time range
time_range_1 = np.arange(-5, 16)
time_range_2 = np.arange(-5, 31)
time_range_3 = np.arange(-5, 31)

# Define custom colors for selected models
custom_colors = {
    "HMM": "r",         
    "Mixed_delta": "b", 
    "PID": "g",         
}

# Filter colormap to avoid red, green, and blue
cmap_full = cm.get_cmap('tab10')
excluded_indices = [0, 1, 2]  # tab10[0]=blue, [1]=orange, [2]=green, [3]=red, etc.
fallback_colors = [cmap_full(i) for i in range(cmap_full.N) if i not in [0, 2, 3]]

# Assign fallback colors to remaining models
mod_names = ["HMM", "HGF", "VKF", "Mixed_delta", "reduced_bayes", "reduced_bayes_lamda", "changepoint", "PID", "p_hall", "delta_rule"]
remaining_models = [m for m in mod_names if m not in custom_colors]
fallback_color_dict = {model: fallback_colors[i] for i, model in enumerate(remaining_models)}

# Merge custom and fallback color dictionaries
final_colors = {**custom_colors, **fallback_color_dict}

# First plot (Foucault & Meyniel, 2024)
ax = axes[0]
for model in mod_names:
    color = final_colors[model]
    coeffs = FM_coefficients_mods[model]
    ax.plot(time_range_1, coeffs[:, 0], linestyle='--', color=color, linewidth=1, zorder=1)
    ax.plot(time_range_1, coeffs[:, 1], linestyle='-', color=color, linewidth=1, zorder=1)

# Plot subject data on top
ax.plot(time_range_1, FM_coeff_p_before, color='k', linestyle='--', linewidth=1, zorder=2)
ax.plot(time_range_1, FM_coeff_p_after, color='k', linestyle='-', linewidth=1, zorder=2)

# Add CI shading
ax.fill_between(time_range_1, FM_ci_lower[:, 0], FM_ci_upper[:, 0], color='grey', alpha=0.2)
ax.fill_between(time_range_1, FM_ci_lower[:, 1], FM_ci_upper[:, 1], color='grey', alpha=0.2)

# Plot key models (HMM, PID, Mixed_delta) on top
for model in ["HMM", "PID", "Mixed_delta"]:
    color = final_colors[model]
    coeffs = FM_coefficients_mods[model]
    ax.plot(time_range_1, coeffs[:, 0], linestyle='--', color=color, linewidth=1, zorder=3)
    ax.plot(time_range_1, coeffs[:, 1], linestyle='-', color=color, linewidth=1, zorder=3)

# Shade region before change point
ax.axvspan(-5, 0, color='gray', alpha=0.2)

# Ticks
ax.set_xticks(np.array([-5, 0, 5, 10, 15]))
ax.set_yticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1]))

# Labels
ax.set_xlabel('Trial number relative to change point')
ax.set_ylabel('Coefficient weight')
ax.set_title('Foucault & Meyniel (2024)')

# First legend (line styles)
marker_b = Line2D([0], [0], color='k', linestyle='--', linewidth=1, label='Generative prior before the change point')
marker_r = Line2D([0], [0], color='k', linestyle='-', linewidth=1, label='Generative prior after the change point')
first_legend = ax.legend(handles=[marker_b, marker_r], loc='lower right', frameon=False)
ax.add_artist(first_legend)

# Second legend (models + subject line)
subject_line = Line2D([0], [0], color='k', lw=2, label='Subjects')
model_lines = [
    Line2D([0], [0], color=final_colors[model], lw=2, label=f"Custom Label for {model}")
    for model in mod_names
]

# Example: Manually set model names
manual_labels = {
    "HMM": "HMM",
    "PID": "PID",
    "Mixed_delta": "Mixture of delta rules",
    "changepoint": "Change Point model",
    "delta_rule": "Delta Rule",
    "p_hall": "Pearce-Hall model",
    "reduced_bayes": "Reduced Bayesian model",
    "reduced_bayes_lamda": "Reduced Bayesian model\n(with under-weighted likelihood)",
    "VKF": "VKF",
    "HGF": "HGF"
}

model_lines = [
    Line2D([0], [0], color=final_colors[model], lw=2, label=manual_labels.get(model, model))
    for model in mod_names
]

second_legend = ax.legend(handles=[subject_line] + model_lines, loc='upper left', ncol=4, frameon=False)

# Second plot (Gallistel et al., 2014)
ax = axes[1]
# Repeat similar plotting logic for G_coefficients_mods and other variables related to the G_dat dataset
for model in mod_names:
    color = final_colors[model]
    coeffs = G_coefficients_mods[model]
    ax.plot(time_range_2, coeffs[:, 0], linestyle='--', color=color, linewidth=1, zorder=1)
    ax.plot(time_range_2, coeffs[:, 1], linestyle='-', color=color, linewidth=1, zorder=1)

ax.plot(time_range_2, G_coeff_p_before, color='k', linestyle='--', linewidth=1, zorder=2)
ax.plot(time_range_2, G_coeff_p_after, color='k', linestyle='-', linewidth=1, zorder=2)

ax.fill_between(time_range_2, G_ci_lower[:, 0], G_ci_upper[:, 0], color='grey', alpha=0.2)
ax.fill_between(time_range_2, G_ci_lower[:, 1], G_ci_upper[:, 1], color='grey', alpha=0.2)

for model in ["HMM", "PID", "Mixed_delta"]:
    color = final_colors[model]
    coeffs = G_coefficients_mods[model]
    ax.plot(time_range_2, coeffs[:, 0], linestyle='--', color=color, linewidth=1, zorder=3)
    ax.plot(time_range_2, coeffs[:, 1], linestyle='-', color=color, linewidth=1, zorder=3)

ax.axvspan(-5, 0, color='gray', alpha=0.2)

ax.set_xticks(np.array([-5, 0, 5, 10, 15, 20, 25, 30]))
ax.set_yticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1]))
ax.set_xlabel('Trial number relative to change point')
ax.set_ylabel('Coefficient weight')
ax.set_title('Gallistel et al. (2014)')

# Third plot (Khaw et al., 2017)
ax = axes[2]
# Repeat similar plotting logic for Khaw_coefficients_mods and other variables related to the K_dat dataset
for model in mod_names:
    color = final_colors[model]
    coeffs = Khaw_coefficients_mods[model]
    ax.plot(time_range_3, coeffs[:, 0], linestyle='--', color=color, linewidth=1, zorder=1)
    ax.plot(time_range_3, coeffs[:, 1], linestyle='-', color=color, linewidth=1, zorder=1)

ax.plot(time_range_3, Khaw_coeff_p_before, color='k', linestyle='--', linewidth=1, zorder=2)
ax.plot(time_range_3, Khaw_coeff_p_after, color='k', linestyle='-', linewidth=1, zorder=2)

ax.fill_between(time_range_3, Khaw_ci_lower[:, 0], Khaw_ci_upper[:, 0], color='grey', alpha=0.2)
ax.fill_between(time_range_3, Khaw_ci_lower[:, 1], Khaw_ci_upper[:, 1], color='grey', alpha=0.2)

for model in ["HMM", "PID", "Mixed_delta"]:
    color = final_colors[model]
    coeffs = Khaw_coefficients_mods[model]
    ax.plot(time_range_3, coeffs[:, 0], linestyle='--', color=color, linewidth=1, zorder=3)
    ax.plot(time_range_3, coeffs[:, 1], linestyle='-', color=color, linewidth=1, zorder=3)

ax.axvspan(-5, 0, color='gray', alpha=0.2)

ax.set_xticks(np.array([-5, 0, 5, 10, 15, 20, 25, 30]))
ax.set_yticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1]))
ax.set_xlabel('Trial number relative to change point')
ax.set_ylabel('Coefficient weight')
ax.set_title('Khaw et al. (2017)')

# Style
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Show the plot
plt.tight_layout()

plt.savefig("figures/Figure5.png", dpi=300)

plt.show()


       
#%% Figure 6

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
fig = plt.figure(figsize=(18, 19))

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

    ax.set_title(title,fontsize=14)
    ax.set_xlabel("Reference Models",fontsize=14)

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



ax1.bar(x - bar_width, ada_values, width=bar_width, label='Foucault & Meyniel (2024)', color='tab:blue', alpha=1,edgecolor='black')
ax1.bar(x, G_values, width=bar_width, label='Gallistel et al. (2014)', color='tab:orange', alpha=1,edgecolor='black')
ax1.bar(x + bar_width, K_values, width=bar_width, label='Khaw et al. (2017)', color='tab:green', alpha=1,edgecolor='black')

# Add "+" marker for highest bars
# Get index of max values
max_ada_idx = np.argmax(ada_values)
max_G_idx = np.argmax(G_values)
max_K_idx = np.argmax(K_values)

# Add "+" above each max bar
ax1.text(x[max_ada_idx] - bar_width, ada_values[max_ada_idx] + 1.5, "+", 
         ha='center', va='bottom', fontsize=18, fontweight='bold', color='tab:blue')
ax1.text(x[max_G_idx], G_values[max_G_idx] + 1.5, "+", 
         ha='center', va='bottom', fontsize=18, fontweight='bold', color='tab:orange')
ax1.text(x[max_K_idx] + bar_width, K_values[max_K_idx] + 1.5, "+", 
         ha='center', va='bottom', fontsize=18, fontweight='bold', color='tab:green')

ax1.set_ylabel('Percentage of Subjects (%)')
ax1.set_title('Best-Fitting Model',fontsize=14)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.yaxis.grid(True, linestyle="--", alpha=0.7)
ax1.legend(fontsize=14)

# Shared x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(custom_x_labels, rotation=45, ha="right",fontsize=14)

plt.tight_layout()
plt.savefig("figures/Figure6.png", dpi=300, bbox_inches='tight')
plt.show()



#%% Figure S2



def load_all_result_dicts(models, folder="results/para_recovery"):
    all_results = {}

    for model in models:
        try:
            in_path = os.path.join(folder, f"{model}_in.csv")
            out_path = os.path.join(folder, f"{model}_out.csv")

            result_dict = {
                "in": pd.read_csv(in_path),
                "out": pd.read_csv(out_path)
            }

            all_results[model] = result_dict
        except FileNotFoundError:
            print(f"Files for model '{model}' not found. Skipping.")
    
    
    return all_results



models = [
    "changepoint", "HMM", "delta_rule", "p_hall", "reduced_bayes",
    "reduced_bayes_lamda", "PID", "Mixed_delta", "VKF", "HGF"
]

all_result_dicts = load_all_result_dicts(models)



def get_confusion_matrix(result_dict, corr_type="pearson"):
    """ corr_type can be 'pearson' or 'spearman' """
    
    invals = result_dict["in"]
    outvals = result_dict["out"]
    
    confmat = pd.DataFrame(columns=invals.columns, index=outvals.columns, 
                           dtype=float)
    confmat_pval = pd.DataFrame(columns=invals.columns, index=outvals.columns, 
                                dtype=float)
    for i in invals.columns:
        for o in outvals.columns:
            if corr_type == "pearson":
                confmat.loc[o, i], confmat_pval.loc[o, i] = sps.pearsonr(
                    invals[i], outvals[o])
            elif corr_type == "spearman":
                confmat.loc[o, i], confmat_pval.loc[o, i] = sps.spearmanr(
                    invals[i], outvals[o])
    
    return {"conf_matrix": confmat, "pvalues": confmat_pval, "type": corr_type}




def plot_confusoin_matrix(result_dict, model, ax, x_labels=None, y_labels=None):
    # Get confusion matrix and p-values
    conf_matrices = get_confusion_matrix(result_dict, corr_type="spearman")
    conf_mat = conf_matrices["conf_matrix"]
    pmat = conf_matrices["pvalues"]

    nrec, ngen = conf_mat.shape
    recnames = y_labels if y_labels is not None else conf_mat.index
    gennames = x_labels if x_labels is not None else conf_mat.columns

    xlabel = "Generative parameters"
    ylabel = "Recovered parameters"
    p_crit = 0.05
    cell_fontsize = 8
    labels_fontsize = 8
    cmap = "coolwarm"

    x = np.arange(ngen) + 0.5
    y = np.arange(nrec) + 0.5

    # Plot heatmap
    im = ax.imshow(conf_mat, cmap=cmap, extent=[0, ngen, 0, nrec],
                   aspect=0.9, vmin=-1, vmax=1, interpolation='nearest')

    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xticklabels(gennames, rotation=0, ha='center', fontsize=labels_fontsize)
    ax.set_yticklabels(np.flip(recnames), fontsize=labels_fontsize)

    ax.set_title(model)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    labels = np.ones((nrec, ngen)).astype(str)

    for ip1, p1 in enumerate(conf_mat.index):
        for ip2, p2 in enumerate(conf_mat.columns):    
            p_val = pmat.loc[p1, p2]
            stars = (
                "***" if p_val <= .001 else
                "**" if p_val <= .01 else
                "*" if p_val < p_crit else
                ""
            )
            labels[ip1, ip2] = f"{conf_mat.loc[p1, p2]:0.2f}\n{stars}"

    for (j, i), label in np.ndenumerate(np.flipud(labels)):
        value = np.flipud(conf_mat)[j, i]
        color = (1, 1, 1) if abs(value) > 0.25 else (0, 0, 0)
        ax.text(i + 0.5, j + 0.5, label, ha='center', va='center', 
                fontsize=cell_fontsize, c=color)

    return im, conf_matrices   



fig, axes = plt.subplots(2, 5, figsize=(21, 9))
axes = axes.flatten()


custom_titles = {
    "HMM": "HMM",
    "HGF": "HGF",
    "VKF": "VKF",
    "Mixed_delta": "Mixture of delta rules",
    "reduced_bayes": "Reduced Bayesian model",
    "reduced_bayes_lamda": "Reduced Bayesian model\n(under-weighted likelihood)",
    "changepoint": "Change Point model",
    "PID": "PID",
    "p_hall": "Pearce-Hall model",
    "delta_rule": "Delta rule"
}



models = [
    ("HMM", [r"$h$"], [r"$h$"]),
    ("HGF", [r"$\vartheta$", "K", r"$\omega$"], [r"$\vartheta$", "K", r"$\omega$"]),
    ("VKF", [r"$\lambda$", r"$\omega$", "v0"], [r"$\lambda$", r"$\omega$", "v0"]),
    ("Mixed_delta", [r"$l_1$", r"$l_2$", r"$h$", r"$v_p$"], [r"$l_1$", r"$l_2$", r"$h$", r"$v_p$"]),
    ("reduced_bayes", [r"$h$"], [r"$h$"]),
    ("reduced_bayes_lamda", [r"$h$", r"$\lambda$"], [r"$h$", r"$\lambda$"]),
    ("changepoint", [r"$T1$", r"$T2$"], [r"$T1$", r"$T2$"]),
    ("PID", [r"$K_P$", r"$K_I$", r"$K_D$", r"$\lambda$"], [r"$K_P$", r"$K_I$", r"$K_D$", r"$\lambda$"]),
    ("p_hall", [r'$\alpha$', r'$\eta$'], [r'$\alpha$', r'$\eta$']),
    ("delta_rule", [r'$\alpha$'], [r'$\alpha$'])
]

# Collect image objects
images = []

for i, (model, x_labels, y_labels) in enumerate(models):
    im,conf_matrices = plot_confusoin_matrix(
        result_dict=all_result_dicts[model],
        model=model,
        ax=axes[i],
        x_labels=x_labels,
        y_labels=y_labels
    )
    
    for key, df in list(conf_matrices.items())[:-1]:
        df.to_csv(f"results/para_recovery/{model}_{key}.csv", index=False)
        
    # Set the custom title for each subplot
    axes[i].set_title(custom_titles[model])
    images.append(im)


# Adjust subplot spacing to leave room for the colorbar and reduce vertical space between rows
fig.subplots_adjust(right=0.88, hspace=0.01)  # Reduce the vertical space # shrink plot area to make space on the right

# Add a single colorbar on the far right
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])  # [left, bottom, width, height]

cbar = fig.colorbar(images[0], cax=cbar_ax)

# Set colorbar ticks to only -1 and 1
cbar.set_ticks([-1, 1])
cbar.set_ticklabels(['-1', '1'])

cbar.set_label("Spearman's r")

plt.savefig("figures/FigureS2.png", dpi=300, bbox_inches='tight')
plt.show()

#%% Figure S3

#### get subjects lr

G_dat['lr_clip'] = []
Khaw_dat['lr_clip'] = []
FM_dat['lr_clip'] = []


for subji in range(G_subj_n):  
    
    subj_lr = []
    
    for sessi in range(len(G_all_sess)):
        
        outcome = G_dat["outcome"][subji][sessi]
        sub_est = G_dat["sub_est"][subji][sessi]
        
        lr = da.learning_rate_clip(outcome,sub_est)
        
        subj_lr.append(lr)
        
    G_dat['lr_clip'].append(np.array(subj_lr))
    
        
for subji in range(K_subj_n):  
    
    subj_lr = []
    
    for sessi in range(len(K_all_sess)):
        
        outcome = Khaw_dat["outcome"][subji][sessi]
        sub_est = Khaw_dat["sub_est"][subji][sessi]
        
        lr = da.learning_rate_clip(outcome,sub_est)
        
        subj_lr.append(lr)
        
    Khaw_dat['lr_clip'].append(np.array(subj_lr))
        
        
for subji in range(FM_subj_n):  
    
    subj_lr = []
    
    for sessi in range(len(FM_all_sess)):
        
        outcome = FM_dat["outcome"][subji][sessi]
        sub_est = FM_dat["sub_est"][subji][sessi]
        
        lr = da.learning_rate_clip(outcome,sub_est)
        
        subj_lr.append(lr)
        
    FM_dat['lr_clip'].append(np.array(subj_lr))


#### get models lr 

mod_names = ["HMM", "changepoint", "delta_rule", "p_hall", "reduced_bayes",
             "reduced_bayes_lamda", "PID", "Mixed_delta", "VKF", "HGF"]


G_dat['mod_lr_clip'] = {}
Khaw_dat['mod_lr_clip'] = {}
FM_dat['mod_lr_clip'] = {}


# Loop through subjects and apply models
for key in mod_names:
    G_dat['mod_lr_clip'][key] = []
    Khaw_dat['mod_lr_clip'][key] = []
    FM_dat['mod_lr_clip'][key] = []
    
    for subji in range(G_subj_n):  
        
        subj_mod_lr = []
        
        for sessi in range(len(G_all_sess)):
            
            outcome = G_dat["outcome"][subji][sessi]
            mod_est = G_dat["mod_est"][key][subji][sessi]
            
            lr = da.learning_rate_clip(outcome,mod_est)
            
            subj_mod_lr.append(lr)
            
        G_dat['mod_lr_clip'][key].append(np.array(subj_mod_lr))

    for subji in range(K_subj_n):  
        
        subj_mod_lr = []
        
        for sessi in range(len(K_all_sess)):
            
            outcome = Khaw_dat["outcome"][subji][sessi]
            mod_est = Khaw_dat["mod_est"][key][subji][sessi]
            
            lr = da.learning_rate_clip(outcome,mod_est)
            
            subj_mod_lr.append(lr)
            
        Khaw_dat['mod_lr_clip'][key].append(np.array(subj_mod_lr))
        
            
    for subji in range(FM_subj_n):  
        
        subj_mod_lr = []
        
        for sessi in range(len(FM_all_sess)):
            
            outcome = FM_dat["outcome"][subji][sessi]
            mod_est = FM_dat["mod_est"][key][subji][sessi]
            
            lr = da.learning_rate_clip(outcome,mod_est)
            
            subj_mod_lr.append(lr)
            
        FM_dat['mod_lr_clip'][key].append(np.array(subj_mod_lr))
    


#### extract lr sequences

FM_dat["CPseq_lr_clip"] = []

# Define the range around change points (-2 before, +15 after)
pre_cp = 5
post_cp = 15


# Loop through subjects
for subji in range(len(FM_dat["CP"])):  
    cp_sequences = []  # Store sequences for this subject
    
    # Loop through sessions
    for sessi in range(len(FM_dat["outcome"][0])): 
        cp_indices = np.where(FM_dat["CP"][subji][sessi] == 1)[0]  # Find CP indices
        lr_sess = FM_dat["lr_clip"][subji][sessi]  # Get learning rate sequence
        
        for i, cp in enumerate(cp_indices):  # Loop through each change point
            start_idx = cp - pre_cp
            end_idx = cp + post_cp + 1  # Exclusive index

            # Exclude sequence if out of trial bounds
            if start_idx < 0 or end_idx > len(FM_dat["outcome"][0][1]):
                continue

            # Exclude sequence if another CP occurs within 15 trials
            if i < len(cp_indices) - 1:  # Check if there's a next change point
                next_cp = cp_indices[i + 1]
                if next_cp < end_idx:  # Another CP is too close
                    continue

            # Extract and store valid sequence
            seq = lr_sess[start_idx:end_idx]
            cp_sequences.append(seq)

    FM_dat["CPseq_lr_clip"].append(np.array(cp_sequences))  # Convert to NumPy array and store


FM_dat['CPseq_mod_lr_clip'] = {}

for key in mod_names:
    
    FM_dat['CPseq_mod_lr_clip'][key] = []
    
    for subji in range(len(FM_dat["CP"])):  
        subj_cp_sequences = []  # Store sequences for this subject
        
        # Loop through sessions
        for sessi in range(len(FM_dat["outcome"][0])): 
            cp_indices = np.where(FM_dat["CP"][subji][sessi] == 1)[0]  # Find CP indices
            lr_sess = FM_dat["mod_lr_clip"][key][subji][sessi]  # Get learning rate sequence
            
            for i, cp in enumerate(cp_indices):  # Loop through each change point
                start_idx = cp - pre_cp
                end_idx = cp + post_cp + 1  # Exclusive index

                # Exclude sequence if out of trial bounds
                if start_idx < 0 or end_idx > len(FM_dat["outcome"][0][1]):
                    continue

                # Exclude sequence if another CP occurs within 15 trials
                if i < len(cp_indices) - 1:  # Check if there's a next change point
                    next_cp = cp_indices[i + 1]
                    if next_cp < end_idx:  # Another CP is too close
                        continue
                    
                # Extract and store valid sequence
                seq = lr_sess[start_idx:end_idx]
                subj_cp_sequences.append(seq)
                
        FM_dat["CPseq_mod_lr_clip"][key].append(np.array(subj_cp_sequences))  # Convert to NumPy array and store



## G_dat

# Initialize CPseq key as an empty list
G_dat["CPseq_lr_clip"] = []

# Define the range around change points 
pre_cp = 5
post_cp = 30

# Loop through subjects
for subji in range(len(G_dat["CP"])):  # 10 subjects
    cp_sequences = []  # Store sequences for this subject
    
    # Loop through sessions
    for sessi in range(len(G_dat["outcome"][0])):  # 10 sessions
        cp_indices = np.where(G_dat["CP"][subji][sessi] == 1)[0]  # Find CP indices
        lr_sess = G_dat["lr_clip"][subji][sessi]  # Get learning rate sequence
        
        for i, cp in enumerate(cp_indices):  # Loop through each change point
            start_idx = cp - pre_cp
            end_idx = cp + post_cp + 1  # Exclusive index

            # Exclude sequence if out of trial bounds
            if start_idx < 0 or end_idx > len(G_dat["outcome"][0][1]):
                continue

            # Exclude sequence if another CP occurs within 15 trials
            if i < len(cp_indices) - 1:  # Check if there's a next change point
                next_cp = cp_indices[i + 1]
                if next_cp < end_idx:  # Another CP is too close
                    continue

            # Extract and store valid sequence
            seq = lr_sess[start_idx:end_idx]
            cp_sequences.append(seq)

    G_dat["CPseq_lr_clip"].append(np.array(cp_sequences))  # Convert to NumPy array and store


G_dat['CPseq_mod_lr_clip'] = {}

for key in mod_names:
    
    G_dat['CPseq_mod_lr_clip'][key] = []
    
    for subji in range(len(G_dat["CP"])):  
        subj_cp_sequences = []  # Store sequences for this subject
        
        # Loop through sessions
        for sessi in range(len(G_dat["outcome"][0])): 
            cp_indices = np.where(G_dat["CP"][subji][sessi] == 1)[0]  # Find CP indices
            lr_sess = G_dat["mod_lr_clip"][key][subji][sessi]  # Get learning rate sequence
            
            for i, cp in enumerate(cp_indices):  # Loop through each change point
                start_idx = cp - pre_cp
                end_idx = cp + post_cp + 1  # Exclusive index

                # Exclude sequence if out of trial bounds
                if start_idx < 0 or end_idx > len(G_dat["outcome"][0][1]):
                    continue

                # Exclude sequence if another CP occurs within 15 trials
                if i < len(cp_indices) - 1:  # Check if there's a next change point
                    next_cp = cp_indices[i + 1]
                    if next_cp < end_idx:  # Another CP is too close
                        continue
                    
                # Extract and store valid sequence
                seq = lr_sess[start_idx:end_idx]
                subj_cp_sequences.append(seq)
                
        G_dat["CPseq_mod_lr_clip"][key].append(np.array(subj_cp_sequences))  # Convert to NumPy array and store


Khaw_dat["CPseq_lr_clip"] = []


# Loop through subjects
for subji in range(len(Khaw_dat["CP"])):  
    cp_sequences = []  # Store sequences for this subject
    
    # Loop through sessions
    for sessi in range(len(Khaw_dat["outcome"][0])): 
        cp_indices = np.where(Khaw_dat["CP"][subji][sessi] == 1)[0]  # Find CP indices
        lr_sess = Khaw_dat["lr_clip"][subji][sessi]  # Get learning rate sequence
        
        for i, cp in enumerate(cp_indices):  # Loop through each change point
            start_idx = cp - pre_cp
            end_idx = cp + post_cp + 1  # Exclusive index

            # Exclude sequence if out of trial bounds
            if start_idx < 0 or end_idx > len(Khaw_dat["outcome"][0][1]):
                continue

            # Exclude sequence if another CP occurs within 15 trials
            if i < len(cp_indices) - 1:  # Check if there's a next change point
                next_cp = cp_indices[i + 1]
                if next_cp < end_idx:  # Another CP is too close
                    continue

            # Extract and store valid sequence
            seq = lr_sess[start_idx:end_idx]
            cp_sequences.append(seq)

    Khaw_dat["CPseq_lr_clip"].append(np.array(cp_sequences))  # Convert to NumPy array and store


Khaw_dat['CPseq_mod_lr_clip'] = {}


for key in mod_names:
    
    Khaw_dat['CPseq_mod_lr_clip'][key] = []
    
    for subji in range(len(Khaw_dat["CP"])):  
        subj_cp_sequences = []  # Store sequences for this subject
        
        # Loop through sessions
        for sessi in range(len(Khaw_dat["outcome"][0])): 
            cp_indices = np.where(Khaw_dat["CP"][subji][sessi] == 1)[0]  # Find CP indices
            lr_sess = Khaw_dat["mod_lr_clip"][key][subji][sessi]  # Get learning rate sequence
            
            for i, cp in enumerate(cp_indices):  # Loop through each change point
                start_idx = cp - pre_cp
                end_idx = cp + post_cp + 1  # Exclusive index

                # Exclude sequence if out of trial bounds
                if start_idx < 0 or end_idx > len(Khaw_dat["outcome"][0][1]):
                    continue

                # Exclude sequence if another CP occurs within 15 trials
                if i < len(cp_indices) - 1:  # Check if there's a next change point
                    next_cp = cp_indices[i + 1]
                    if next_cp < end_idx:  # Another CP is too close
                        continue
                    
                # Extract and store valid sequence
                seq = lr_sess[start_idx:end_idx]
                subj_cp_sequences.append(seq)
                
        Khaw_dat["CPseq_mod_lr_clip"][key].append(np.array(subj_cp_sequences))  # Convert to NumPy array and store


#### group level median lrs 

FM_avg_across_subjects_clip = []
FM_sem_across_subjects_clip = []

time_range = np.arange(-5, 16)
n_boot = 1000  # Number of bootstrap samples

for t in range(len(time_range)):
    values_at_t = []

    # Collect values across subjects for this timepoint
    for subji in range(len(FM_dat["CPseq_lr_clip"])):
        subj_cpseq = FM_dat["CPseq_lr_clip"][subji]
        avg_at_t_for_subj = np.mean(subj_cpseq[:, t])
        values_at_t.append(avg_at_t_for_subj)

    values_at_t = np.array(values_at_t)

    # Compute median across subjects
    median_at_t = np.median(values_at_t)
    FM_avg_across_subjects_clip.append(median_at_t)

    # === Bootstrap SEM for the median ===
    boot_medians = []
    for _ in range(n_boot):
        boot_sample = np.random.choice(values_at_t, size=len(values_at_t), replace=True)
        boot_medians.append(np.median(boot_sample))
    sem_at_t = np.std(boot_medians)  # Standard deviation of bootstrap medians = SEM estimate

    FM_sem_across_subjects_clip.append(sem_at_t)

# Convert to numpy arrays
FM_avg_across_subjects_clip = np.array(FM_avg_across_subjects_clip)
FM_sem_across_subjects_clip = np.array(FM_sem_across_subjects_clip)


FM_lrs_avg_across_subjects_mod = {}
FM_lrs_sem_across_subjects_mod = {}

for key in mod_names:  
    
    FM_lrs_avg_across_subjects_mod[key] = []
    FM_lrs_sem_across_subjects_mod[key] = []
    
    # Loop through each timestep (from -2 to 15)
    for t in range(len(time_range)):  # 18 timesteps, corresponding to -2 to 15
        # Initialize a list to hold the value for each subject at this timestep
        values_at_t = []
        
        # Loop through each subject in G_dat['CPseq']
        for subji in range(len(FM_dat["CPseq_mod_lr_clip"][key])):
            subj_cpseq = FM_dat["CPseq_mod_lr_clip"][key][subji]  # Extract the change point sequences for this subject
            
            
            avg_at_t_for_subj = np.mean(subj_cpseq[:, t])  # Average for the current subject at timestep t
                
            values_at_t.append(avg_at_t_for_subj)  # Collect the averaged value for the subject
                
        # Compute the average for this timestep across all subjects
        avg_at_t = np.median(values_at_t)  
        
        # Append the averaged value to the list
        FM_lrs_avg_across_subjects_mod[key].append(avg_at_t)
        
        # Compute the SEM for this timestep across subjects
        sem_at_t = np.std(values_at_t) / np.sqrt(len(values_at_t))  # SEM across subjects
        FM_lrs_sem_across_subjects_mod[key].append(sem_at_t)
    
    # Now `avg_across_subjects` contains the average value at each timestep across subjects
    FM_lrs_sem_across_subjects_mod[key] = np.array(FM_lrs_sem_across_subjects_mod[key])  # Convert to a numpy array for easy plotting   
    FM_lrs_avg_across_subjects_mod[key] = np.array(FM_lrs_avg_across_subjects_mod[key]) 



G_avg_across_subjects_clip = []
G_sem_across_subjects_clip = []
time_range = np.arange(-5, 31) 
n_boot = 1000  # Number of bootstrap samples

for t in range(len(time_range)):
    values_at_t = []

    # Collect values across subjects for this timepoint
    for subji in range(len(G_dat["CPseq_lr_clip"])):
        subj_cpseq = G_dat["CPseq_lr_clip"][subji]
        avg_at_t_for_subj = np.mean(subj_cpseq[:, t])
        values_at_t.append(avg_at_t_for_subj)

    values_at_t = np.array(values_at_t)

    # Compute median across subjects
    median_at_t = np.median(values_at_t)
    G_avg_across_subjects_clip.append(median_at_t)

    # === Bootstrap SEM for the median ===
    boot_medians = []
    for _ in range(n_boot):
        boot_sample = np.random.choice(values_at_t, size=len(values_at_t), replace=True)
        boot_medians.append(np.median(boot_sample))
    sem_at_t = np.std(boot_medians)  # Standard deviation of bootstrap medians = SEM estimate

    G_sem_across_subjects_clip.append(sem_at_t)

# Convert to numpy arrays
G_avg_across_subjects_clip = np.array(G_avg_across_subjects_clip)
G_sem_across_subjects_clip = np.array(G_sem_across_subjects_clip)

G_lrs_avg_across_subjects_mod = {}

for key in mod_names:  
    
    G_lrs_avg_across_subjects_mod[key] = []

    
    # Loop through each timestep (from -2 to 15)
    for t in range(len(time_range)):  # 18 timesteps, corresponding to -2 to 15
        # Initialize a list to hold the value for each subject at this timestep
        values_at_t = []
        
        # Loop through each subject in G_dat['CPseq']
        for subji in range(len(G_dat["CPseq_mod_lr_clip"][key])):
            subj_cpseq = G_dat["CPseq_mod_lr_clip"][key][subji]  # Extract the change point sequences for this subject
            
            
            avg_at_t_for_subj = np.mean(subj_cpseq[:, t])  # Average for the current subject at timestep t
                
            values_at_t.append(avg_at_t_for_subj)  # Collect the averaged value for the subject
                
        # Compute the average for this timestep across all subjects
        avg_at_t = np.median(values_at_t)  
        
        # Append the averaged value to the list
        G_lrs_avg_across_subjects_mod[key].append(avg_at_t)
   
    G_lrs_avg_across_subjects_mod[key] = np.array(G_lrs_avg_across_subjects_mod[key]) 


Khaw_avg_across_subjects_clip = []
Khaw_sem_across_subjects_clip = []

for t in range(len(time_range)):
    values_at_t = []

    # Collect values across subjects for this timepoint
    for subji in range(len(Khaw_dat["CPseq_lr_clip"])):
        subj_cpseq = Khaw_dat["CPseq_lr_clip"][subji]
        avg_at_t_for_subj = np.mean(subj_cpseq[:, t])
        values_at_t.append(avg_at_t_for_subj)

    values_at_t = np.array(values_at_t)

    # Compute median across subjects
    median_at_t = np.median(values_at_t)
    Khaw_avg_across_subjects_clip.append(median_at_t)

    # === Bootstrap SEM for the median ===
    boot_medians = []
    for _ in range(n_boot):
        boot_sample = np.random.choice(values_at_t, size=len(values_at_t), replace=True)
        boot_medians.append(np.median(boot_sample))
    sem_at_t = np.std(boot_medians)  # Standard deviation of bootstrap medians = SEM estimate

    Khaw_sem_across_subjects_clip.append(sem_at_t)

# Convert to numpy arrays
Khaw_avg_across_subjects_clip = np.array(Khaw_avg_across_subjects_clip)
Khaw_sem_across_subjects_clip = np.array(Khaw_sem_across_subjects_clip)


Khaw_lrs_avg_across_subjects_mod = {}

for key in mod_names:  
    
    Khaw_lrs_avg_across_subjects_mod[key] = []

    
    # Loop through each timestep (from -2 to 15)
    for t in range(len(time_range)):  # 18 timesteps, corresponding to -2 to 15
        # Initialize a list to hold the value for each subject at this timestep
        values_at_t = []
        
        # Loop through each subject in G_dat['CPseq']
        for subji in range(len(Khaw_dat["CPseq_mod_lr_clip"][key])):
            subj_cpseq = Khaw_dat["CPseq_mod_lr_clip"][key][subji]  # Extract the change point sequences for this subject
            
            
            avg_at_t_for_subj = np.mean(subj_cpseq[:, t])  # Average for the current subject at timestep t
                
            values_at_t.append(avg_at_t_for_subj)  # Collect the averaged value for the subject
                
        # Compute the average for this timestep across all subjects
        avg_at_t = np.median(values_at_t)  
        
        # Append the averaged value to the list
        Khaw_lrs_avg_across_subjects_mod[key].append(avg_at_t)
        
        
    
   
    Khaw_lrs_avg_across_subjects_mod[key] = np.array(Khaw_lrs_avg_across_subjects_mod[key]) 


#### plot

# Model names
mod_names = ["HMM", "HGF", "VKF", "Mixed_delta", "reduced_bayes", "reduced_bayes_lamda",
                "changepoint", "PID", "p_hall"]

custom_labels = {
    "HMM": "HMM",
    "HGF": "HGF",
    "VKF": "VKF",
    "Mixed_delta": "Mixture of delta rules",
    "reduced_bayes": "Reduced Bayesian model",
    "reduced_bayes_lamda": "Reduced Bayesian model\n(under-weighted likelihood)",
    "changepoint": "Change Point model",
    "PID": "PID",
    "p_hall": "Pearce-Hall model"
}
# Manual colors
manual_color_indices = {"Mixed_delta": 0, "PID": 1, "HMM": 3}
manual_colors = {model: plt.get_cmap("tab10")(idx) for model, idx in manual_color_indices.items()}

# Assign other colors, skipping indices already used
used_indices = set(manual_color_indices.values())
cmap = plt.get_cmap("tab10")
available_indices = [i for i in range(cmap.N) if i not in used_indices]
other_models = [m for m in mod_names if m not in manual_colors]
other_colors = {model: cmap(idx) for model, idx in zip(other_models, available_indices)}


# Prepare figure with 2 rows (adjust the height of the second plot)
fig = plt.figure(figsize=(14.5, 7.5))

# Create a gridspec with two rows, where the second row is shorter
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # Height ratio: 3 for first, 1 for second

# --- First plot: Learning Rate Trajectories ---
ax1 = plt.subplot(gs[0])  # First subplot (larger one)
line_map = {}

time_range = np.arange(-5, 16)
xticks_positions = np.array([-5, 0, 5, 10, 15])

all_colors = {**manual_colors, **other_colors}


# Plot all models
for key in mod_names:
    avg = FM_lrs_avg_across_subjects_mod[key]
    baseline = avg[0:6].mean()
    avg_corrected = avg - baseline
    color = all_colors[key]
    linewidth = 1.5
    line = ax1.plot(time_range, avg_corrected, color=color, linestyle='-', linewidth=linewidth)[0]
    line_map[key] = line

# Plot subjects last
subject_curve = FM_avg_across_subjects_clip
subject_sem = FM_sem_across_subjects_clip
baseline_subject = subject_curve[0:6].mean()
subject_corrected = subject_curve - baseline_subject

subject_line = ax1.plot(time_range, subject_corrected, label="Subjects", color="k", linewidth=2)[0]
ax1.fill_between(time_range,
                 subject_corrected - subject_sem,
                 subject_corrected + subject_sem,
                 color="grey", alpha=0.3)

line_map["Subjects"] = subject_line


# Final plot setup for the first plot
legend_keys = ["Subjects"] + mod_names
legend_labels = [custom_labels.get(k, k) if k != "Subjects" else "Subjects" for k in legend_keys]
legend_lines = [line_map[k] for k in legend_keys]

ax1.axvline(0, color='black', linestyle='--', linewidth=1)
ax1.set_xticks(xticks_positions)
ax1.set_xlabel("Trial number relative to change point")
ax1.set_ylabel("Learning Rate (Baseline-substracted)")
ax1.set_title("Foucault & Meyniel (2024)")

ax1.legend(legend_lines, legend_labels, loc='upper right', ncol=3, frameon=False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


# Add "A" in bold at the top-left corner of the first plot
ax1.text(-0.05, 1.04, "A", transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')


# --- Second plot: Baseline Bar Plot ---
ax2 = plt.subplot(gs[1])  # Second subplot (shorter one)
baseline_values = {}

# Compute baselines for models
for key in mod_names:
    avg = FM_lrs_avg_across_subjects_mod[key]
    baseline = avg[0:6].mean()
    baseline_values[key] = baseline
    
    
# Compute subject baseline
subject_baseline = FM_avg_across_subjects_clip[0:6].mean()
baseline_values["Subjects"] = subject_baseline

# Add label and color for subject
custom_labels["Subjects"] = "Subjects"
all_colors["Subjects"] = "black"

# Put subject first in display
display_keys = ["Subjects"] + mod_names
baseline_heights = [baseline_values[k] for k in display_keys]
bar_colors = [all_colors[k] for k in display_keys]
bar_labels = [custom_labels[k] for k in display_keys]

# Plot baseline bar plot
ax2.bar(range(len(display_keys)), baseline_heights, color=bar_colors)
# ax2.set_xticks(range(len(display_keys)))
# ax2.set_xticklabels(bar_labels, rotation=45, ha='right')

ax2.set_xticks([])
ax2.set_xticklabels([])

ax2.set_ylabel("Baseline Learning Rate")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)



plt.savefig("figures/FigureS3A.png", dpi=300, bbox_inches='tight')
plt.show()
    
## S3B

time_range = np.arange(-5, 31) 
xticks_positions = np.array([-5,0,5,10,15,20,25,30])


fig = plt.figure(figsize=(14.5, 7.5))

# Create a gridspec with two rows, where the second row is shorter
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # Height ratio: 3 for first, 1 for second

# --- First plot: Learning Rate Trajectories ---
ax1 = plt.subplot(gs[0])  # First subplot (larger one)
line_map = {}

# Plot all models
for key in mod_names:
    avg = G_lrs_avg_across_subjects_mod[key]
    baseline = avg[0:6].mean()
    avg_corrected = avg - baseline
    color = all_colors[key]
    linewidth = 1.5
    line = ax1.plot(time_range, avg_corrected, color=color, linestyle='-', linewidth=linewidth)[0]
    line_map[key] = line


# Plot subjects last
subject_curve = G_avg_across_subjects_clip
subject_sem = G_sem_across_subjects_clip
baseline_subject = subject_curve[0:6].mean()
subject_corrected = subject_curve - baseline_subject

subject_line = ax1.plot(time_range, subject_corrected, label="Subjects", color="k", linewidth=2)[0]
ax1.fill_between(time_range,
                 subject_corrected - subject_sem,
                 subject_corrected + subject_sem,
                 color="grey", alpha=0.3)

line_map["Subjects"] = subject_line


# Final plot setup for the first plot
legend_keys = ["Subjects"] + mod_names
legend_labels = [custom_labels.get(k, k) if k != "Subjects" else "Subjects" for k in legend_keys]
legend_lines = [line_map[k] for k in legend_keys]

ax1.axvline(0, color='black', linestyle='--', linewidth=1)
ax1.set_xticks(xticks_positions)
ax1.set_xlabel("Trial number relative to change point")
ax1.set_ylabel("Learning Rate (Baseline-substracted)")
ax1.set_title("Gallistel et al. (2014)")

ax1.legend(legend_lines, legend_labels, loc='upper right', ncol=3, frameon=False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


ax1.text(-0.05, 1.04, "B", transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')


# --- Second plot: Baseline Bar Plot ---
ax2 = plt.subplot(gs[1])  # Second subplot (shorter one)
baseline_values = {}

# Compute baselines for models
for key in mod_names:
    avg = G_lrs_avg_across_subjects_mod[key]
    baseline = avg[0:6].mean()
    baseline_values[key] = baseline

# Compute subject baseline
subject_baseline = G_avg_across_subjects_clip[0:6].mean()
baseline_values["Subjects"] = subject_baseline

# Add label and color for subject
custom_labels["Subjects"] = "Subjects"
all_colors["Subjects"] = "black"

# Put subject first in display
display_keys = ["Subjects"] + mod_names
baseline_heights = [baseline_values[k] for k in display_keys]
bar_colors = [all_colors[k] for k in display_keys]
bar_labels = [custom_labels[k] for k in display_keys]

# Plot baseline bar plot
ax2.bar(range(len(display_keys)), baseline_heights, color=bar_colors)
# ax2.set_xticks(range(len(display_keys)))
# ax2.set_xticklabels(bar_labels, rotation=45, ha='right')

ax2.set_xticks([])
ax2.set_xticklabels([])

ax2.set_ylabel("Baseline Learning Rate")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)



plt.savefig("figures/FigureS3B.png", dpi=300, bbox_inches='tight')
plt.show()


## S3C

fig = plt.figure(figsize=(14.5, 7.5))

# Create a gridspec with two rows, where the second row is shorter
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # Height ratio: 3 for first, 1 for second

# --- First plot: Learning Rate Trajectories ---
ax1 = plt.subplot(gs[0])  # First subplot (larger one)
line_map = {}

# Plot all models
for key in mod_names:
    avg = Khaw_lrs_avg_across_subjects_mod[key]
    baseline = avg[0:6].mean()
    avg_corrected = avg - baseline
    color = all_colors[key]
    linewidth = 1.5
    line = ax1.plot(time_range, avg_corrected, color=color, linestyle='-', linewidth=linewidth)[0]
    line_map[key] = line

# Plot subjects last
subject_curve = Khaw_avg_across_subjects_clip
subject_sem = Khaw_sem_across_subjects_clip
baseline_subject = subject_curve[0:6].mean()
subject_corrected = subject_curve - baseline_subject

subject_line = ax1.plot(time_range, subject_corrected, label="Subjects", color="k", linewidth=2)[0]
ax1.fill_between(time_range,
                 subject_corrected - subject_sem,
                 subject_corrected + subject_sem,
                 color="grey", alpha=0.3)

line_map["Subjects"] = subject_line

# Final plot setup for the first plot
legend_keys = ["Subjects"] + mod_names
legend_labels = [custom_labels.get(k, k) if k != "Subjects" else "Subjects" for k in legend_keys]
legend_lines = [line_map[k] for k in legend_keys]

ax1.axvline(0, color='black', linestyle='--', linewidth=1)
ax1.set_xticks(xticks_positions)
ax1.set_xlabel("Trial number relative to change point")
ax1.set_ylabel("Learning Rate (Baseline-substracted)")
ax1.set_title("Khaw et al. (2017)")

ax1.legend(legend_lines, legend_labels, loc='upper right', ncol=3, frameon=False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)



ax1.text(-0.05, 1.04, "C", transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')



# --- Second plot: Baseline Bar Plot ---
ax2 = plt.subplot(gs[1])  # Second subplot (shorter one)
baseline_values = {}

# Compute baselines for models
for key in mod_names:
    avg = Khaw_lrs_avg_across_subjects_mod[key]
    baseline = avg[0:6].mean()
    baseline_values[key] = baseline

# Compute subject baseline
subject_baseline = Khaw_avg_across_subjects_clip[0:6].mean()
baseline_values["Subjects"] = subject_baseline

# Add label and color for subject
custom_labels["Subjects"] = "Subjects"
all_colors["Subjects"] = "black"

# Put subject first in display
display_keys = ["Subjects"] + mod_names
baseline_heights = [baseline_values[k] for k in display_keys]
bar_colors = [all_colors[k] for k in display_keys]
bar_labels = [custom_labels[k] for k in display_keys]

# Plot baseline bar plot
ax2.bar(range(len(display_keys)), baseline_heights, color=bar_colors)
# ax2.set_xticks(range(len(display_keys)))
# ax2.set_xticklabels(bar_labels, rotation=45, ha='right')

ax2.set_xticks([])
ax2.set_xticklabels([])

ax2.set_ylabel("Baseline Learning Rate")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)



plt.savefig("figures/FigureS3C.png", dpi=300, bbox_inches='tight')
plt.show()


#%% Figure S4

mod_names = ["HMM", "Mixed_delta","PID"]

# Create an empty dictionary to store DataFrames for each model
G_cv_test_sum_dict = {}

# Loop through each model and load the corresponding CSV file into the dictionary
for key in mod_names:
    file_path = f"results/cv_MSE_by_subj_sess/G_{key}_cv_test_sum.csv"
    cv_test_sum_df = pd.read_csv(file_path)
    
    # Add the DataFrame to the dictionary with the model name as the key
    G_cv_test_sum_dict[key] = cv_test_sum_df
    
    
    
# Create an empty dictionary to store DataFrames for each model
K_cv_test_sum_dict = {}

# Loop through each model and load the corresponding CSV file into the dictionary
for key in mod_names:
    file_path = f"results/cv_MSE_by_subj_sess/K_{key}_cv_test_sum.csv"
    cv_test_sum_df = pd.read_csv(file_path)
    
    # Add the DataFrame to the dictionary with the model name as the key
    K_cv_test_sum_dict[key] = cv_test_sum_df



models = ["HMM", "Mixed_delta", "PID"]  
colors = ['tab:blue', 'tab:orange', 'tab:green']
sessions = np.arange(10)
bar_width = 0.25

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7), sharex=True)

# === G_dat ===
n_subjects_G = 10
all_means_G = []

for i, model in enumerate(models):
    df = G_cv_test_sum_dict[model]  # shape (10, 10)
    means = df.mean(axis=0)
    all_means_G.append(means)

    axes[0].bar(sessions + i * bar_width, means, width=bar_width,
                label=model, color=colors[i], edgecolor='black')

all_means_G = np.array(all_means_G)

# for s in range(10):
#     min_idx = np.argmin(all_means_G[:, s])
#     min_height = all_means_G[min_idx, s]
#     bar_x = sessions[s] + min_idx * bar_width
#     axes[0].text(bar_x, min_height + 0.004, '*', ha='center', va='bottom', fontsize=14, fontweight='bold')

axes[0].set_ylabel('Mean MSE')
axes[0].set_title('Gallistel et al. (2014)')
axes[0].set_ylim(0, 0.17)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[0].legend(["HMM", "Mixture of delta rules", "PID"])

# === K_dat ===
n_subjects_K = 11
all_means_K = []

for i, model in enumerate(models):
    df = K_cv_test_sum_dict[model]  # shape (11, 10)
    means = df.mean(axis=0)
    all_means_K.append(means)

    axes[1].bar(sessions + i * bar_width, means, width=bar_width,
                label=model, color=colors[i], edgecolor='black')

all_means_K = np.array(all_means_K)

# for s in range(10):
#     min_idx = np.argmin(all_means_K[:, s])
#     min_height = all_means_K[min_idx, s]
#     bar_x = sessions[s] + min_idx * bar_width
#     axes[1].text(bar_x, min_height + 0.004, '*', ha='center', va='bottom', fontsize=14, fontweight='bold')

axes[1].set_xlabel('Session')
axes[1].set_ylabel('Mean MSE')
axes[1].set_title('Khaw et al. (2017)')
axes[1].set_xticks(sessions + bar_width)
axes[1].set_xticklabels([str(s) for s in range(10)])
axes[1].set_ylim(0, 0.17)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("figures/FigureS4.png", dpi=300, bbox_inches='tight')
plt.show()

    



