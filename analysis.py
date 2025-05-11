
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


from models.Gallistel_2014 import ChangePointModel 
from models.HMM_fit import HMM

from models.delta_learning import delta_rule,pearce_hall

from models.Nassar_et_al_2010 import reduced_bayes_lamda,reduced_bayes
from models.PID import PID
from models.Mixed_delta import mixed_delta
from models.VKF import VKF
from models.HGF import HGF
#%% optimised pararmeters

G_optimize_results =  {}

G_paras = {}

G_MSE= {}


K_optimize_results = {}
K_paras= {}
K_MSE= {}

FM_optimize_results = {}
FM_paras = {}

FM_MSE= {}


mod_names = ["HMM", "changepoint", "delta_rule", "p_hall","reduced_bayes",
"reduced_bayes_lamda", "PID" , "Mixed_delta", "VKF", "HGF"]


G_subj_n = 10
G_all_sess = list(range(10))

K_subj_n = 11
K_all_sess = list(range(10))

FM_subj_n = 96
FM_all_sess = list(range(15))

for key in mod_names:

    model_idx = key
    
    G_optimize_results[key] = []
    G_paras[key] = []
    G_MSE[key] = []
    
    K_optimize_results[key] = []
    K_paras[key] = []
    K_MSE[key] = []
    
    FM_optimize_results[key] = []
    FM_paras[key] = []
    FM_MSE[key] = []
    
  
    t_start_optimised = time()
    
    for subji in range(G_subj_n):
    
        result, x_min, fval = da.fit_model(1, model_idx, subji, G_all_sess)
    
        
        G_optimize_results[key].append(result)
        G_paras[key].append(x_min)
        G_MSE[key].append(fval)
    
    
    
    for subji in range(K_subj_n):
    
        result, x_min, fval = da.fit_model(2, model_idx,subji , K_all_sess)
    
        K_optimize_results[key].append(result)
        K_paras[key].append(x_min)
        K_MSE[key].append(fval)
     
    
    for subji in range(FM_subj_n):
    
        result, x_min, fval = da.fit_model(3, model_idx, subji , FM_all_sess)
    
        FM_optimize_results[key].append(result)
        FM_paras[key].append(x_min)
        FM_MSE[key].append(fval)
        
    
    t_end_optimised = (time() - t_start_optimised) / 60
    print(f"DONE IN {key} OPTIMISATION {t_end_optimised:0.3f} MIN.")


# store results
da.save_optimized_results([G_paras, K_paras, FM_paras], ['G', 'K', 'FM'])



#%% CV-MSE

G_dat = da.load_data("Gallistel")
Khaw_dat = da.load_data("Khaw")
FM_dat = da.load_data("FM")



def leave_one_out_splits(num_sessions):
    return [[j for j in range(num_sessions) if j != i] for i in range(num_sessions)]

G_leave_one_out_sess = leave_one_out_splits(len(G_all_sess))
K_leave_one_out_sess = leave_one_out_splits(len(K_all_sess))
FM_leave_one_out_sess = leave_one_out_splits(len(FM_all_sess))

    
  
mod_names = ["HMM", "changepoint", "delta_rule", "p_hall", "reduced_bayes",
             "reduced_bayes_lamda", "PID", "Mixed_delta", "VKF", "HGF"]



G_cv_paras = {}

G_cv_test= {}
G_cv_test_sum = {}


K_cv_paras = {}

K_cv_test= {}
K_cv_test_sum = {}



FM_cv_paras = {}

FM_cv_test= {}
FM_cv_test_sum = {}



for key in mod_names:
    
    
    G_cv_paras[key] = []

    G_cv_test[key] = []
    G_cv_test_sum[key] = []
    

    K_cv_paras[key] = []

    K_cv_test[key] = []
    K_cv_test_sum[key] = []
    
    

    FM_cv_paras[key] = []

    FM_cv_test[key] = []
    FM_cv_test_sum[key] = []
    
    
    model_idx = key


    
    t_start_cv = time()
    
    
    # training
    # Loop through subjects
    for subji in range(G_subj_n):
        
        expID = 1
        
        # Initialize lists for the current subject
      
        subject_paras = []

    
        # Loop through leave-one-out sessions
        for sessi in range(len(G_leave_one_out_sess)):
                
            result, x_min, fval = da.fit_model(expID, model_idx,subji, G_leave_one_out_sess[sessi])
    
            # Store the results for the current session
     
            subject_paras.append(x_min)

    
        # Append the subject-specific results to the main lists
       
        G_cv_paras[key].append(subject_paras)

        
    
    # testing
    
    for subji in range(G_subj_n):
        
        subj_cv_test = []
    
        for testi in range(len(G_leave_one_out_sess)):
            
            data = {}
            data['p_true'] = np.array(G_dat['true_p'][subji][testi])
            data['outcomes'] = np.array(G_dat['outcome'][subji][testi])
            data['slider_value'] = np.array(G_dat['sub_est'][subji][testi])
    
            paras = np.array(G_cv_paras[key][subji][testi])
        
            sess_MSE = da.MSE_fun(paras,data,expID,model_idx)
            subj_cv_test.append(sess_MSE)
            
        G_cv_test[key].append(subj_cv_test)
    


    for subji in range(len(G_cv_test[key])):  # Iterate over subjects
        subject_sum = sum(G_cv_test[key][subji])  # Sum MSEs across sessions for this subject
        G_cv_test_sum[key].append(subject_sum)   # Append to the dictionary
    
    
    
    # Loop through subjects
    for subji in range(K_subj_n):
        
        expID = 2
        # Initialize lists for the current subject

        subject_paras = []

    
        # Loop through leave-one-out sessions
        for sessi in range(len(K_leave_one_out_sess)):
            # Fit model for current subject and sessions
            result, x_min, fval = da.fit_model(expID, model_idx,subji, K_leave_one_out_sess[sessi])
    
            # Store the results for the current session

            subject_paras.append(x_min)

    
        # Append the subject-specific results to the main lists
        K_cv_paras[key].append(subject_paras)

    
    
    for subji in range(K_subj_n):
        
        subj_cv_test = []
    
        for testi in range(len(K_leave_one_out_sess)):
          
            
            data = {}
            data['p_true'] = np.array(Khaw_dat["true_p"][subji][testi])
            data['outcomes'] = np.array(Khaw_dat["outcome"][subji][testi])
            data['slider_value'] = np.array(Khaw_dat["sub_est"][subji][testi])
        
            paras = np.array(K_cv_paras[key][subji][testi])
        
            sess_MSE = da.MSE_fun(paras,data,expID,model_idx)
            
            subj_cv_test.append(sess_MSE)
            
        K_cv_test[key].append(subj_cv_test)
    
     
    for subji in range(len(K_cv_test[key])):  # Iterate over subjects
        subject_sum = sum(K_cv_test[key][subji])  # Sum MSEs across sessions for this subject
        K_cv_test_sum[key].append(subject_sum)   # Append to the dictionary
    
    
    # Loop through subjects
    for subji in range(FM_subj_n):
        
        expID = 3
        # Initialize lists for the current subject

        subject_paras = []

    
        # Loop through leave-one-out sessions
        for sessi in range(len(FM_leave_one_out_sess)):
            # Fit model for current subject and sessions
            result, x_min, fval = da.fit_model(expID, model_idx,subji, FM_leave_one_out_sess[sessi])
    
            # Store the results for the current session

            subject_paras.append(x_min)

    
        # Append the subject-specific results to the main lists
       
        FM_cv_paras[key].append(subject_paras)

        
    
    for subji in range(FM_subj_n):
        
        subj_cv_test = []
    
        for testi in range(len(FM_leave_one_out_sess)):
            
            data = {}
            data['p_true'] = np.array(FM_dat['true_p'][subji][testi])
            data['outcomes'] = np.array(FM_dat['outcome'][subji][testi])
            data['slider_value'] = np.array(FM_dat['sub_est'][subji][testi])
            
            paras = np.array(FM_cv_paras[key][subji][testi])
           
            sess_MSE = da.MSE_fun(paras,data,expID,model_idx)
            
            subj_cv_test.append(sess_MSE)
            
        FM_cv_test[key].append(subj_cv_test)
        
    for subji in range(len(FM_cv_test[key])):  # Iterate over subjects
        subject_sum = sum(FM_cv_test[key][subji])  # Sum MSEs across sessions for this subject
        FM_cv_test_sum[key].append(subject_sum)   # Append to the dictionary
    
        
        
    t_end_cv = (time() - t_start_cv) / 60
    print(f"DONE IN {key} CV {t_end_cv:0.3f} MIN.")


# store cv reuslts 

mod_names = ["HMM", "changepoint", "delta_rule", "p_hall","reduced_bayes",
                  "reduced_bayes_lamda", "PID" , "Mixed_delta", "VKF", "HGF"]


datasets = {
    "G": (G_cv_test, G_cv_test_sum, G_cv_paras),
    "K": (K_cv_test, K_cv_test_sum, K_cv_paras),
    "FM": (FM_cv_test, FM_cv_test_sum, FM_cv_paras),
}


model_param_headers = {
    "HMM": ["p_c"],
    "changepoint": ["T1", "T2"],
    "delta_rule": ["lr"],
    "p_hall": ["lr", "weight"],
    "reduced_bayes": ["p_c"],
    "reduced_bayes_lamda": ["pc", "lamda"],
    "PID": ["Kp", "Ki", "Kd", "lamda"],
    "Mixed_delta": ["delta1", "delta2", "hrate", "nu_p"],
    "VKF": ["lamda", "omega", "v0"],
    "HGF": ["nu", "kappa", "omega"]
}


# Loop through each dataset
for dataset_name, (cv_test, cv_test_sum, cv_paras) in datasets.items():
    for key in mod_names:  # Loop through models
        # Save cv_test as CSV
        cv_test_df = pd.DataFrame(cv_test[key])
        cv_test_filename = f"results/cv_MSE/{dataset_name}_cv_test_{key}.csv"
        cv_test_df.to_csv(cv_test_filename, index=False)
       
        
        # Save cv_test_sum as CSV
        cv_test_sum_df = pd.DataFrame({key: cv_test_sum[key]})
        cv_test_sum_filename = f"results/cv_MSE/{dataset_name}_cv_test_sum_{key}.csv"
        cv_test_sum_df.to_csv(cv_test_sum_filename, index=False)
       
        # Save cv_paras with headers
        # Get correct headers based on dataset name
       
        headers = model_param_headers.get(
                key, [f"param_{i}" for i in range(len(cv_paras[key][0][0]))])
        
        # Flatten parameters and add subject index
        flattened_paras = []
        subject_indices = []
        for subj_idx, subject_paras in enumerate(cv_paras[key]):
            for param_set in subject_paras:
                flattened_paras.append(param_set)
                subject_indices.append(subj_idx)

        # Create DataFrame with subject indices and parameter values
        subject_paras_df = pd.DataFrame(flattened_paras, columns=headers)
        subject_paras_df.insert(0, "subject_idx", subject_indices)

        # Save as CSV
        cv_paras_filename = f"results/cv_MSE/{dataset_name}_cv_paras_{key}.csv"
        subject_paras_df.to_csv(cv_paras_filename, index=False, header=True)

#%% model recovery 
    


mod_names = ["HMM", "changepoint", "delta_rule", "p_hall","reduced_bayes",
"reduced_bayes_lamda", "PID" , "Mixed_delta", "VKF", "HGF"]


leave_one_out_sess = []

num_sessions = 10

# leave one out
for i in range(num_sessions):
    # Create a list of sessions excluding the current one (i)
    excluded_sessions = [j for j in range(num_sessions) if j != i]
    leave_one_out_sess.append(excluded_sessions)


# Initialize result containers
mod_recov_results = {key: {
    "cv_optimize_results": {},
    "cv_paras": {},
    "cv_MSE": {},
    "cv_test": {},
    "cv_test_sum": {}
} for key in mod_names}



for key in mod_names:

    if key  == "HMM":
        seed = 223
    elif key  == "changepoint":
        seed = 224
    elif key  == "delta_rule":
        seed = 225
    elif key  == "p_hall":
       seed = 226
    elif key == "reduced_bayes":
        seed = 227
    elif key == "reduced_bayes_lamda":
        seed = 228
    elif key == "PID":
        seed = 229
    elif key == "Mixed_delta":
        seed = 230
    elif key == "VKF":
        seed = 231
    elif key == "HGF":
        seed = 232


    
    niter = 100
    n_sessions =150
    n_trials = 1000
    model = key
    
    
    # get stimulated paras
    
    mod_est, mod_paras, seqs, truep, seqs_index = da.stimulate_mod_data(model,niter,seed,n_sessions,n_trials,1)
    
    reorganized_mod_est = [np.array_split(array, 10) for array in mod_est]
    
    df_seqs_index = pd.DataFrame(seqs_index)
    
    # Save to CSV
    df_seqs_index.to_csv(f"results/mod_recovery/{model}_seqs_index.csv")  
    
    
    
    # for HGF to handle bad traj
    if model == "HGF":
        nan_indices = [i for i, lst in enumerate(mod_est) if np.isnan(lst).any()]
        
        # Save the indices to a text file
        np.savetxt("results/mod_recovery/HGF_nan_indices.txt", nan_indices, fmt="%d")
        
        df = pd.DataFrame({
            "nu": mod_paras[0],
            "kappa": mod_paras[1],
            "omega": mod_paras[2]})
        
        df.to_csv("results/mod_recovery/HGF_whole_in.csv")
        
        # Remove the lists in mod_est that correspond to the nan_indices
        mod_est = [lst for i, lst in enumerate(mod_est) if i not in nan_indices]

        # Remove the corresponding values from each list in mod_paras at nan_indices
        mod_paras = [np.delete(lst, nan_indices) for lst in mod_paras]
        
        seqs = [lst for i, lst in enumerate(seqs) if i not in nan_indices]
        
    
    mods = ["HMM", "changepoint", "delta_rule", "p_hall","reduced_bayes",
                     "reduced_bayes_lamda", "PID" , "Mixed_delta", "VKF", "HGF"]
    
    t_start_optimised = time()
    
    
    for mod in mods:
            
        model_idx = mod
        
        
        mod_recov_results[key]["cv_optimize_results"][mod] = []
        mod_recov_results[key]["cv_paras"][mod] = []
        mod_recov_results[key]["cv_MSE"][mod] = []
        mod_recov_results[key]["cv_test"][mod] = []
        mod_recov_results[key]["cv_test_sum"][mod] = []

        t_start_cv = time()
        
        for simu_i in range(len(seqs)):
            
            
            # Initialize lists for the current simulation
            simu_optimize_results = []
            simu_paras = []
            simu_MSE = []
            
            print(f"Mod_recovery_CV for {key}: simulatoin {simu_i} , fitting {mod}")
            
            # Loop through leave-one-out sessions
            for sessi in leave_one_out_sess:
            
                simu_seqs = [seqs[simu_i][sess] for sess in sessi]
                data_seq = np.concatenate(simu_seqs).flatten()   
                
                
                stimulation_results = [reorganized_mod_est[simu_i][sess] for sess in sessi]
                mod_stim = np.concatenate(stimulation_results).flatten()
                
                
                result, x_min, fval = da.modr_get_optimised_result(model_idx,data_seq,mod_stim)
        
                # Store the results for the current session
                simu_optimize_results.append(result)
                simu_paras.append(x_min)
                simu_MSE.append(fval) 
        
        
        
            # Append the subject-specific results to the main lists
            mod_recov_results[key]["cv_optimize_results"][mod].append(simu_optimize_results)
            mod_recov_results[key]["cv_paras"][mod].append(simu_paras)
            mod_recov_results[key]["cv_MSE"][mod].append(simu_MSE)
            
            
         # testing
         
        for simu_i in range(len(seqs)):
             
             simu_cv_test = []
         
             for testi in range(len(leave_one_out_sess)):
                 
                 
         
                 paras = np.array(mod_recov_results[key]["cv_paras"][mod][simu_i][testi])
             
                # Define the testing data for each session
                 
                 data_seq_test = seqs[simu_i][testi]
            
                 
                 mod_stim_test = reorganized_mod_est[simu_i][testi]
             
                
                 sess_MSE = da.modr_MSE_fun(paras,data_seq_test,mod_stim_test,model_idx)
                 simu_cv_test.append(sess_MSE)
                 
             mod_recov_results[key]["cv_test"][mod].append(simu_cv_test)
         
            


        for simu_i in range(len(mod_recov_results[key]["cv_test"][mod])):  
             simu_sum = sum(mod_recov_results[key]["cv_test"][mod][simu_i])  # Sum MSEs across sessions for this subject
             mod_recov_results[key]["cv_test_sum"][mod].append(simu_sum)   # Append to the dictionary
         
            
           

        t_end_cv = (time() - t_start_cv) / 60
        print(f"DONE IN mod_recovery_CV for {key}, fitting {mod}: {t_end_cv:0.3f} MIN.")
            
            
            
# store results    
        

# Define model parameter headers
model_param_headers = {
    "HMM": ["p_c"],
    "changepoint": ["T1", "T2"],
    "delta_rule": ["lr"],
    "p_hall": ["lr", "weight"],
    "reduced_bayes": ["p_c"],
    "reduced_bayes_lamda": ["p_c", "lamda"],
    "PID": ["Kp", "Ki", "Kd", "lamda"],
    "Mixed_delta": ["delta1", "delta2", "hrate", "nu_p"],
    "VKF": ["lamda", "omega", "v0"],
    "HGF": ["nu", "kappa", "omega"]
}

# Save results for each key and metric
mod_names = ["HMM", "changepoint", "delta_rule", "p_hall","reduced_bayes",
"reduced_bayes_lamda", "PID" , "Mixed_delta", "VKF", "HGF"]


for key in mod_names:  # Loop over datasets (`key`)

    # Save cv_test as CSV
    cv_test_rows = []
    for mod, tests in mod_recov_results[key]["cv_test"].items():
        for subj_idx, test_list in enumerate(tests):
            for sess_idx, test_value in enumerate(test_list):
                cv_test_rows.append({
                    "mod": mod,
                    "simu_idx": subj_idx,
                    "session_idx": sess_idx,
                    "cv_test_value": test_value
                })

    cv_test_df = pd.DataFrame(cv_test_rows)
    cv_test_filename = f"results/mod_recovery/cv_test/{key}_cv_test.csv"
    cv_test_df.to_csv(cv_test_filename, index=False)
    

    # Save cv_test_sum as CSV
    cv_test_sum_rows = []
    for mod, sums in mod_recov_results[key]["cv_test_sum"].items():
        for subj_idx, test_sum in enumerate(sums):
            cv_test_sum_rows.append({
                "mod": mod,
                "simu_idx": subj_idx,
                "cv_test_sum": test_sum
            })

    cv_test_sum_df = pd.DataFrame(cv_test_sum_rows)
    cv_test_sum_filename = f"results/mod_recovery/cv_test/{key}_cv_test_sum.csv"
    cv_test_sum_df.to_csv(cv_test_sum_filename, index=False)
    

    # Save cv_paras as CSV

    for mod, paras in mod_recov_results[key]["cv_paras"].items():
        # Get headers for parameters for the current mod
        headers = model_param_headers.get(mod, [f"param_{i}" for i in range(len(paras[0][0]))])
    
        # Prepare rows for the CSV
        cv_paras_rows = []
        for subj_idx, subject_paras in enumerate(paras):
            for sess_idx, param_set in enumerate(subject_paras):
                row = {
                    "simu_idx": subj_idx,
                    "session_idx": sess_idx  # Include session index
                }
                row.update({header: value for header, value in zip(headers, param_set)})
                cv_paras_rows.append(row)
    
        # Create DataFrame for this mod
        cv_paras_df = pd.DataFrame(cv_paras_rows)
    
        # Save to a CSV file
        cv_paras_filename = f"results/mod_recovery/cv_paras/{key}_{mod}_cv_paras.csv"
        cv_paras_df.to_csv(cv_paras_filename, index=False)
        
        
        
#%% parameters recovery

mod_names = ["HMM", "changepoint", "delta_rule", "p_hall","reduced_bayes",
"reduced_bayes_lamda", "PID" , "Mixed_delta", "VKF", "HGF"]




for key in mod_names:

    if key  == "HMM":
        seed = 123
    elif key  == "changepoint":
        seed = 124
    elif key  == "delta_rule":
        seed = 125
    elif key  == "p_hall":
       seed = 126
    elif key == "reduced_bayes":
        seed = 127
    elif key == "reduced_bayes_lamda":
        seed = 128
    elif key == "PID":
        seed = 129
    elif key == "Mixed_delta":
        seed = 130
    elif key == "VKF":
        seed = 131
    elif key == "HGF":
        seed = 132

    
    
    niter = 1000
    n_sessions =150
    n_trials = 1000
    model = key
    
    
    # get stimulated paras
    
    mod_est, mod_paras, seqs, truep, seqs_index = da.stimulate_mod_data(model,niter,seed,n_sessions,n_trials,2)
    
 
    
    df_seqs_index = pd.DataFrame(seqs_index)
    
    # Save to CSV
    df_seqs_index.to_csv(f"results/para_recovery/{model}_seqs_index.csv")  
    
    
    
    # for HGF to handle bad traj
    if model == "HGF":
        nan_indices = [i for i, lst in enumerate(mod_est) if np.isnan(lst).any()]
        
        # Save the indices to a text file
        np.savetxt("results/para_recovery/HGF_nan_indices.txt", nan_indices, fmt="%d")
        
        df = pd.DataFrame({
            "nu": mod_paras[0],
            "kappa": mod_paras[1],
            "omega": mod_paras[2]})
        
        df.to_csv("results/para_recovery/HGF_whole_in.csv")
        
        # Remove the lists in mod_est that correspond to the nan_indices
        mod_est = [lst for i, lst in enumerate(mod_est) if i not in nan_indices]

        # Remove the corresponding values from each list in mod_paras at nan_indices
        mod_paras = [np.delete(lst, nan_indices) for lst in mod_paras]
        
        seqs = [lst for i, lst in enumerate(seqs) if i not in nan_indices]
        
        
    
    
    # get optimised paras
    
    # opt_res  = pr.get_optimised_result(niter,model,seqs,mod_est)
    
    opt_res  = da.pr_get_optimised_result(len(mod_est),model,seqs,mod_est)
    
    if model  == "changepoint":
    
        result_dict = {
            "in": pd.DataFrame({
                "T1": mod_paras[0],
                "T2": mod_paras[1]
            }),
            "out": pd.DataFrame({
                "T1": [arr[0] for arr in opt_res],
                "T2": [arr[1] for arr in opt_res]
            })
        }
    
    elif model  == "HMM":
        result_dict = {
            "in": pd.DataFrame({
                "p_c": mod_paras[0]
            }),
            "out": pd.DataFrame({
                "p_c": [arr[0] for arr in opt_res]
            })
        }
    elif model  == "delta_rule":
        result_dict = {
            "in": pd.DataFrame({
                "lr": mod_paras[0]
            }),
            "out": pd.DataFrame({
                "lr": [arr[0] for arr in opt_res]
            })
        }
    elif model  == "p_hall":
        result_dict = {
            "in": pd.DataFrame({
                "lr": mod_paras[0],
                "w": mod_paras[1]
            }),
            "out": pd.DataFrame({
                "lr": [arr[0] for arr in opt_res],
                "w": [arr[1] for arr in opt_res]
            })
        }
    elif model  == "reduced_bayes":
        result_dict = {
            "in": pd.DataFrame({
                "p_c": mod_paras[0]
            }),
            "out": pd.DataFrame({
                "p_c": [arr[0] for arr in opt_res]
            })
        }
    elif model  == "reduced_bayes_lamda":
        result_dict = {
            "in": pd.DataFrame({
                "p_c": mod_paras[0],
                "lamda": mod_paras[1]
            }),
            "out": pd.DataFrame({
                "p_c": [arr[0] for arr in opt_res],
                "lamda": [arr[1] for arr in opt_res]
            })
        }
    elif model  == "PID":
        result_dict = {
            "in": pd.DataFrame({
                "Kp": mod_paras[0],
                "Ki": mod_paras[1],
                "Kd": mod_paras[2],
                "lamda": mod_paras[3]
            }),
            "out": pd.DataFrame({
                "Kp": [arr[0] for arr in opt_res],
                "Ki": [arr[1] for arr in opt_res],
                "Kd": [arr[2] for arr in opt_res],
                "lamda": [arr[3] for arr in opt_res]
            })
        }
        
    elif model  == "Mixed_delta": 
        result_dict = {
            "in": pd.DataFrame({
                "delta1": mod_paras[0],
                "delta2": mod_paras[1],
                "hRate": mod_paras[2],
                "nu_p": mod_paras[3]
            }),
            "out": pd.DataFrame({
                "delta1": [arr[0] for arr in opt_res],
                "delta2": [arr[1] for arr in opt_res],
                "hRate": [arr[2] for arr in opt_res],
                "nu_p": [arr[3] for arr in opt_res]
            })
        }
        
        
    elif model  == "VKF": 
        result_dict = {
            "in": pd.DataFrame({
                "lamda": mod_paras[0],
                "omega": mod_paras[1],
                "v0": mod_paras[2]
            }),
            "out": pd.DataFrame({
                "lamda": [arr[0] for arr in opt_res],
                "omega": [arr[1] for arr in opt_res],
                "v0": [arr[2] for arr in opt_res]
            })
        }
        
    elif model  == "HGF":
        result_dict = {
            "in": pd.DataFrame({
                "nu": mod_paras[0],
                "kappa": mod_paras[1],
                "omega": mod_paras[2]
            }),
            "out": pd.DataFrame({
                "nu": [arr[0] for arr in opt_res],
                "kappa": [arr[1] for arr in opt_res],
                "omega": [arr[2] for arr in opt_res]
            })
        }
            
      
    
    for key, df in result_dict.items():
        df.to_csv(f"results/para_recovery/{model}_{key}.csv", index=False)
    
    
        
#%% cv-MSE by session in Gallistel and Khaw dataset
      
    
def get_cv_folds(data, n_folds=5):
    
    n = len(data)
    base_fold_size = n // n_folds
    extra = n % n_folds  # leftovers to add to the last fold

    folds = []
    start = 0
    for i in range(n_folds):
        # last fold takes the extra trials
        end = start + base_fold_size + (1 if i == n_folds - 1 and extra > 0 else 0)
        test_idx = np.arange(start, end)
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        folds.append({'train_idx': train_idx, 'test_idx': test_idx})
        start = end

    return folds    

G_folds = get_cv_folds(G_dat["outcome"][0][0])

K_folds = get_cv_folds(Khaw_dat["outcome"][0][0])

    
  
mod_names = ["HMM", "PID", "Mixed_delta"]

G_cv_paras = {}

G_cv_test= {}
G_cv_test_sum = {}


K_cv_paras = {}

K_cv_test= {}
K_cv_test_sum = {}


for key in mod_names:
    
    
    G_cv_paras[key] = []

    G_cv_test[key] = []
    G_cv_test_sum[key] = []
    

    K_cv_paras[key] = []

    K_cv_test[key] = []
    K_cv_test_sum[key] = []
    
    
    
    
    model_idx = key


    
    t_start_cv = time()
    
    # training
    # Loop through subjects
    for subji in range(G_subj_n):
        
        print(f"cv - model: {key}, subj: {subji}")
        
        expID = 1
        
        # Initialize lists for the current subject
      
        subject_paras = []
        subject_cv_test = []

    
        # Loop through sessions
        for sessi in range(len(G_all_sess)):
            
            sess_paras = []
            
            sess_cv_test = []
            
            for trial_i in range(len(G_folds)):
                
                ## training
                
                train_index = G_folds[trial_i]['train_idx']
                
                outcome = G_dat["outcome"][subji][sessi][train_index]
                
                sub_est = G_dat["sub_est"][subji][sessi][train_index]
                
                
                result, x_min, fval = da.get_optimised_result_trial(model_idx,outcome,sub_est,expID)
        
                # Store the results for the current session
         
                sess_paras.append(x_min)
                
                ## testing
                
                test_index = G_folds[trial_i]['test_idx']
                
                paras = x_min
                
                data_seq = G_dat["outcome"][subji][sessi][test_index]
                estimate = G_dat["sub_est"][subji][sessi][test_index]
            
                sess_MSE = da.MSE_fun_trial(paras,data_seq,estimate,model_idx,expID)
                
                sess_cv_test.append(sess_MSE)
                
                
                

            subject_paras.append(sess_paras)
            subject_cv_test.append(sess_cv_test)
            

        # Append the subject-specific results to the main lists
       
        G_cv_paras[key].append(subject_paras)
        G_cv_test[key].append(subject_cv_test)
        
        # Sum across folds for each session
        subject_cv_test_sum = [np.sum(sess_cv) for sess_cv in subject_cv_test]

        # Store the subject's test MSE sums
        G_cv_test_sum[key].append(subject_cv_test_sum)
        
        
    print(f"cv - model: {key} finished")   
    
    # training
    # Loop through subjects
    for subji in range(K_subj_n):
        
        print(f"cv - model: {key}, subj: {subji}")
        
        expID = 2
        
        # Initialize lists for the current subject
      
        subject_paras = []
        subject_cv_test = []

    
        # Loop through sessions
        for sessi in range(len(K_all_sess)):
            
            sess_paras = []
            
            sess_cv_test = []
            
            for trial_i in range(len(K_folds)):
                
                ## training
                
                train_index = K_folds[trial_i]['train_idx']
                
                outcome = Khaw_dat["outcome"][subji][sessi][train_index]
                
                sub_est = Khaw_dat["sub_est"][subji][sessi][train_index]
                
                
                result, x_min, fval = da.get_optimised_result_trial(model_idx,outcome,sub_est,expID)
        
                # Store the results for the current session
         
                sess_paras.append(x_min)
                
                ## testing
                
                test_index = K_folds[trial_i]['test_idx']
                
                paras = x_min
                
                data_seq = Khaw_dat["outcome"][subji][sessi][test_index]
                estimate = Khaw_dat["sub_est"][subji][sessi][test_index]
            
                sess_MSE = da.MSE_fun_trial(paras,data_seq,estimate,model_idx,expID)
                
                sess_cv_test.append(sess_MSE)
                
                
                

            subject_paras.append(sess_paras)
            subject_cv_test.append(sess_cv_test)
            

        # Append the subject-specific results to the main lists
       
        K_cv_paras[key].append(subject_paras)
        K_cv_test[key].append(subject_cv_test)
        
        # Sum across folds for each session
        subject_cv_test_sum = [np.sum(sess_cv) for sess_cv in subject_cv_test]

        # Store the subject's test MSE sums
        K_cv_test_sum[key].append(subject_cv_test_sum)
        
    print(f"cv - model: {key} finished") 


## save data 

# Assuming G_cv_test and G_cv_test_sum have already been populated

for key in mod_names:
    # Prepare a list to store cv_test data for this model
    cv_test_data = []
    cv_test_sum_data = []
    
    # Loop through subjects and prepare the data
    for subji in range(G_subj_n):
        # For cv_test, create a row of session_fold for each subject
        cv_test_row = []
        
        for sessi in range(len(G_all_sess)):  # Loop through sessions
            for trial_i in range(len(G_folds)):  # Loop through folds
                # Get the MSE for current fold, session, and subject
                cv_test_row.append(G_cv_test[key][subji][sessi][trial_i])
        
        # For cv_test_sum, sum across folds for each session
        cv_test_sum_row = G_cv_test_sum[key][subji]
        
        # Add the rows to their respective lists
        cv_test_data.append(cv_test_row)
        cv_test_sum_data.append(cv_test_sum_row)
    
    # Convert the lists to DataFrames
    # For cv_test, the columns will be of the form "session_fold" (e.g., "0_0", "0_1", ...)
    columns_cv_test = [f"{sessi}_{fold}" for sessi in range(len(G_all_sess)) for fold in range(len(G_folds))]
    df_cv_test = pd.DataFrame(cv_test_data, columns=columns_cv_test)
    
    # For cv_test_sum, the columns will be the session number (e.g., "0", "1", ...)
    df_cv_test_sum = pd.DataFrame(cv_test_sum_data, columns=[str(sessi) for sessi in range(len(G_all_sess))])
    
    # Save the DataFrames to CSV
    df_cv_test.to_csv(f"results/cv_MSE_by_subj_sess/G_{key}_cv_test.csv", index=False)
    df_cv_test_sum.to_csv(f"results/cv_MSE_by_subj_sess/G_{key}_cv_test_sum.csv", index=False)      
    
print("saved results")


for key in mod_names:
    # Prepare a list to store cv_test data for this model
    cv_test_data = []
    cv_test_sum_data = []
    
    # Loop through subjects and prepare the data
    for subji in range(K_subj_n):
        # For cv_test, create a row of session_fold for each subject
        cv_test_row = []
        
        for sessi in range(len(K_all_sess)):  # Loop through sessions
            for trial_i in range(len(K_folds)):  # Loop through folds
                # Get the MSE for current fold, session, and subject
                cv_test_row.append(K_cv_test[key][subji][sessi][trial_i])
        
        # For cv_test_sum, sum across folds for each session
        cv_test_sum_row = K_cv_test_sum[key][subji]
        
        # Add the rows to their respective lists
        cv_test_data.append(cv_test_row)
        cv_test_sum_data.append(cv_test_sum_row)
    
    # Convert the lists to DataFrames
    # For cv_test, the columns will be of the form "session_fold" (e.g., "0_0", "0_1", ...)
    columns_cv_test = [f"{sessi}_{fold}" for sessi in range(len(K_all_sess)) for fold in range(len(K_folds))]
    df_cv_test = pd.DataFrame(cv_test_data, columns=columns_cv_test)
    
    # For cv_test_sum, the columns will be the session number (e.g., "0", "1", ...)
    df_cv_test_sum = pd.DataFrame(cv_test_sum_data, columns=[str(sessi) for sessi in range(len(K_all_sess))])
    
    # Save the DataFrames to CSV
    df_cv_test.to_csv(f"results/cv_MSE_by_subj_sess/K_{key}_cv_test.csv", index=False)
    df_cv_test_sum.to_csv(f"results/cv_MSE_by_subj_sess/K_{key}_cv_test_sum.csv", index=False)      
    

print("saved results")      






