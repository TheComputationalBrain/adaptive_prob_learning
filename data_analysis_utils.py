
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

from models.Gallistel_2014 import ChangePointModel, bern_cp_kl_fun
from models.HMM_fit import HMM,HMM_trial
from models import IdealObserver as IO

from models.delta_learning import delta_update,delta_rule,pearce_hall_update,pearce_hall

from models.Nassar_et_al_2010 import reduced_bayes_update,reduced_bayes,reduced_bayes_lamda_update,reduced_bayes_lamda
from models.PID import PID_update,PID
from models.Mixed_delta import mixed_delta,update_mixed_delta
from models.VKF import VKF, update_VKF
from models.HGF import HGF, update_HGF




#%%

def load_data(dat):
    """
    "FM" = Foucault& Meyniel (2024)
    "Gallistel" = Gallistel et al(2014)
    "Khaw" = Khaw et al(2017)
    """   
    
   
    
    if dat == "FM":
        
        FM_subj_n = 96
        FM_all_sess = list(range(15))
        
        data_outcomelevel = pd.read_csv('data/Foucault_Meyniel_2024/structured-dataset/ada-learn_study/data_outcome-level.csv')
        data_outcomelevel = data_outcomelevel[data_outcomelevel['task'] != 'ada-pos']

        # fix 604b169fe4b7991ec08da3a6
        index_range = np.arange(130725,131849+1)

        # Find the rows within the specified index range and replace the 'subject' column
        data_outcomelevel.loc[data_outcomelevel.index.isin(index_range), 'subject'] = '604b169fe4b7991ec08da3a7'


        data_outcomelevel_pliot = pd.read_csv('data/Foucault_Meyniel_2024/structured-dataset/ada-prob_study/data_outcome-level.csv')


        # This file describes the dataset of all the different outcome sequences
        # that were presented to subjects in our study
        # (100 sequences for ada-pos, 150 sequences for ada-prob).
        # For each subject, we randomly sampled from this dataset a subset of sequences
        # to present at each session (the sampling was performed without replacement at the subject-level,
        # but with replacement at the group level).
        # As a result, each outcome sequence was presented multiple times across the subjects.
        SEQ_DATA_FILE = pd.read_csv('data/Foucault_Meyniel_2024/seq-data/seq-data_pos-prob_ntrials-75_seed-1_nsessions-pos-100-prob-150_std-pos-10by300_pc-pos-1by10-prob-1by20_min-run-length-pos-3-prob-6_min-odd-change-prob-4_apply-min-for-last-run-length-prob_max-run-length-prob-60.csv')
        SEQ_DATA_FILE = SEQ_DATA_FILE[SEQ_DATA_FILE['taskName'] != 'ada-pos']



        FM_dat = {}
        FM_dat['outcome'] = []
        FM_dat['sub_est'] = []
        FM_dat['true_p'] = []
        FM_dat['update'] = []


        # Get unique subjects
        unique_subjects = data_outcomelevel['subject'].unique()

        # Loop through each subject and session
        for subj in unique_subjects:
            subj_outcomes = []  # To store sessions for each subject
            subj_est = []
            subj_true_p = []
            subj_update = []

            subject_data = data_outcomelevel[data_outcomelevel['subject'] == subj]  # Filter data for each subject
            unique_sessions = subject_data['session_idx'].unique()
            
            for sess in unique_sessions:
                session_outcome = subject_data[subject_data['session_idx'] == sess]['outcome'].tolist()
                subj_outcomes.append(session_outcome)

                estimate = subject_data[subject_data['session_idx'] == sess]['estimate'].tolist()
                subj_est.append(estimate)

                true_p = subject_data[subject_data['session_idx'] == sess]['hidden_parameter'].tolist()
                subj_true_p.append(true_p)
                
                # Compute the update flag for the current session directly
                update_flag_session = np.concatenate(([1], np.abs(np.diff(estimate)) > 0))
                subj_update.append(update_flag_session)
                
            
            FM_dat['outcome'].append(np.array(subj_outcomes))
            FM_dat['sub_est'].append(np.array(subj_est))
            FM_dat['true_p'].append(np.array(subj_true_p))
            FM_dat['update'].append(np.array(subj_update))
            
            
        
        
        
        return FM_dat
            
    elif dat == "Gallistel" :
               
            G_subj_n = 10
            G_all_sess = list(range(10))
           
                       
            # Load the .mat file
            mat_data = scipy.io.loadmat('data/GallistelPsychSci2014/Data.mat')
            
            # Assuming the data you need is stored under a specific key (e.g., 'data')
            # You may need to adjust the key depending on the structure of the .mat file
            data = mat_data['Data1']
            
            
            selected_columns = data[:, [1, 2, 3, 4, 8, 9]]
            
            # Define custom headers for the selected columns
            headers = ['subj', 'sess_idx', 'outcome', 'sub_estimate', 'true_p', 'p1_change']
            
            # Create a pandas DataFrame with the selected columns and headers
            Gallistel_2014 = pd.DataFrame(selected_columns, columns=headers)
            
            
            
            
            G_dat = {}
            G_dat['outcome'] = []
            G_dat['sub_est'] = []
            G_dat['true_p'] = []
            G_dat['update'] = []
            
            
            
            # Get unique subjects
            unique_subjects = Gallistel_2014['subj'].unique()
            
            # Loop through each subject and session
            for subj in unique_subjects:
                subj_outcomes = []  # To store sessions for each subject
                subj_est = []
                subj_true_p = []
                subj_update = []
                
            
                subject_data = Gallistel_2014[Gallistel_2014['subj'] == subj]  # Filter data for each subject
                unique_sessions = subject_data['sess_idx'].unique()
                
                for sess in unique_sessions:
                    session_outcome = subject_data[subject_data['sess_idx'] == sess]['outcome'].tolist()
                    subj_outcomes.append(session_outcome)
            
                    estimate = subject_data[subject_data['sess_idx'] == sess]['sub_estimate'].tolist()
                    subj_est.append(estimate)
            
                    true_p = subject_data[subject_data['sess_idx'] == sess]['true_p'].tolist()
                    subj_true_p.append(true_p)
                    
                    # Compute the update flag for the current session directly
                    update_flag_session = np.concatenate(([1], np.abs(np.diff(estimate)) > 0))
                    subj_update.append(update_flag_session)
                  
                
                
                G_dat['outcome'].append(np.array(subj_outcomes))
                G_dat['sub_est'].append(np.array(subj_est))
                G_dat['true_p'].append(np.array(subj_true_p))
                G_dat['update'].append(np.array(subj_update))             
                
            return G_dat
                
    elif dat == "Khaw":
        
        # Khaw et al 2017
        # 11 subjects, 10 sessions per subject, and 999 observations per session 
        # Out of 110 sessions, 91 had unique sequences of ring realizations and 19 were repetitions of one of these sequences. 

        
        K_subj_n = 11
        K_all_sess = list(range(10))
        
                
        khaw_prob = pd.read_csv('data/Khaw_2017_Data/probs.txt', delimiter='\t', header = None)
        khaw_prob 
        
        khaw_subj_est = pd.read_csv('data/Khaw_2017_Data/phats.txt', delimiter='\t', header = None)
        khaw_subj_est
        
        khaw_outcome = pd.read_csv('data/Khaw_2017_Data/obs.txt', delimiter='\t', header = None)
        
        
        khaw_subj = [khaw_subj_est[col].to_numpy() for col in khaw_subj_est.columns]
        khaw_prob = [khaw_prob[col].to_numpy() for col in khaw_prob.columns]
        
        khaw_out = [khaw_outcome[col].to_numpy() for col in khaw_outcome.columns]
        
        
        
        
        # Initialize the dictionary structure
        Khaw_dat = {
            "outcome": [],
            "sub_est": [],
            "true_p": [],
            "update": []
        }
        
        # Populate the dictionary with subject data
        for i in range(K_subj_n):
            # For each subject, reshape the data into (10, 999) arrays
            subj_outcome = np.array(khaw_out[i * 10:(i + 1) * 10])
            subj_sub_est = np.array(khaw_subj[i * 10:(i + 1) * 10])
            subj_true_p = np.array(khaw_prob[i * 10:(i + 1) * 10])
            
            # Reshape each subject's data into (10, 999) format
            Khaw_dat["outcome"].append(subj_outcome)
            Khaw_dat["sub_est"].append(subj_sub_est)
            Khaw_dat["true_p"].append(subj_true_p)
            
            # Calculate update flags for each session
            update_flags = []
            for session in subj_sub_est:
                session_update_flag = np.concatenate(([1], np.abs(np.diff(session)) > 0))
                update_flags.append(session_update_flag)
            
           
            Khaw_dat["update"].append(np.array(update_flags))
        
        return Khaw_dat

        





  
def learning_rate(outcome,sub_est):
    
    lr = np.zeros(len(outcome))
    num_trial = len(outcome)
    
    lr[0] = np.nan
    
    for t in range(1,num_trial):
    
        prev_estimate = sub_est[t-1] 
        estimate = sub_est[t]
        
        nom = estimate - prev_estimate
        denom = outcome[t] - prev_estimate
     
        
        if denom == 0:
            lr[t] = np.nan
        else:
            lr[t] = nom / denom
             
    
    return lr


def learning_rate_clip(outcome,sub_est):
    
    lr = np.zeros(len(outcome))
    num_trial = len(outcome)
    
    lr[0] = np.nan
    
    sub_est = np.clip(sub_est, 0.1, 0.9)
    
    for t in range(1,num_trial):
    
        prev_estimate = sub_est[t-1] 
        estimate = sub_est[t]
        
        nom = estimate - prev_estimate
        denom = outcome[t] - prev_estimate
     
        
        lr[t] = nom / denom
    
    return lr



#### model fitting 



def fit_model(expID, model, subjidx, sessions):
    """
    expID: 1 = Gallistel et al; 2 = Khaw et al; 3 = Foucault& Meyniel (2024)
    
    model: mod_names = ["HMM", "changepoint", "delta_rule", "p_hall","reduced_bayes",
                 "reduced_bayes_lamda", "PID","Mixed_delta","VKF", "HGF"]
    
    subjidx 
    
    sessions[list]
    """   
    
    # import data
    if expID == 1:
        
        
        G_dat = load_data("Gallistel")

            
        data = {}
        data['p_true'] = G_dat["true_p"][subjidx][sessions]
        data['outcomes'] = G_dat["outcome"][subjidx][sessions]
        data['slider_value'] = G_dat["sub_est"][subjidx][sessions]
        data['update_flag'] = G_dat["update"][subjidx][sessions]


            
    elif expID == 2:
        
        # Khaw et al 2017
        # 11 subjects, 10 sessions per subject, and 999 observations per session 
        # Out of 110 sessions, 91 had unique sequences of ring realizations and 19 were repetitions of one of these sequences. 

        K_dat = load_data("Khaw")

            
        data = {}
        data['p_true'] = K_dat["true_p"][subjidx][sessions]
        data['outcomes'] = K_dat["outcome"][subjidx][sessions]
        data['slider_value'] = K_dat["sub_est"][subjidx][sessions]
        data['update_flag'] = K_dat["update"][subjidx][sessions]
            
        
    
    elif expID == 3:
        
        FM_dat = load_data("FM")

            
        data = {}
        data['p_true'] = FM_dat["true_p"][subjidx][sessions]
        data['outcomes'] = FM_dat["outcome"][subjidx][sessions]
        data['slider_value'] = FM_dat["sub_est"][subjidx][sessions]
        data['update_flag'] = FM_dat["update"][subjidx][sessions]
        
    
        # concatenate seesions 
    data['p_true'] = np.concatenate(data['p_true'])
    data['outcomes'] = np.concatenate(data['outcomes'])
    data['slider_value'] = np.concatenate(data['slider_value'])
    data['update_flag'] = np.concatenate(data['update_flag'])

    
    
#### optimisation
    # best_result = None
    # best_x_min = None
    # best_fval = np.inf
    # n_run = 10
    
    # for run_i in range(n_run):
     
    print(f'ExpID: {expID}, Subject: {subjidx}, Model: {model}, Running optimization...')
            
    fp_init, bounds = get_initial_parameters(data,expID,model)  
            
    fp_init = np.array(fp_init)
        
            
    opt = so.minimize(MSE_fun, fp_init, args=(data,expID,model),
                                 method='Powell', bounds=bounds, options={'disp': False}) 
               
    fval = opt['fun']
    x_min = opt['x']
    result = opt["success"]
        
        # if fval < best_fval:
        #     best_fval = fval
        #     best_x_min = x_min
        #     best_result = result
            
    
    
    return result, x_min, fval 


# def likelihood_fun(pars,data,expID):
#     sigma_resp = pars[0] # response noise
    
#     p_resp_sum = np.zeros(len(data['slider_value']))
#     nruns = 100 
#     nbins = 100
#     binwidth = 1/nbins
#     lapse_rate = .001
    
#     for ii in range(nruns):
        
#         T1 = pars[1]
#         T2 = pars[2];
        
        
#         predicted_response = ChangePointModel(T1,T2,data['outcomes'],expID)
        
#         # Compute p(response | correct update prediction)
#         p_resp_correct_update_decision = norm.cdf(data['slider_value'] + binwidth / 2, 
#                                           loc= predicted_response, scale= sigma_resp) - \
#                                  norm.cdf(data['slider_value'] - binwidth / 2, 
#                                           loc= predicted_response, scale= sigma_resp)
                            
#         # Set p(response | incorrect update prediction)
#         p_resp_incorrect_update_decision = 0
        
#         # Determine predicted update flag for each trial
#         predicted_update_flag = np.concatenate(([1], np.abs(np.diff(predicted_response)) > 0))
    

#         # Compute probability of correct update decision for each trial
#         p_correct_update_decision = np.ones_like(data['update_flag'], dtype=float)
        
#         p_correct_update_decision[data['update_flag'] == 1] = predicted_update_flag[data['update_flag'] == 1]
#         p_correct_update_decision[data['update_flag'] == 0] = 1 - predicted_update_flag[data['update_flag'] == 0]

#         # Compute probability of subject responses given model predictions
#         p_resp_vec = (p_correct_update_decision * p_resp_correct_update_decision) + \
#                      ((1 - p_correct_update_decision) * p_resp_incorrect_update_decision)

#         # Incorporate random responses
#         p_resp_vec = (1 - lapse_rate) * p_resp_vec + lapse_rate * (1 / nbins)

#         # Add to p_resp_sum
#         p_resp_sum += p_resp_vec

    
#     # Marginalize
#     p_resp = p_resp_sum / nruns

#     # Compute combined log likelihood
#     LLH = np.sum(np.log(p_resp))
#     LLH = -LLH # negative log lik
    
    
#     return LLH



def MSE_fun(pars,data,expID,model):
    
      
    if model =="HMM":
       
        p_c = pars[0]
        
        predicted_response = HMM(p_c,data['outcomes'],expID)
        
    
    elif model == "changepoint":
        
        T1 = pars[0]
        T2 = pars[1]
                
                
        predicted_response = ChangePointModel(T1,T2,data['outcomes'],expID)
    
    elif model == "delta_rule":
       
       lr = pars[0]
       
       
       predicted_response = delta_rule(lr,data['outcomes'],expID)
       
    elif model == "p_hall":
       
       lr = pars[0]
       weight = pars[1]
       
       predicted_response = pearce_hall(lr,weight,data['outcomes'],expID)
       
    elif model == "reduced_bayes":
       p_c = pars[0]
       
       predicted_response = reduced_bayes(data['outcomes'],p_c,expID)
       
    elif model == "reduced_bayes_lamda":
        p_c = pars[0]
        lamda = pars[1]
        predicted_response = reduced_bayes_lamda(data['outcomes'],p_c,lamda,expID)
        
    elif model == "PID":
        Kp = pars[0]
        Ki = pars[1]
        Kd = pars[2]
        lamda = pars[3]
        predicted_response = PID(data['outcomes'],Kp,Ki,Kd,lamda,expID)
        
    elif model == "Mixed_delta":
        delta = pars[:-2]
        hrate = pars[-2]
        nu_p = pars[-1]
              
        predicted_response = mixed_delta(data['outcomes'],delta,hrate,nu_p,expID)
        
    elif model == "VKF": 
        lamda = pars[0]
        omega = pars[1]
        v0 = pars[2]
        
        predicted_response = VKF(data['outcomes'],lamda,omega,v0,expID)
        
    elif model == "HGF": 
        nu = pars[0]
        kappa = pars[1]
        omega = pars[2]
        
        predicted_response = HGF(data['outcomes'],nu,kappa,omega,expID)
        
        # if np.any(np.isnan(predicted_response)) or bad_traj == True:
        #     return 1e10  # High penalty value
    
    # Compute squared errors
    squared_errors = (predicted_response - data['slider_value']) ** 2
    
    # sum of squared error
    sse = np.sum(squared_errors)
    
    # mean squared error
    mse = sse / len(predicted_response)  
    
    # rmse = np.sqrt(mse)
        
    return mse
    
  


def get_initial_parameters(data,expID,model):
   
    if model =="HMM":
        
        # HMM: p_c
        lb = np.array([0.0001])
        ub = np.array([0.9])
        
        bounds = [(0.0001, 0.9)]
        
    elif model == "changepoint":
        # IIAB: T1, T2 
        lb = np.array([0.01, 0.001])
        ub = np.array([2, 20])
        
        bounds = [(0.01,2), (.001,20)]
       
        
    elif model == "delta_rule":
        # delta_rule: lr
        lb = np.array([0.0001])
        ub = np.array([0.9])
        
        bounds = [(0.0001, 0.9)]
        
    elif model == "p_hall":
        # Pearce-Hall: intial lr, weight
        lb = np.array([0.0001, 0])
        ub = np.array([0.9, 1])
        
        bounds = [(0.0001, 0.9), (0, 1)]
        
    elif model == "reduced_bayes":
        # reduced Bayes: p_c
        lb = np.array([0.0001])
        ub = np.array([0.9])       
        bounds = [(0.0001, 0.9)]
        
    elif model == "reduced_bayes_lamda":
        # reduced_bayes_lamda: p_c, lamda
        lb = np.array([0.0001, 0])
        ub = np.array([0.9, 1])
        
        bounds = [(0.0001, 0.9), (0, 1)]
        
    elif model == "PID":
        # PID: Kp,Ki,Kd,lamda
        lb = np.array([-1, -1, -1, 0])
        ub = np.array([1, 1, 1, 1])
        
        bounds = [(-1, 1), (-1, 1),(-1, 1),(0, 1)]
        
    elif model == "Mixed_delta":
        # Mixed_delta: node2, node3,hrate,nu_p
        lb = np.array([1.01, 1.01, 0, 0])
        ub = np.array([8, 17, 1, 5])
        
        bounds = [(1.01, 8), (1.01, 17),(0, 1),(0, 5)]
        
    
        
    elif model == "VKF":
        # VKF: lamda,omega,v0
        lb = np.array([0, 0, 0])
        ub = np.array([1, 1, 1])
        
        bounds = [(0,1), (0,1), (0, 1)]
        
    elif model == "HGF":
        # HGF: nu,kappa,omega
        lb = np.array([0.0001, 0.01, -5])
        ub = np.array([10, 1, 2])
        
        bounds = [(0.0001,10), (0.01,1), (-5, 2)]
        
        
    # best_LLH = -np.inf
    
    best_MSE = np.inf
    fp_init = None
    
    n_initial_guesses = 100
    
    for ii in range(n_initial_guesses):
        
        fp_random = lb + (ub - lb) * np.random.rand(len(lb))
        fp_random = np.maximum(np.minimum(fp_random, ub), lb)
        
        MSE = MSE_fun(fp_random, data,expID,model)
        
        if MSE < best_MSE:
            best_MSE = MSE
            fp_init = fp_random
        
        
        # # Compute likelihood for the current random parameters
        # LLH = likelihood_fun(fp_random, data,expID)
        # LLH = -LLH # change it back to log lik
        
        # # Check if this is the best likelihood so far
        # if LLH > best_LLH:
        #     best_LLH = LLH
        #     fp_init = fp_random
        
        

    return fp_init, bounds
                   
           
           
def save_optimized_results(paras_dict, dataset_name):
    """
    Save optimization results to CSV for a given parameter dictionary.
    """
        
    for i, key in enumerate(paras_dict):
        df = pd.DataFrame({
            'HMM_pc': [p[0] for p in key['HMM']],  
            'changepoint_T1': [p[0] for p in key['changepoint']],
            'changepoint_T2': [p[1] for p in key['changepoint']], 
            'delta_lr': [p[0] for p in key['delta_rule']],
            'p_hall_lr': [p[0] for p in key['p_hall']],
            'p_hall_w': [p[1] for p in key['p_hall']],         
            "reduced_bayes_pc": [p[0] for p in key["reduced_bayes"]],
            "reduced_bayes_lamda_pc": [p[0] for p in key["reduced_bayes_lamda"]],
            "reduced_bayes_lamda":[p[1] for p in key["reduced_bayes_lamda"]],
            "PID_kp":[p[0] for p in key["PID"]],
            "PID_ki":[p[1] for p in key["PID"]],
            "PID_kd":[p[2] for p in key["PID"]],
            "PID_lamda":[p[3] for p in key["PID"]],
            "mixed_delta_delta1":[p[0] for p in key["Mixed_delta"]],
            "mixed_delta_delta2":[p[1] for p in key["Mixed_delta"]],
            "mixed_delta_hrate":[p[2] for p in key["Mixed_delta"]],
            "mixed_delta_nu_p":[p[3] for p in key["Mixed_delta"]],
            "VKF_lamda":[p[0] for p in key["VKF"]],
            "VKF_omega":[p[1] for p in key["VKF"]],
            "VKF_v0":[p[2] for p in key["VKF"]],
            "HGF_nu":[p[0] for p in key["HGF"]],
            "HGF_kappa":[p[1] for p in key["HGF"]],
            "HGF_omega":[p[2] for p in key["HGF"]],
            
        })
        
       
        df.to_csv(f"results/optimisation/{dataset_name[i]}_opt_results.csv")
    

#### prob. weight function 

# Log-odds transformation function
def log_odds(x):
    return np.log(x / (1 - x))

# Define the LLO function (rewritten in log-odds form)
def llo_function_logodds(log_p, delta, gamma):
    return np.log(delta) + gamma * log_p


# Define the LLO function
def llo_function(p, delta, gamma):
    return (delta * (p / (1 - p))**gamma) / (1 + delta * (p / (1 - p))**gamma)


#### model recovery

## gerneate sequences 


P_C = 1/200
P1_MIN = 0
P1_MAX = 1
MIN_RUN_LENGTH = 6
MAX_RUN_LENGTH = 250
MIN_ODD_CHANGE = 4

# Adapted from NeuralProb/utilities/utils.py
def generate_p1s_sequence(n_trials, p_c,
    p1_min=P1_MIN, # minimum value for probabilities
    p1_max=P1_MAX, # maximum value for probabilities
    min_run_length=MIN_RUN_LENGTH, # minimum length for a stable period
                                   # /!\ NOTE: this minimum does not effectively apply
                                   # for the last stable period at the end of the sequence
                                   # (this is desirable)
    max_run_length=MAX_RUN_LENGTH, # maximum length for a stable period
    min_odd_change=MIN_ODD_CHANGE, # minimum change in odd at a change point 
    apply_min_for_last_run_length=False,
    ):
    """
    This function generates a "jumping" probability sequence of length n_trials.
    The probability of a jump in the probability at any trial is p_c.
    """
    L = n_trials
    LJumpMax = max_run_length # maximum length for a stable period
    MinOddChange = min_odd_change # fold change (from pre to post jump) of odd for at least 1 transition
    pMin = p1_min # minimum value for probabilities
    pMax = p1_max # maximum value for probabilities
    """
    Define jumps occurrence with a geometric law of parameter p_c
    CDF = 1-(1-p_c)^k -> k = log(1-CDF)/log(1-p_c)
    The geometric distribution can be sampled uniformly from CDF.
    """
    SubL = []
    while sum(SubL) < L:
        if (apply_min_for_last_run_length and ((L - sum(SubL)) < min_run_length)):
            # restart from scratch as the last run length cannot be above the minimum
            SubL = [] 
        RandGeom = None
        while (RandGeom is None
            or RandGeom > max_run_length # note the change from >= to > from the original code
            or RandGeom < min_run_length):
            if p_c > 0:
                RandGeom = round(np.log(1-np.random.rand()) / np.log(1-p_c))
            else:
                assert L <= max_run_length
                RandGeom = L
        SubL.append(RandGeom)

    # Define probabilities
    tmp_p1 = np.zeros(len(SubL))

    for kJump in range(len(SubL)):
        if kJump == 0:
            tmp_p1[0] = np.random.rand()*(pMax-pMin) + pMin
        else:
            while True:
                tmp_p1[kJump] = np.random.rand()*(pMax-pMin) + pMin

                # compute transition odd change from pre to post jump
                oddChange = ((tmp_p1[kJump-1])/(1-tmp_p1[kJump-1])) /\
                    ((tmp_p1[kJump])/(1-tmp_p1[kJump]))

                # take this value if the odd change is sufficiently large
                if abs(np.log(oddChange)) > np.log(min_odd_change):
                    break

    # assign probs value to each trial
    p1 = np.zeros(L)
    p1[0:int(SubL[0])] = tmp_p1[0]
    for kJump in range(1, len(SubL)):
        p1[int(sum(SubL[0:int(kJump-1)])):int(sum(SubL[0:kJump]))] = tmp_p1[kJump-1] # this was kJump in the original code
    p1[int(sum(SubL[0:-1])):] = tmp_p1[-1]

    change_trials = np.cumsum(SubL[:-1], dtype=int)

    return p1, change_trials




def random_outcomes_from_p1s(p1s):
    """
    Return a random sequence of binary outcomes using the given sequence of probabilities.
    Each value is sampled from a Bernoulli distribution. The probability represents
    the probability of this value being equal to 1.
    The input and output sequences are numpy array.
    """
    runif = np.random.random_sample(p1s.shape)
    seq = np.where((runif > p1s), 0, 1).astype(int)
    return seq



def gen_sequences(n_sessions,n_trials,seed,model,mode):

    model = model
    n_sessions = n_sessions
    n_trials = n_trials
    p_c = P_C
    min_odd_change = MIN_ODD_CHANGE
    min_run_length =  MIN_RUN_LENGTH
    max_run_length =  MAX_RUN_LENGTH
    apply_min_for_last_run_length = False
            
            
    
    p1s = np.zeros((n_sessions, n_trials))
    did_p1_change = np.zeros_like(p1s, dtype=bool)
    did_p1_change[:, 0] = True
    n_change_points = 0
    
    np.random.seed(seed)
    
    for i_sess in range(n_sessions):
        p1s_sess, change_trials = generate_p1s_sequence(n_trials, p_c,
        min_odd_change=min_odd_change, min_run_length=min_run_length,
        max_run_length=max_run_length,
        apply_min_for_last_run_length=apply_min_for_last_run_length)
        p1s[i_sess] = p1s_sess
        did_p1_change[i_sess, change_trials] = True
        n_change_points += len(change_trials)
        outcomes = random_outcomes_from_p1s(p1s)
    
    
            
    print("p1s", p1s)
    print("outcomes", outcomes)
    
    print("total num. of change points", n_change_points)
    print("avg num. of change points", n_change_points / n_sessions)
    
    session_indices = np.repeat(np.arange(n_sessions)[:, np.newaxis], n_trials, axis=1)
    trial_indices = np.repeat(np.arange(n_trials)[np.newaxis, :], n_sessions, axis=0)
    data_dict = {
                "nSessions": n_sessions,
                "nTrials": n_trials,
                "pC": p_c,
                "p1Min": P1_MIN,
                "p1Max": P1_MAX,
                "minOddChange": min_odd_change,
                "minRunLength": min_run_length,
                "maxRunLength": max_run_length,
                "applyMinForLastRunLength": apply_min_for_last_run_length,
                "sessionIndex": session_indices,
                "trialIndex": trial_indices,
                "p1": p1s,
                "didP1Change": did_p1_change,
                "outcome": outcomes           
            }
    
    
    
           
    data_dict_df = {
                    key: (value.reshape(-1) if type(value) == np.ndarray
                        else value) for (key, value) in data_dict.items()
                }
                # print(data_dict_df)
    data_df = pd.DataFrame.from_dict(data_dict_df)
    
    if mode == 1:
    
        data_df.to_csv(f"results/mod_recovery/{model}_gen_sequences.csv")
    elif mode == 2:
        data_df.to_csv(f"results/para_recovery/{model}_gen_sequences.csv")
    
    
        
    # Extract session indices, outcomes, and true probabilities from the DataFrame
    sessionIndex= data_dict_df['sessionIndex'] 
    outcomes = data_dict_df['outcome']             
    true_p = data_dict_df['p1']                
    
                  
    
    # Create the new dictionary
    organized_outcomes = {}
    
    # Assuming each session has 1000 trials, iterate over 10 sessions
    for session in range(len(np.unique(sessionIndex))):
        # Slice the outcomes and true probabilities for the current session
        start_index = session * 1000  # Calculate the starting index for the current session
        end_index = start_index + 1000 # Calculate the ending index for the current session
    
        organized_outcomes[session] = {
            "outcome": outcomes[start_index:end_index],  # Trials for the current session
            "true_p": true_p[start_index:end_index]      # True probabilities for the current session
        }

    
    return organized_outcomes
                


## get models parameters

def sample_params(model,niter, seed):
    """get_estimated_params
       
    mod_names = ["HMM", "changepoint", "delta_rule", "p_hall","reduced_bayes",
                 "reduced_bayes_lamda", "PID"]
    """
      
    
    FM_df = pd.read_csv('optimisation_results/FM_opt_results.csv')
    G_df = pd.read_csv('optimisation_results/G_opt_results.csv')
    K_df = pd.read_csv('optimisation_results/K_opt_results.csv')
    
    combined_df = pd.concat([FM_df, G_df, K_df], ignore_index=True)
        
    
    
    if model =="HMM":
        prm_range = [combined_df['HMM_pc'].min(), combined_df['HMM_pc'].max()] 
        prm_range = [prm_range]
        
    elif model == "changepoint":
        prm_range_T1 = [combined_df['changepoint_T1'].min(), combined_df['changepoint_T1'].max()]  
        prm_range_T2 = [combined_df['changepoint_T2'].min(), combined_df['changepoint_T2'].max()]  
        prm_range = [prm_range_T1, prm_range_T2]
        
    elif model == "delta_rule":
        prm_range = [combined_df['delta_lr'].min(), combined_df['delta_lr'].max()] 
        prm_range = [prm_range]
        
    elif model == "p_hall":
        prm_range_lr = [combined_df['p_hall_lr'].min(), combined_df['p_hall_lr'].max()]  
        prm_range_w = [combined_df['p_hall_w'].min(), combined_df['p_hall_w'].max()]  
        prm_range = [prm_range_lr, prm_range_w]
        
    elif model == "reduced_bayes":
        prm_range = [combined_df['reduced_bayes_pc'].min(), combined_df['reduced_bayes_pc'].max()] 
        prm_range = [prm_range]
        
    elif model == "reduced_bayes_lamda":
        prm_range_pc = [combined_df['reduced_bayes_lamda_pc'].min(), combined_df['reduced_bayes_lamda_pc'].max()]  
        prm_range_lamda = [combined_df['reduced_bayes_lamda'].min(), combined_df['reduced_bayes_lamda'].max()]  
        prm_range = [prm_range_pc, prm_range_lamda]
        
    elif model == "PID":
        prm_range_kp = [combined_df['PID_kp'].min(), combined_df['PID_kp'].max()]  
        prm_range_ki = [combined_df['PID_ki'].min(), combined_df['PID_ki'].max()] 
        prm_range_kd = [combined_df['PID_kd'].min(), combined_df['PID_kd'].max()] 
        prm_range_lamda = [combined_df['PID_lamda'].min(), combined_df['PID_lamda'].max()]
        prm_range = [prm_range_kp, prm_range_ki,prm_range_kd,prm_range_lamda]
        
    elif model == "Mixed_delta":
        prm_range_delta1 = [combined_df['mixed_delta_delta1'].min(), combined_df['mixed_delta_delta1'].max()]  
        prm_range_delta2 = [combined_df['mixed_delta_delta2'].min(), combined_df['mixed_delta_delta2'].max()] 
        prm_range_hrate = [combined_df['mixed_delta_hrate'].min(), combined_df['mixed_delta_hrate'].max()] 
        prm_range_nu_p = [combined_df['mixed_delta_nu_p'].min(), combined_df['mixed_delta_nu_p'].max()]
        prm_range = [prm_range_delta1, prm_range_delta2,prm_range_hrate,prm_range_nu_p]
        
    elif model == "VKF":
        prm_range_lamda = [combined_df['VKF_lamda'].min(), combined_df['VKF_lamda'].max()]  
        prm_range_omega = [combined_df['VKF_omega'].min(), combined_df['VKF_omega'].max()] 
        prm_range_v0 = [combined_df['VKF_v0'].min(), combined_df['VKF_v0'].max()] 
        prm_range = [prm_range_lamda, prm_range_omega,prm_range_v0]
        
    elif model == "HGF":
        prm_range_nu = [combined_df['HGF_nu'].min(), combined_df['HGF_nu'].max()]  
        prm_range_kappa = [combined_df['HGF_kappa'].min(), combined_df['HGF_kappa'].max()] 
        prm_range_omega = [combined_df['HGF_omega'].min(), combined_df['HGF_omega'].max()] 
        prm_range = [prm_range_nu, prm_range_kappa,prm_range_omega]

    
    sampled = []
    np.random.seed(seed)
    
    for p in range(len(prm_range)):
        samples = np.random.uniform(prm_range[p][0], 
                                  prm_range[p][1], 
                                  size=niter)
        sampled.append(samples)
        
        
    return sampled





##  stimulated model data


def stimulate_mod_data(model,niter,seed,n_sessions,n_trials,mode):
    
    gen_seqs = gen_sequences(n_sessions,n_trials,seed,model,mode) 
    mod_paras = sample_params(model,niter,seed)
    
    # get the estimate 

    stimulation_results = []
    stimulation_seqs = []
    stimulation_seqs_truep = []
    stimulation_seqs_index =[]
    np.random.seed(seed)
    
    for i in range(niter):
        # randomly draw 10 for each iteration
        used_seqs = np.random.choice(np.arange(len(gen_seqs)), size=10, replace=False)
        
        selected_seqs = [np.array(gen_seqs[index]["outcome"]) for index in used_seqs]
        selected_seqs_truep = [np.array(gen_seqs[index]["true_p"]) for index in used_seqs]
    
        update_response = []
            
        for session in range(len(selected_seqs)):
            
                
            stim = selected_seqs[session]  
            
            if model == "HMM":           
                
                resol = 20
                p_c = mod_paras[0][i]
                p1_min = 0
                p1_max = 1
                do_inference_on_current_trial = True
    
                options = {
                        'resol': resol,
                        'p_c': p_c,
                        'p1_min': p1_min,
                        'p1_max': p1_max,
                        'do_inference_on_current_trial': do_inference_on_current_trial,
                    }
                inference_out = IO.run_inference(stim, options=options)
                estimates = inference_out[1,]['mean']
                
            elif model == "changepoint":
                
                PChyp = [0.5, 0.5] 
                PgHyp = [0.5, 0.5]  
                T1 = mod_paras[0][i]
                T2 = mod_paras[1][i]
                
                # Number of trials for the current session
                ntrials = len(stim)
                       
                # Call to the BernCPKLfun function (assumed to be defined)
                record = bern_cp_kl_fun(stim, PgHyp[0], PgHyp[1], PChyp[0], PChyp[1], T1, T2)
            
                # Create a temporary variable to manipulate the record
                tmp = np.copy(record)
                tmp = np.vstack([tmp, [0,ntrials]])  # Add a row to tmp with ntrials
            
                estimates = np.zeros(ntrials)
            
                for r in range(len(tmp) - 1):
                    estimates[int(tmp[r, 1]):int(tmp[r + 1, 1])] = tmp[r, 0]
                    
            elif model == "delta_rule":
                lr = mod_paras[0][i]
                estimates = delta_update(stim,lr)
                
            elif model == "p_hall":
                lr = mod_paras[0][i]
                weight = mod_paras[1][i]
                estimates = pearce_hall_update(stim,lr,weight)
                
            elif model == "reduced_bayes":
                p_c = mod_paras[0][i]
                estimates = reduced_bayes_update(stim,p_c)
                
            elif model == "reduced_bayes_lamda":
                p_c = mod_paras[0][i]
                lamda = mod_paras[1][i]
                estimates = reduced_bayes_lamda_update(stim,p_c,lamda)
                
            elif model == "PID":
                Kp = mod_paras[0][i]
                Ki = mod_paras[1][i]
                Kd = mod_paras[2][i]
                lamda = mod_paras[3][i]
                estimates = PID_update(stim,Kp,Ki,Kd,lamda)
                
            elif model == "Mixed_delta":
                delta = [mod_paras[0][i],mod_paras[1][i]]
                hRate = mod_paras[2][i]
                nu_p = mod_paras[3][i]
                estimates = update_mixed_delta(stim,delta,hRate,nu_p)
                
            elif model == "VKF":
                lamda = mod_paras[0][i]
                omega = mod_paras[1][i]
                v0 = mod_paras[2][i]
                estimates = update_VKF(stim,lamda,omega,v0)
                
            elif model == "HGF":
                nu = mod_paras[0][i]
                kappa = mod_paras[1][i]
                omega = mod_paras[2][i]
                estimates = update_HGF(stim,nu,kappa,omega)
                              
            
            
            update_response.append(estimates)
            
        predicted_response = np.concatenate(update_response).flatten()
        
             
        stimulation_results.append(predicted_response)
        stimulation_seqs.append(selected_seqs)
        stimulation_seqs_truep.append(selected_seqs_truep)
        stimulation_seqs_index.append(used_seqs)
        
    
    return stimulation_results, mod_paras, stimulation_seqs, stimulation_seqs_truep, stimulation_seqs_index



def modr_MSE_fun(pars,data_seq,mod_stim,model,expID=1):
    
     
    if model =="HMM":
       
        p_c = pars[0]
        
        predicted_response = HMM(p_c,data_seq,expID)
        
    
    elif model == "changepoint":
        
        T1 = pars[0]
        T2 = pars[1]
                
                
        predicted_response = ChangePointModel(T1,T2,data_seq,expID)
    
    elif model == "delta_rule":
       
       lr = pars[0]
       
       
       predicted_response = delta_rule(lr,data_seq,expID)
       
    elif model == "p_hall":
       
       lr = pars[0]
       weight = pars[1]
       
       predicted_response = pearce_hall(lr,weight,data_seq,expID)
       
    elif model == "reduced_bayes":
       p_c = pars[0]
       
       predicted_response = reduced_bayes(data_seq,p_c,expID)
       
    elif model == "reduced_bayes_lamda":
        p_c = pars[0]
        lamda = pars[1]
        predicted_response = reduced_bayes_lamda(data_seq,p_c,lamda,expID)
        
    elif model == "PID":
        Kp = pars[0]
        Ki = pars[1]
        Kd = pars[2]
        lamda = pars[3]
        predicted_response = PID(data_seq,Kp,Ki,Kd,lamda,expID)
        
    elif model == "Mixed_delta":
        delta = pars[:-2]
        hRate = pars[-2]
        nu_p = pars[-1]
        predicted_response = mixed_delta(data_seq,delta,hRate,nu_p,expID)
        
    elif model == "VKF":
        lamda = pars[0]
        omega = pars[1]
        v0 = pars[2]
        predicted_response = VKF(data_seq,lamda,omega,v0,expID)
        
    elif model == "HGF":
        nu = pars[0]
        kappa = pars[1]
        omega = pars[2]
        predicted_response = HGF(data_seq,nu,kappa,omega,expID)
        
        
        

    
   
        # Compute squared errors
    squared_errors = (predicted_response - mod_stim) ** 2
    
    # if np.any(np.isnan(predicted_response)):
    #     return np.nan
    
    # sum of squared error
    sse = np.sum(squared_errors)
    
    # mean squared error
    mse = sse / len(predicted_response)  
    
    # rmse = np.sqrt(mse)
    
    
        
    return mse


def modr_get_initial_parameters(data_seq,mod_stim,model):
   
    if model =="HMM":
        
        # HMM: p_c
        lb = np.array([0.0001])
        ub = np.array([0.9])
        
        bounds = [(0.0001, 0.9)]
        
    elif model == "changepoint":
        # IIAB: T1, T2 
        lb = np.array([0.01, 0.001])
        ub = np.array([2, 20])
        
        bounds = [(0.01,2), (.001,20)]
       
        
    elif model == "delta_rule":
        # delta_rule: lr
        lb = np.array([0.0001])
        ub = np.array([0.9])
        
        bounds = [(0.0001, 0.9)]
        
    elif model == "p_hall":
        # Pearce-Hall: intial lr, weight
        lb = np.array([0.0001, 0])
        ub = np.array([0.9, 1])
        
        bounds = [(0.0001, 0.9), (0, 1)]
        
    elif model == "reduced_bayes":
        # reduced Bayes: p_c
        lb = np.array([0.0001])
        ub = np.array([0.9])       
        bounds = [(0.0001, 0.9)]
        
    elif model =="reduced_bayes_lamda":
        # reduced_bayes_lamda: p_c, lamda
        lb = np.array([0.0001, 0])
        ub = np.array([0.9, 1])
        
        bounds = [(0.0001, 0.9), (0, 1)]
        
    elif model == "PID":
        # PID: Kp,Ki,Kd,lamda
        lb = np.array([-1, -1, -1, 0])
        ub = np.array([1, 1, 1, 1])
        
        bounds = [(-1, 1), (-1, 1),(-1, 1),(0, 1)]
        
    elif model == "Mixed_delta":
        # Mixed_delta: delta1, delta2,hrate,nu_p
        lb = np.array([1.01, 1.01, 0, 0])
        ub = np.array([8, 17, 1, 5])
        
        bounds = [(1.01, 8), (1.01, 17),(0, 1),(0, 5)]
        
    elif model == "VKF":
        # VKF: lamda,omega,v0
        lb = np.array([0, 0, 0])
        ub = np.array([1, 1, 1])
        
        bounds = [(0,1), (0,1), (0, 1)]
        
    elif model == "HGF":
        # HGF: nu,kappa,omega
        lb = np.array([0.0001, 0.01, -5])
        ub = np.array([10, 1, 2])
        
        bounds = [(0.0001,10), (0.01,1), (-5, 2)]
            
    
    best_MSE = np.inf
    fp_init = None
    
    n_initial_guesses = 100
    
    for ii in range(n_initial_guesses):
        
        fp_random = lb + (ub - lb) * np.random.rand(len(lb))
        fp_random = np.maximum(np.minimum(fp_random, ub),lb)
        
        MSE = MSE_fun(fp_random, data_seq,mod_stim,model)
        
        if MSE < best_MSE :
            best_MSE = MSE
            fp_init = fp_random
        
    

    return fp_init, bounds




def modr_get_optimised_result(model,stimulation_seqs,stimulation_results):
    
    
    # get optimised paras
   
    data_seq = stimulation_seqs  
    mod_stim = stimulation_results
    model = model

    fp_init, bounds = get_initial_parameters(data_seq,mod_stim,model)  
                
    fp_init = np.array(fp_init)
        
        
    opt = so.minimize(MSE_fun, fp_init, args=(data_seq,mod_stim,model),
                                     method='Powell', bounds=bounds, options={'disp': False}) 
                   
    fval = opt['fun']
    x_min = opt['x']
    result = opt["success"]
            
     
    
    return result, x_min, fval 



#### parameter recovery


def pr_get_optimised_result(niter, model,stimulation_seqs,stimulation_results):
    # get optimised paras
    opt_res = []
    
    t_start_optimised = time()

    for i in range(niter):

        data_seq = stimulation_seqs[i]
        data_seq = np.concatenate(data_seq).flatten()    
        mod_stim = stimulation_results[i]
        model = model

        fp_init, bounds = modr_get_initial_parameters(data_seq,mod_stim,model)  
                
        fp_init = np.array(fp_init)
        
            
        print(f'Paras_recovery: Running optimization {i} for {model}')  
        
        opt = so.minimize(modr_MSE_fun, fp_init, args=(data_seq,mod_stim,model),
                                     method='Powell', bounds=bounds, options={'disp': False}) 
                   
        fval = opt['fun']
        x_min = opt['x']
        result = opt["success"]
            
        opt_res.append(x_min)
    
    t_end_optimised = (time() - t_start_optimised) / 60
    print(f"Paras_recovery for {model}: DONE IN {t_end_optimised:0.3f} MIN.")
    
    return opt_res 



#### MSE by session

def MSE_fun_trial(pars,data_seq,mod_stim,model,expID):
    
     
    if model =="HMM":
       
        p_c = pars[0]
        
        predicted_response = HMM_trial(p_c,data_seq,expID)
        
    
    
    elif model == "PID":
        Kp = pars[0]
        Ki = pars[1]
        Kd = pars[2]
        lamda = pars[3]
        predicted_response = PID_update(data_seq,Kp,Ki,Kd,lamda)
        
    elif model == "Mixed_delta":
        delta = pars[:-2]
        hRate = pars[-2]
        nu_p = pars[-1]
        predicted_response = update_mixed_delta(data_seq,delta,hRate,nu_p)
        
    
    
    predicted_response = np.array(predicted_response)
    
   
        # Compute squared errors
    squared_errors = (predicted_response - mod_stim) ** 2
    
    # if np.any(np.isnan(predicted_response)):
    #     return np.nan
    
    # sum of squared error
    sse = np.sum(squared_errors)
    
    # mean squared error
    mse = sse / len(predicted_response)  
    
    # rmse = np.sqrt(mse)
    
    
        
    return mse

def get_initial_parameters_trial(data_seq,mod_stim,model,expID):
   
    if model =="HMM":
        
        # HMM: p_c
        lb = np.array([0.0001])
        ub = np.array([0.9])
        
        bounds = [(0.0001, 0.9)]
        
    elif model == "changepoint":
        # IIAB: T1, T2 
        lb = np.array([0.01, 0.001])
        ub = np.array([2, 20])
        
        bounds = [(0.01,2), (.001,20)]
       
        
    elif model == "delta_rule":
        # delta_rule: lr
        lb = np.array([0.0001])
        ub = np.array([0.9])
        
        bounds = [(0.0001, 0.9)]
        
    elif model == "p_hall":
        # Pearce-Hall: intial lr, weight
        lb = np.array([0.0001, 0])
        ub = np.array([0.9, 1])
        
        bounds = [(0.0001, 0.9), (0, 1)]
        
    elif model == "reduced_bayes":
        # reduced Bayes: p_c
        lb = np.array([0.0001])
        ub = np.array([0.9])       
        bounds = [(0.0001, 0.9)]
        
    elif model =="reduced_bayes_lamda":
        # reduced_bayes_lamda: p_c, lamda
        lb = np.array([0.0001, 0])
        ub = np.array([0.9, 1])
        
        bounds = [(0.0001, 0.9), (0, 1)]
        
    elif model == "PID":
        # PID: Kp,Ki,Kd,lamda
        lb = np.array([-1, -1, -1, 0])
        ub = np.array([1, 1, 1, 1])
        
        bounds = [(-1, 1), (-1, 1),(-1, 1),(0, 1)]
        
    elif model == "Mixed_delta":
        # Mixed_delta: delta1, delta2,hrate,nu_p
        lb = np.array([1.01, 1.01, 0, 0])
        ub = np.array([8, 17, 1, 5])
        
        bounds = [(1.01, 8), (1.01, 17),(0, 1),(0, 5)]
        
    elif model == "VKF":
        # VKF: lamda,omega,v0
        lb = np.array([0, 0, 0])
        ub = np.array([1, 1, 1])
        
        bounds = [(0,1), (0,1), (0, 1)]
        
    elif model == "HGF":
        # HGF: nu,kappa,omega
        lb = np.array([0.0001, 0.01, -5])
        ub = np.array([10, 1, 2])
        
        bounds = [(0.0001,10), (0.01,1), (-5, 2)]
            
    
    best_MSE = np.inf
    fp_init = None
    
    n_initial_guesses = 100
    
    for ii in range(n_initial_guesses):
        
        fp_random = lb + (ub - lb) * np.random.rand(len(lb))
        fp_random = np.maximum(np.minimum(fp_random, ub),lb)
        
        MSE = MSE_fun_trial(fp_random, data_seq,mod_stim,model,expID)
        
        if MSE < best_MSE :
            best_MSE = MSE
            fp_init = fp_random
        
    

    return fp_init, bounds



def get_optimised_result_trial(model,stimulation_seqs,stimulation_results,expID):
    
    
    # get optimised paras
   
    data_seq = stimulation_seqs  
    mod_stim = stimulation_results
    model = model

    fp_init, bounds = get_initial_parameters_trial(data_seq,mod_stim,model,expID)  
                
    fp_init = np.array(fp_init)
        
        
    opt = so.minimize(MSE_fun_trial, fp_init, args=(data_seq,mod_stim,model,expID),
                                     method='Powell', bounds=bounds, options={'disp': False}) 
                   
    fval = opt['fun']
    x_min = opt['x']
    result = opt["success"]
    
    return result, x_min, fval 






