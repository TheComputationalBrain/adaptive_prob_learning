#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For details of the models, please refer to
    Nassar, M. R., Wilson, R. C., Heasly, B., & Gold, J. I. (2010). An approximately Bayesian delta-rule model explains the dynamics of belief updating in a changing environment. Journal of Neuroscience, 30(37), 12366-12378.

"""


import numpy as np
import scipy.stats
from scipy.stats import binom
import matplotlib.pyplot as plt


#%%

def reduced_bayes(omega,p_c,expID):
       
    
    if expID == 1:
        
        # Determine the number of sessions
        n_trials_per_session = 1000  
    
    elif expID == 2:
        n_trials_per_session = 999
        
    elif expID == 3:
        n_trials_per_session = 75
    
    
    n_sessions = omega.shape[0] // n_trials_per_session
    
    # Initialize a list to store predicted responses for all sessions
    update_response = []
    
    
    for session in range(n_sessions):
    
        # Get the trials for the current session
        session_trials = omega[session * n_trials_per_session:(session + 1) * n_trials_per_session]
        
        
        stim = session_trials  # Set the stimulus to be the trials of the current session
        
        
        estimates = reduced_bayes_update(stim,p_c)
            
        
        
        update_response.append(estimates)
    
    predicted_response = np.concatenate(update_response).flatten()
    
    return predicted_response

#%% Reduced Bayesian model with under-weighted likelihood information

# two free parameters , p_c and lamda 

def reduced_bayes_update(dat,p_c):

    
    outcome = dat
    
    n_trials = outcome.shape[0]
    
    b = np.empty(n_trials + 1) # include the prior the inference loop
    b[0] = 0.5 # estimate
    p_c = p_c # fixed for now
    lamda = 1 # fixed for now, 1 = the model is equivalent to the reduced Bayesian model
    
    r = np.empty(n_trials + 1) # expected run length
    r[0] = 1 # is it correct ? 
    
    omega = np.empty(n_trials)
    outcome_probability = np.empty(n_trials)
    
    for t in range(n_trials):
        
        # outcome probability under prev. belief: p(X_t | no_cp)
       
        outcome_probability[t] = b[t] if outcome[t] == 1 else (1 - b[t])
        
        
        # Nasser 2010, eq(25)
        omega_num = p_c * (.5 ** lamda)
        
        omega_denom = omega_num + (outcome_probability[t]**lamda) * (1 - p_c)
        
        omega[t] = (omega_num / omega_denom)
        
          
        # Nasser 2010, eq(19) and eq(20)
        
        learning_rate = (1 + omega[t] * r[t]) / (r[t] + 1)
        
        predictoin_error = outcome[t] - b[t]
        
        b[t+1] = b[t] + learning_rate * predictoin_error
        
        
        # Nasser 2010, eq(21)
        r[t+1]= (r[t]+ 1)*(1-omega[t]) + omega[t]
        
    
    return b[1:]


#%%

def reduced_bayes_lamda(omega,p_c,lamda,expID):
       
    
    if expID == 1:
        
        # Determine the number of sessions
        n_trials_per_session = 1000  
    
    elif expID == 2:
        n_trials_per_session = 999
        
    elif expID == 3:
        n_trials_per_session = 75
    
    
    n_sessions = omega.shape[0] // n_trials_per_session
    
    # Initialize a list to store predicted responses for all sessions
    update_response = []
    
    
    for session in range(n_sessions):
    
        # Get the trials for the current session
        session_trials = omega[session * n_trials_per_session:(session + 1) * n_trials_per_session]
        
        
        stim = session_trials  # Set the stimulus to be the trials of the current session
        
        
        estimates = reduced_bayes_lamda_update(stim,p_c,lamda)
            
        
        
        update_response.append(estimates)
    
    predicted_response = np.concatenate(update_response).flatten()
    
    return predicted_response


#%%
def reduced_bayes_lamda_update(dat,p_c,lamda):

    
    outcome = dat
    
    n_trials = outcome.shape[0]
    
    b = np.empty(n_trials + 1) # include the prior the inference loop
    b[0] = 0.5 # estimate
    p_c = p_c 
    lamda = lamda 
    
    r = np.empty(n_trials + 1) # expected run length
    r[0] = 1 # is it correct ? 
    
    omega = np.empty(n_trials)
    outcome_probability = np.empty(n_trials)
    
    for t in range(n_trials):
        
        # outcome probability under prev. belief: p(X_t | no_cp)
       
        outcome_probability[t] = b[t] if outcome[t] == 1 else (1 - b[t])
        
        
        # Nasser 2010, eq(25)
        omega_num = p_c * (.5 ** lamda)
        
        omega_denom = omega_num + (outcome_probability[t]**lamda) * (1 - p_c)
        
        omega[t] = (omega_num / omega_denom)
        
          
        # Nasser 2010, eq(19) and eq(20)
        
        learning_rate = (1 + omega[t] * r[t]) / (r[t] + 1)
        
        predictoin_error = outcome[t] - b[t]
        
        b[t+1] = b[t] + learning_rate * predictoin_error
        
        
        # Nasser 2010, eq(21)
        r[t+1]= (r[t]+ 1)*(1-omega[t]) + omega[t]
        
    
    return b[1:]

