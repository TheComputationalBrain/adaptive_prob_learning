#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For details of the model, please refer to:
   Wagner, A. R., & Rescorla, R. A. (1972). Inhibition in‬ Pavlovian conditioning: Application 
   Pearce, J. M., & Hall, G. (1980). A Model for Pavlovian‬ Learning: Variations in the Effectiveness of‬ Conditioned But Not of Unconditioned Stimuli.‬ Psychological Review‬‭ ,‬‭ 87‬‭ (6), 532–552.‬
‭ 
"""

import numpy as np



#%% moving average

def delta_rule(lr, omega,expID):
    
    
    
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
        
        
        estimates = delta_update(stim,lr)
            
        
        
        update_response.append(estimates)
    
    predicted_response = np.concatenate(update_response).flatten()
    
    return predicted_response

#%% delta rule
def delta_update(omega,lr,prior_estimate=0.5):
    
    estimates = np.zeros(len(omega))
    for t in range(len(omega)):
    
        prev_estimate = estimates[t-1] if t > 0 else prior_estimate
    
    
    # Update the predicted response for time t based on the delta rule
        estimates[t] = prev_estimate + lr * (omega[t] - prev_estimate)
    
    return estimates
    
        

#%%

def pearce_hall(lr,weight,omega,expID):
    
    
    
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
        
        
        estimates = pearce_hall_update(stim,lr,weight)
            
        
        
        update_response.append(estimates)
    
    predicted_response = np.concatenate(update_response).flatten()
    
    return predicted_response


#%% Pearce-Hall model 
# fit initial lr and weight factor

def pearce_hall_update(dat,lr,weight,prior_estimate=0.5):
    
    n = len(dat)
    estimates = np.zeros(n)  # Predictions for each trial
    lamda = np.zeros(n)     # Dynamic learning rates
    
    
    for t in range(len(dat)):
        prev_estimate = estimates[t-1] if t > 0 else prior_estimate
        prediction_error = dat[t] - prev_estimate
        
        # Update the learning rate based on prediction error and previous learning rate
        lamda[t] = weight * (abs(prediction_error)) + (1-weight) * lamda[t-1] if t > 0 else lr
        
        
        # Update the predicted response for time t based on the delta rule
        estimates[t] = prev_estimate + lamda[t] * prediction_error
    

    return estimates






