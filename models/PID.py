#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For details of the model, please refer to
    Ritz, H., Nassar, M. R., Frank, M. J., & Shenhav, A. (2018). A control theoretic model of adaptive learning in dynamic environments. Journal of cognitive neuroscience, 30(10), 1405-1421.

"""

import numpy as np
import matplotlib.pyplot as plt

#%%

def PID(omega,Kp,Ki,Kd,lamda,expID):
       
    
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
        
        
        estimates = PID_update(stim,Kp,Ki,Kd,lamda)
            
        
        
        update_response.append(estimates)
    
    predicted_response = np.concatenate(update_response).flatten()
    
    return predicted_response
#%%
def PID_update(dat,Kp,Ki,Kd,lamda):

    outcome = dat
    
    n_trials = dat.shape[0]
    
    
    
    # Initialize the variables for PID
    
    integral = 0
    integrals = []
    
    # Lists to store the estimates and actual probabilities
    
    errors = []
    
    probability_estimates = np.empty(n_trials+1) 
    probability_estimates[0] = 0.5 # estimate 
    
    # Define logistic (sigmoid) function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
       
    
    for t in range(n_trials):
       
        
        # Calculate the error (difference between actual outcome and estimate)
    
        error = outcome[t]- probability_estimates[t]
        
        
        # Update integral (sum of errors over time)
        integral = lamda * integral + error 
        
        if t == 0:
            derivative = 0
        else:
        # Calculate derivative (rate of change of error)
            derivative = error - errors[t-1]
        
        
        # PID controller formula to adjust the estimate, 
        correction = Kp * error + Ki * integral + Kd * derivative
        current_estimate = probability_estimates[t] + correction
        
        # Ensure the estimate remains between 0 and 1
        # current_estimate = min(max(current_estimate, 0), 1)
        
        current_estimate = sigmoid(current_estimate)
        
    
        
        # Store the estimates and errors for plotting
        probability_estimates[t+1] = current_estimate
        errors.append(error)
        integrals.append(integral)
    
    return probability_estimates[1:]
    
 




