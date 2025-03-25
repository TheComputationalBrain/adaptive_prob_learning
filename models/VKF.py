#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For details of the model, please refer to
    Piray, P., & Daw, N. D. (2020). A simple model for learning in volatile environments. PLoS computational biology, 16(7), e1007963.

the code is adpated from (https://github.com/payampiray/VKF)

"""


import numpy as np
import matplotlib.pyplot as plt


#%%

def VKF(dat,lamda,omega,v0,expID):
       
    
    if expID == 1:
        
        # Determine the number of sessions
        n_trials_per_session = 1000  
    
    elif expID == 2:
        n_trials_per_session = 999
        
    elif expID == 3:
        n_trials_per_session = 75
    
    
    n_sessions = dat.shape[0] // n_trials_per_session
    
    # Initialize a list to store predicted responses for all sessions
    update_response = []
    
    
    for session in range(n_sessions):
    
        # Get the trials for the current session
        session_trials = dat[session * n_trials_per_session:(session + 1) * n_trials_per_session]
        
        
        stim = session_trials  # Set the stimulus to be the trials of the current session
        
        
        estimates = update_VKF(stim,lamda,omega,v0)
            
        
        
        update_response.append(estimates)
    
    predicted_response = np.concatenate(update_response).flatten()
    
    return predicted_response


def update_VKF(dat,lamda,omega,v0):

    
    # Initialize parameters and arrays
    nt = len(dat)
    m = 0  
    w = omega  
    v = v0   # Initial volatility
        
    # Initialize output arrays with NaNs
    dv = np.full((nt, 1), np.nan)       
    lr = np.full((nt, 1), np.nan)       # Array for learning rates
    vol = np.full((nt, 1), np.nan)      # Array for volatility
    
    um = np.zeros(nt)    # Array for updated estimates
    
    
    # Define logistic (sigmoid) function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    
    for t in range(nt):
        
        dv[t, :] = m
        lr[t, :] = np.sqrt(w + v) # eq 15 
        vol[t, :] = v
        
        wpre = w
        
        delta = np.sqrt(w + v) * (dat[t] - sigmoid(m)) # eq 15, 16 , lr * prediciotn error
        
        m = m + delta # eq 16
        
        k = (w + v) / (w + v + omega) # eq 14
        
        w = (1 - k) * (w + v) # eq 17
        
        v = v + lamda * (delta**2 + k * wpre - k * v) # eq 19
        
        um[t] = sigmoid(m)
        
    return um





