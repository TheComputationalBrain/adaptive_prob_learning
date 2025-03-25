#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For details of the model, please refer to:
    Mathys, C., Daunizeau, J., Friston, K. J., & Stephan, K. E. (2011). A Bayesian foundation for individual learning under uncertainty. Frontiers in human neuroscience, 5, 39.
    
the code of the model is adpated from (https://github.com/payampiray/piray_daw_2020_ploscb) Piray P, Daw ND, 2019, "A simple model for learning in volatile environments", biorxiv
http://dx.doi.org/10.1101/701466
"""


import numpy as np
import matplotlib.pyplot as plt
import warnings

#%%

def HGF(dat,nu,kappa,omega,expID):
       
    
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
        
        
        estimates = update_HGF(stim,nu,kappa,omega)
            
        
        
        update_response.append(estimates)
    
    predicted_response = np.concatenate(update_response).flatten()
    
    return predicted_response


def update_HGF(dat,nu,kappa,omega):
    
    N = len(dat)
    
    y = np.vstack([np.zeros((1, 1)), dat.reshape(-1, 1)])
    
    mu3 = np.full((N + 1, 1), np.nan)
    mu2 = np.full((N + 1, 1), np.nan)
    sigma2 = np.full((N + 1, 1), np.nan)
    sigma3 = np.full((N + 1, 1), np.nan)
    
    mu1hat = np.full((N + 1, 1), np.nan)
    
    # Initial conditions
    mu2[0, :] = 0
    sigma2[0, :] = 0.1
    mu3[0, :] = 1
    sigma3[0, :] = 1
    
    bad_traj = False  # Flag to track invalid trajectories

    
    # Define logistic (sigmoid) function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    
    for n in range(1, N + 1):
        
        expmu3 = np.exp(omega + kappa * mu3[n - 1, :])
        
        mu1hat[n, :] = sigmoid(mu2[n - 1, :])  # Eq 24
        delta1 = y[n, :] - mu1hat[n, :]        # Eq 25
        sigma1hat = mu1hat[n, :] * (1 - mu1hat[n, :])  # Eq 26
        sigma2hat = sigma2[n - 1, :] + expmu3  # Eq 27
        pi2 = 1 / sigma2hat + sigma1hat        # Eq 28
        
        LR = 1 / pi2
        
        mu2[n, :] = mu2[n - 1, :] + LR * delta1  # Eq 23
        sigma2[n, :] = LR                        # Eq 22
        
        pihat3 = 1 / (sigma3[n - 1, :] + nu)     # Eq 31
        w2 = expmu3 / (expmu3 + sigma2[n - 1, :])  # Eq 32
        r2 = (expmu3 - sigma2[n - 1, :]) / (expmu3 + sigma2[n - 1, :])  # Eq 33
        
        delta2 = (sigma2[n, :] + (mu2[n, :] - mu2[n - 1, :])**2) / (sigma2[n - 1, :] + expmu3) - 1  # Eq 34
        
        
        pi3 = pihat3 + (kappa**2 / 2) * w2 * (w2 + r2 * delta2)  # Eq 29
        
        if np.any(pi3 <= 0):
                bad_traj = True
                
        sigma3[n, :] = 1 / pi3
        mu3[n, :] = mu3[n - 1, :] + (kappa / 2) * sigma3[n, :] * w2 * delta2  # Eq 30
      
        
    # Flatten variables to check for NaNs
    var = np.hstack([mu2.ravel(), mu3.ravel(), sigma2.ravel(), sigma3.ravel()])
    
    
    if np.any(np.isnan(var)):
        bad_traj = True 
        
        
        
    m = mu1hat[1:].flatten()
    
    return m







