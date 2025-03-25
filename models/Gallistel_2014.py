#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
the code was provided by Gallistel et al., for details of the model, pleas refer to 
    Gallistel, Charles R., et al. "The perception of probability." Psychological Review 121.1 (2014): 96.

"""

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
from scipy.special import betaln
from pybads import BADS

from time import time



#%% Change Point Model 
# =============================================================================
# #%%  Change Point Model by Gallistelet al (2014, Psych Rev)
# =============================================================================

def ChangePointModel(T1,T2,omega,expID):
    
    
    PChyp = [0.5, 0.5] 
    PgHyp = [0.5, 0.5]  
    
    
    if expID == 1:
        
        # Determine the number of sessions
        n_trials_per_session = 1000  
    
    elif expID == 2:
        n_trials_per_session = 999
        
    elif expID == 3:
        n_trials_per_session = 75
    
    
    n_sessions = omega.shape[0] // n_trials_per_session
    
    # Initialize a list to store predicted responses for all sessions
    estimate = []
    
    
    for session in range(n_sessions):
    
        # Get the trials for the current session
        session_trials = omega[session * n_trials_per_session:(session + 1) * n_trials_per_session]
        
        # Number of trials for the current session
        ntrials = len(session_trials)
        
        stim = session_trials  # Set the stimulus to be the trials of the current session
        
    
        # Call to the BernCPKLfun function (assumed to be defined)
        record = bern_cp_kl_fun(stim, PgHyp[0], PgHyp[1], PChyp[0], PChyp[1], T1, T2)
    
        # Create a temporary variable to manipulate the record
        tmp = np.copy(record)
        tmp = np.vstack([tmp, [0,ntrials]])  # Add a row to tmp with ntrials
    
        mdl_ps = np.zeros(ntrials)
    
        for r in range(len(tmp) - 1):
            mdl_ps[int(tmp[r, 1]):int(tmp[r + 1, 1])] = tmp[r, 0]
        

        estimate.append(mdl_ps) 
        
        
    
    predicted_response = np.concatenate(estimate).flatten()

    return predicted_response



def bern_cp_kl_fun(data, alpha_p, beta_p, alpha_c, beta_c, T1, T2):
    
    # Initialize parameters
    p_hat = alpha_p / (alpha_p + beta_p)  # Initial estimate of p_hat
    alpha_c_h = alpha_c  # Initial hyperparameter for beta distribution on p_c
    beta_c_h = beta_c  # Initial hyperparameter for beta distribution on p_c
    hyper_c_rec = np.array([[alpha_c, beta_c]])  # Record of values of hyperparameters on p_c
    alpha_a = alpha_p
    beta_a = beta_p  # Hyperparameters on p_hat after a change
    hyper_p_rec = np.array([[alpha_p, beta_p]])  # Record of values of hyperparameters on p_hat
    Nc = alpha_c  # Track of number of change points
    cp = [0] # Change point vector
    DP = [] # Detection points
    D = data  # data vector
    
    Det = 0  # Initialize detection point
    
    Record = [[p_hat, Det]]  # Record the initial state
    
    while len(D) > 20:  # Ignore the last 20 trials
        N = np.arange(1, len(D) + 1)  # Create a range of trial indices
        p_o = np.cumsum(D) / N  # Observed probability of success
        
        # E = KL divergence
        E = N * bernd_kl(p_o, p_hat)  # Compute KL divergences
        
        
        
        if np.all(E[Det:] <= T1):  # No evidence exceeds the decision criterion
            break  # Exit the loop
        else:  # Evidence exceeds the decision criterion
            Det += np.argmax(E[Det:] > T1) + 1  # Find the first index exceeding T1_noisy
            
        DP.append(cp[-1] + Det if cp else Det)  # Record latest detection point
        alpha_c = Nc + alpha_c_h  # Update parameters of beta distribution on p_c
        beta_c = DP[-1] - Nc + beta_c_h  # Update parameters
        pc_hat = alpha_c / (alpha_c + beta_c)  # Estimate of p_c
             
        
        CP, PstO = bern_cp(D[:Det], alpha_p, beta_p, alpha_a, beta_a, pc_hat)
        
        
        if PstO > T2:  # Detected a change point
            cp.append(cp[-1] + CP if cp else CP)  # Record change point
            alpha_p = 1 + np.sum(D[CP:Det])  # Update parameters
            beta_p = 1 + (Det - CP) - np.sum(D[CP:Det])
            hyper_p_rec = np.vstack((hyper_p_rec, [alpha_p, beta_p]))  # Record hyperparameters
            hyper_c_rec = np.vstack((hyper_c_rec, [alpha_c, beta_c]))  # Record hyperparameters
            D = D[CP:]  # Remove data up to change point
            Det = 1 + DP[-1] - cp[-1]  # Reset Det
            p_hat = alpha_p / (alpha_p + beta_p)  # Recompute current estimate
            Nc += 1  # Increment change point count
            
        elif (len(cp) > 1) and (PstO < T2):  # No apparent change point
            TestPrevCP_data = data[cp[-2] :DP[-1]]  # Data since last change point
            CPr, Or = bern_cp(TestPrevCP_data, alpha_a, beta_a, alpha_a, beta_a, pc_hat)
            
            if CPr is None or Or < T2:  # Previous change point was unjustified
                cp.pop()  # Remove most recent cp
                hyper_p_rec = hyper_p_rec[:-1]  # Remove last record
                hyper_c_rec = hyper_c_rec[:-1]  # Remove last record
                Nc -= 1  # Decrement change-point count
                
                # Update parameters
                alpha_p = hyper_p_rec[-1, 0] + np.sum(TestPrevCP_data)
                beta_p = hyper_p_rec[-1, 1] + len(TestPrevCP_data) - np.sum(TestPrevCP_data)
                p_hat = alpha_p / (alpha_p + beta_p)  # Update p_hat
                
                # Restore data to last cp before false alarm
                D = data[cp[-1]:]
                Det = 1 + DP[-1] - cp[-1]
                
            else:  # Previous estimate of p_g was wrong
                cp[-1] = cp[-2] +CPr  # Update change point
                alpha_p = 1 + np.sum(data[cp[-1]:DP[-1]])
                beta_p = 1 + len(data[cp[-1]:DP[-1]])- np.sum(data[cp[-1]:DP[-1]])
                p_hat = alpha_p / (alpha_p + beta_p)  # Update p_hat
                D = data[cp[-1]:]  # Update data
                Det = 1 + DP[-1] - cp[-1]
                
        else:  # No change point detected
            alpha_p = 1 + np.sum(D[:Det])
            beta_p = 1 + len(D[:Det]) - np.sum(D[:Det])
            p_hat = alpha_p / (alpha_p + beta_p)  # Update p_hat

        Record.append([p_hat, DP[-1]])  # Record the state
        

    return np.array(Record)




def bernd_kl(p1,p2):
    # Ensure p1 and p2 are between 1e-9 and 1 - 1e-9
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    p2 = np.clip(p2, 1e-9, 1 - 1e-9)
    
    # Calculate the Kullback-Leibler divergence
    Dkl = p1 * np.log(p1 / p2) + (1 - p1) * np.log((1 - p1) / (1 - p2))
    
    return Dkl
    


def bern_cp(dat, alpha_b, beta_b, alpha_a, beta_a, pc):
    
    L = np.arange(1, len(dat) + 1)  # Cumulative number of observations (column vector)
    Sf = np.cumsum(dat, axis=0)  # Cumulative successes up to and including the nth datum
    Ff = L - Sf  # Cumulative failures up to and including the nth datum
    Sr = np.sum(dat) - Sf  # Successes after nth datum
    Lr = np.arange(len(dat) - 1, -1, -1)  # Number of observations after the nth datum
    Fr = Lr - Sr  # Failures after the nth datum

    # Log of the posterior likelihood function for the forward and reverse pass
    lg_pst_lkf = betaln(alpha_b + Sf, beta_b + Ff)  # Forward likelihood
    lg_pst_lkr = betaln(alpha_a + Sr, beta_a + Fr)  # Reverse likelihood

    # The log likelihood function as a function of the change point
    lg_lk_fun = lg_pst_lkf + lg_pst_lkr
    lg_lk_fun_n = lg_lk_fun + abs(np.max(lg_lk_fun))  # Normalizing log-likelihood peak to 0
    lk_fun = np.exp(lg_lk_fun_n)  # Likelihood function

    # Expectation of the likelihood function (not its mean)
    cp_mean = np.sum((np.arange(1, len(dat) + 1) * lk_fun)) / np.sum(lk_fun)
    
    # Bayes Factor (ratio of posterior likelihoods of change vs no-change) 
    rel_lkhd = (np.sum(lk_fun) / len(dat)) / lk_fun[-1]
    
    # Prior odds in favor of change
    prior_odds = L[-1] * pc / (1 - pc)

    # Posterior odds (relative likelihood * prior odds)
    odds = prior_odds * rel_lkhd

    # Round the change point mean to get the estimated change point
    cp = round(cp_mean)

    return cp, odds

