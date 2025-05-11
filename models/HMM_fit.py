#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
import scipy.io


# Get the current working directory
# current_directory = os.getcwd()

# Walk through the directory tree and add all subdirectories to sys.path
# for root, dirs, files in os.walk(current_directory):
#     for dir in dirs:
#         sys.path.append(os.path.join(root, dir))

from models import IdealObserver as IO
from models import Inference_ChangePoint as IC


#%%

def HMM(p_c,data,expID):
    
    if expID == 1:
        
       
        n_trials_per_session = 1000
        
        resol = 20
        p_c = p_c
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

    
    elif expID == 2:
        n_trials_per_session = 999
        
        resol = 20
        p_c =  p_c
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
        
    elif expID == 3:
        n_trials_per_session = 75
        
        resol = 20
        p_c =  p_c
        p1_min = 0.1
        p1_max = 0.9
        do_inference_on_current_trial = True


        options = {
                'resol': resol,
                'p_c': p_c,
                'p1_min': p1_min,
                'p1_max': p1_max,
                'do_inference_on_current_trial': do_inference_on_current_trial,
            }
    
    
    n_sessions = data.shape[0] // n_trials_per_session
    
    # Initialize a list to store predicted responses for all sessions
    estimate = []
    
    
    for session in range(n_sessions):
    
        # Get the trials for the current session
        session_trials = data[session * n_trials_per_session:(session + 1) * n_trials_per_session]
        
        
        
        stim = np.array(session_trials)  # Set the stimulus to be the trials of the current session
        
    
        inference_out = IO.run_inference(stim, options=options)
        mod_est = inference_out[1,]['mean']
        

        estimate.append(mod_est) 
        
        
    
    predicted_response = np.concatenate(estimate).flatten()

    return predicted_response



def HMM_trial(p_c,data,expID):
    
    if expID == 1:
        
       
        n_trials_per_session = 1000
        
        resol = 20
        p_c = p_c
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

    
    elif expID == 2:
        n_trials_per_session = 999
        
        resol = 20
        p_c =  p_c
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
        
    elif expID == 3:
        n_trials_per_session = 75
        
        resol = 20
        p_c =  p_c
        p1_min = 0.1
        p1_max = 0.9
        do_inference_on_current_trial = True


        options = {
                'resol': resol,
                'p_c': p_c,
                'p1_min': p1_min,
                'p1_max': p1_max,
                'do_inference_on_current_trial': do_inference_on_current_trial,
            }
        
        
    inference_out = IO.run_inference(data, options=options)
    mod_est = inference_out[1,]['mean']
    
    
    predicted_response = mod_est
    
    
    return predicted_response


