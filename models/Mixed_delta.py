#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For details of the model, please refer to
    Wilson, R. C., Nassar, M. R., & Gold, J. I. (2013). A mixture of delta-rules approximation to bayesian inference in change-point problems. PLoS computational biology, 9(7), e1003150.

the code is adapted from (https://github.com/lacerbi/ChangeProb) Norton EH, Acerbi L, Ma WJ, Landy MS (2019) Human online adaptation to changes in prior probability. PLoS Comput Biol 15(7): e1006681. https://doi.org/10.1371/journal.pcbi.1006681 
"""

import numpy as np

#%%
def mixed_delta(omega,delta,hRate,nu_p,expID):
       
    
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
        
        
        estimates = update_mixed_delta(stim,delta,hRate,nu_p)
            
        
        
        update_response.append(estimates)
    
    predicted_response = np.concatenate(update_response).flatten()
    
    return predicted_response



def update_mixed_delta(dat,delta,hRate,nu_p):

    
    C = dat
    NumTrials = len(C)
    p_initial = 0.5
    
    
    # delta1 = delta1 
    # delta2 = delta2 
    # nodes = np.array([1, 1+delta1, delta1+delta2+1])
    
    nodes = np.array([1] + list(np.cumsum(delta) + 1))
    hRate = hRate #[0,1]
    nu_p = nu_p # [0,5]
    numNodes = len(nodes)
    numEndnode = numNodes - 1
    
    
    p_estNode = np.zeros((NumTrials, numNodes))  # Probability estimate at each node
    likelihood_dataNode = np.zeros((NumTrials, numNodes))  # Likelihood of data at each node
    changept_prior = np.zeros((numNodes, numNodes))  # Change-point prior matrix
    p_nodeWeight = np.zeros((NumTrials, numNodes))  # Weights of each node
    p_estimate = np.zeros(NumTrials)  # Final probability estimate
    
    # Initial conditions
    p_estNode[0, :] = p_initial
   # Set the first value to 1 (assuming the change occurred at the first node)
    p_nodeWeight[0, 0] = 1
   
    p_nodeWeight[0, 1:] = 0
    p_estimate[0] = np.dot(p_estNode[0, :], p_nodeWeight[0, :])
    
    # Initialize previous outcome
    Cprev = C[0]
    
    for t in range(1, NumTrials):
        for idx_node in range(numNodes):
            
            # Wilson et al., 2013, Eq (24)
            # Update probability estimate at each node
            p_estNode[t, idx_node] = p_estNode[t-1, idx_node] + (1 / (nodes[idx_node] + nu_p)) * (Cprev - p_estNode[t-1, idx_node])
            
            # Compute likelihood of new data at each node
            likelihood_dataNode[t, idx_node] = (
                p_estNode[t, idx_node] if Cprev == 1 else 1 - p_estNode[t, idx_node]
            )
            
            # Compute the change-point prior
            endNodeState = nodes[idx_node] # Subscript i in text
            
            for idx_startNode in range(numNodes): # Subscript j in text
                startNodeState = nodes[idx_startNode]
                
                if idx_node == 0:
                    p_change = 1
                else:
                    p_change = 0
                    
                    
                if idx_startNode != numEndnode:
                    if idx_node == idx_startNode:
                        #  % Wilson et al 2013, eq (29) , i = j
                        p_noChange = (nodes[idx_startNode + 1] - nodes[idx_startNode] - 1) / (nodes[idx_startNode + 1] - nodes[idx_startNode])
                        
                    elif idx_node == idx_startNode + 1:
                        #  Wilson et al 2013, eq (29) , i = j+1
                        p_noChange = 1 / (nodes[idx_startNode + 1] - nodes[idx_startNode])
                    else:
                        p_noChange = 0
                
                elif idx_startNode == numEndnode:
                    # Wilson et al 2013, eq (31)
                    # Self-transition probability at final node
                    p_noChange = 1 if idx_node == numEndnode else 0
                    
                    
                # Take a weighted average of the change and no change
                # probabilities, weighted by h and (1-h), respectively
                # Wilson et al 2013, eq (26)           
                changept_prior[idx_node, idx_startNode] = hRate * p_change + (1 - hRate) * p_noChange
    
        # Update weight for each node's probability estimate
        # Wilson et al 2013, eq (25)
        p_nodeWeight[t, :] = likelihood_dataNode[t, :] * np.dot(changept_prior, p_nodeWeight[t-1, :])
        
        # Normalize weights so they sum to 1
        p_nodeWeight[t, :] /= np.sum(p_nodeWeight[t, :])
        
        # Take a weighted average of the probability estimates for each node
        p_estimate[t] = np.dot(p_estNode[t, :], p_nodeWeight[t, :])
        
        Cprev = C[t]
        
    return p_estimate
        
        
        




