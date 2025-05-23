"""
This script provides functions for fitting HMM models to real or simulated data.

Author: @emilebdn  
Created date: 2025-05-20
"""
#%%
import sys

import numpy as np
import os.path as op
import scipy.optimize as so

from joblib import Parallel, delayed
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold

# Add the root of the repository to sys.path
sys.path.append(op.dirname(op.dirname(op.dirname(op.abspath(__file__)))))

import emilebdn.simulations.model_learner_pos as MP

from emilebdn.config.variables import (
    n_jobs,
    task_types, train_size_ratio,
    p_c_bounds,
    b_0, tau_0, n_initial_guesses,
    expID, model,
    std_dev_pos
)
from models.HMM_fit import HMM

#%%
def HMM_prediction(p_c, data, task, expID, b_0, tau_0, std_dev_pos, return_MSE=False):
    """
    Computes the mean squared error (MSE) or returns HMM model predictions.
    This function is inspired by data_analysis_utils.MSE_fun().

    Parameters
    ----------
    p_c : array-like or float
        HMM model parameter(s).
    data : pd.DataFrame
        Input data containing at least the 'session_idx' and 'outcome' columns.
    task : str
        Task name, either 'ada-pos' or 'ada-prob'.
    expID : str or int
        Experiment identifier.
    b_0 : float
        Prior belief about the mean (used for 'ada-pos').
    tau_0 : float
        Prior belief about the variance (used for 'ada-pos').
    std_dev_pos : float
        Standard deviation of the observations (used for 'ada-pos').
    return_MSE : bool, optional
        If True, return the mean squared error. If False, return predictions.

    Returns
    -------
    float or np.ndarray
        The mean MSE if return_MSE is True.
        An array of predictions if return_MSE is False.
    """
    if task not in task_types:
        raise ValueError(f"Error: task has to be {task_types[0]} or {task_types[1]}.")
    if expID != 3:
        raise ValueError("Error: expID has to be 3 for the moment.")
    
    MSEs = []
    predictions = np.array([], dtype=np.float64)

    for _, group in data.groupby('session_idx'):
        outcomes = group['outcome'].values

        if task == 'ada-pos':
            inference_result = MP.run_inference(
                outcomes,
                p_c=p_c,
                std_gen=std_dev_pos,
                b_0=b_0,
                tau_0=tau_0
            )

            estimates = inference_result['mean']

        if task == 'ada-prob':
            estimates = HMM(p_c,outcomes,expID)
            
        if return_MSE:
            mse = np.mean((estimates - outcomes) ** 2)
            MSEs.append(mse)
        else:
            predictions = np.append(predictions, estimates)

    if return_MSE:
        return np.mean(MSEs)
    else:
        return predictions

def get_initial_parameters_new(data, expID, model, task, p_c_bounds, b_0, tau_0, n_initial_guesses, std_dev_pos):
    """
    Generates a good initial guess for model parameters by random search minimizing MSE,
    supporting both 'ada-pos' and 'ada-prob' tasks with the 'HMM' model.
    This function is inspired by data_analysis_utils.get_initial_parameters().

    Parameters
    ----------
    data : pd.DataFrame
        Subject-level data containing 'session_idx' and 'outcome' columns.
    expID : str or int
        Experiment identifier.
    model : str
        Model name.
    task : str
        Task name, either 'ada-pos' or 'ada-prob'.
    p_c_bounds : dict
        Dictionary with parameter bounds for each task.
    b_0 : float
        Prior belief about the mean (used in 'ada-pos').
    tau_0 : float
        Prior belief about the variance (used in 'ada-pos').
    n_initial_guesses : int
        Number of random initial guesses to try.
    std_dev_pos : float
        Standard deviation of the observations.

    Returns
    -------
    p_c : np.ndarray
        Initial guess yielding the lowest MSE for the HMM model parameter.
    bounds : list of tuple
        Bounds for parameter optimization.
    """
    if expID != 3:
        raise ValueError("Error: expID has to be 3 for the moment.")
    if model != 'HMM':
        raise ValueError("Error: model has to be HMM for the moment.")
    if task not in task_types:
        raise ValueError(f"Error: task has to be {task_types[0]} or {task_types[1]}.")
    
    lb = np.array([p_c_bounds[task]['min']])
    ub = np.array([p_c_bounds[task]['max']])
    bounds = [(p_c_bounds[task]['min'], p_c_bounds[task]['max'])]

    best_MSE = np.inf
    p_c = None
    
    for _ in range(n_initial_guesses):
        p_c_rd = np.random.uniform(low=lb, high=ub)
        MSE = HMM_prediction(p_c_rd, data, task, expID, b_0, tau_0, std_dev_pos, return_MSE=True)
        if MSE < best_MSE:
            best_MSE = MSE
            p_c = p_c_rd

    print("Best initial parameter:", p_c)
    return p_c, bounds

def fit_HMM_for_each_subject(subj_idx, subject_data, task, expID, model, p_c_bounds, b_0, tau_0, \
                             n_initial_guesses, std_dev_pos):
    """
    Fits the HMM model to the provided subject data using multiple optimization restarts.

    This function is inspired by data_analysis_utils.fit_model().

    Parameters
    ----------
    subj_idx : int
        Subject index.
    subject_data : pd.DataFrame
        Data for the subject.
    task : str
        Task name, either 'ada-pos' or 'ada-prob'.
    expID : str or int
        Experiment identifier.
    model : str
        Model name.
    p_c_bounds : dict
        Bounds for the model parameter.
    b_0 : float
        Prior belief about the mean (used in 'ada-pos').
    tau_0 : float
        Prior belief about the variance (used in 'ada-pos').
    n_initial_guesses : int
        Number of random initial guesses to try.
    std_dev_pos : float
        Standard deviation of the observations.

    Returns
    -------
    result : bool
        Whether the optimization was successful.
    x_min : np.ndarray
        The optimized parameter values.
    fval : float
        The minimized loss function value.
    """
    if expID != 3:
        raise ValueError("Error: expID has to be 3 for the moment.")
    if model != 'HMM':
        raise ValueError("Error: model has to be HMM for the moment.")
    if task not in task_types:
        raise ValueError(f"Error: task has to be {task_types[0]} or {task_types[1]}.")
     
    print(f'ExpID: {expID}, Subject: {subj_idx}, Model: {model}, Task: {task}, Running optimization...')

    best_opt = None
    best_fun = np.inf

    for _ in range(10):
        fp_init, bounds = get_initial_parameters_new(subject_data, expID, model, task, p_c_bounds, b_0, tau_0,
                                                     n_initial_guesses, std_dev_pos)
        fp_init = np.array(fp_init)

        opt = so.minimize(
            HMM_prediction,
            fp_init,
            args=(subject_data, task, expID, b_0, tau_0, std_dev_pos, True),
            method='Powell',
            bounds=bounds,
            options={'disp': False}
        )

        if opt['fun'] < best_fun:
            best_fun = opt['fun']
            best_opt = opt

    opt = best_opt
    
    result = opt["success"]
    p_c_min = opt['x']
    fval = opt['fun']
    
    return result, p_c_min, fval

def fit_HMM_for_every_subject(data_outcome_level, task, n_jobs=n_jobs, expID=expID, model=model, \
                              p_c_bounds=p_c_bounds, b_0=b_0, tau_0=tau_0, \
                                n_initial_guesses=n_initial_guesses, std_dev_pos=std_dev_pos):
    """
    Fits the HMM model for every subject in the provided data for a specific task.

    Parameters
    ----------
    data_outcome_level : pd.DataFrame
        The full dataset containing all subjects and sessions.
    task : str
        The task name to filter the data.
    n_jobs : int, optional
        Number of parallel jobs to run (default is n_jobs from emilebdn.config.variables).
    expID : str or int, optional
        Experiment identifier (default is expID from emilebdn.config.variables).
    model : str, optional
        Model name (default is model from emilebdn.config.variables).
    p_c_bounds : dict, optional
        Bounds for the model parameters (default is p_c_bounds from emilebdn.config.variables).
    b_0 : float, optional
        Prior mean for inference (default is b_0 from emilebdn.config.variables).
    tau_0 : float, optional
        Prior precision for inference (default is tau_0 from emilebdn.config.variables).
    n_initial_guesses : int, optional
        Number of random initial guesses to try (default is n_initial_guesses from emilebdn.config.variables).
    std_dev_pos : float, optional
        Standard deviation parameter for inference (default is std_dev_pos from emilebdn.config.variables).

    Returns
    -------
    dict
        Dictionary mapping subject IDs to their optimized p_c values.
    """
    if task not in task_types:
        raise ValueError(f"Error: task has to be {task_types[0]} or {task_types[1]}.")
    
    task_data = data_outcome_level[data_outcome_level['task'] == task] 
    
    subjects = task_data['subject'].unique()
    subjects_data = {
        subject: task_data[task_data['subject'] == subject]
        for subject in subjects
    }

    p_c_fitted = Parallel(n_jobs=n_jobs)(
        delayed(fit_HMM_for_each_subject)(subject_id, subjects_data[subject], task, expID, model, p_c_bounds, b_0, tau_0, \
                                          n_initial_guesses, std_dev_pos)
        for subject_id, subject in enumerate(subjects)
    )
    
    # If p_c_fitted is a list/array with a single value, extract the value
    p_c_fitted_values = [pc_min[0] if isinstance(pc_min, (np.ndarray, list)) else pc_min \
                     for _, pc_min, _ in p_c_fitted]

    
    return dict(zip(subjects, p_c_fitted_values))

def compute_evf_for_subject(subject_id, subject_data, task, n_splits):
    """
    Compute the Explained Variance Score (EVF) for a single subject using cross-validation.
    """
    kf = KFold(n_splits=n_splits)
    evf_scores = []

    for train_index, test_index in kf.split(subject_data):
        train_data = subject_data.iloc[train_index]
        test_data = subject_data.iloc[test_index]

        # Fit HMM model
        p_c_min = fit_HMM_for_each_subject(
            subj_idx=subject_id,
            subject_data=train_data,
            task=task,
            expID=expID,
            model=model,
            p_c_bounds=p_c_bounds,
            b_0=b_0,
            tau_0=tau_0,
            n_initial_guesses=n_initial_guesses,
            std_dev_pos=std_dev_pos,
        )[1]

        # Predict on test set
        predictions = HMM_prediction(
            p_c=p_c_min,
            data=test_data,
            task=task,
            expID=expID,
            b_0=b_0,
            tau_0=tau_0,
            std_dev_pos=std_dev_pos,
            return_MSE=False
        )

        evf = explained_variance_score(test_data['outcome'], predictions)
        evf_scores.append(evf)

    return np.mean(evf_scores)

def compute_evf_for_all_subjects(data_outcome_level, task, n_splits=int(1/(1-train_size_ratio))):
    """
    Compute evf for all subjects using cross-validation.
    """
    if task not in task_types:
        raise ValueError(f"Task must be one of: {task_types}")
        
    task_data = data_outcome_level[data_outcome_level['task'] == task]
    subjects = task_data['subject'].unique()

    subjects_id = {
        subject: idx
        for idx, subject in enumerate(subjects)
    }

    subjects_data = {
        subject: task_data[task_data['subject'] == subject]
        for subject in subjects
    }

    evf_scores = {}
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_evf_for_subject)(subjects_id[subject], subjects_data[subject], task, n_splits)
        for subject in subjects
    )

    for subject, evf in zip(subjects, results):
        evf_scores[subject] = evf

    return evf_scores
