"""
This module provides functions for fitting Hidden Markov Models (HMM).
It includes utilities for predicting outcomes using HMM and generating predictions based on trained models.
The module supports the evaluation of model performance using metrics such as Mean Squared Error (MSE).

Author: @emilebdn
Created date: 2025-05-20
"""
import sys

import numpy as np
import os.path as op
import pandas as pd

sys.path.append(op.dirname(op.dirname(op.dirname(__file__))))

from data_analysis_utils import fit_model
from emilebdn.config.paths import data_outcome_level_preprocessed_path
from emilebdn.config.variables import (
    b_0, tau_0,
    expID, model,
    std_dev_pos,
    change_prob_prob
)
from models.HMM_fit import HMM

def HMM_prediction(p_c, data, expID=expID, b_0=b_0, tau_0=tau_0, std_dev_pos=std_dev_pos, return_MSE=False):
    """
    Computes the Mean Squared Error (MSE) or generates predictions using an HMM model.

    This function evaluates the performance of an HMM model by calculating the MSE between the predicted
    and actual outcomes or generates predictions based on the model parameters.

    Parameters
    ----------
    p_c : array-like or float
        HMM model parameter(s) used for prediction.
    data : pd.DataFrame
        Input data containing at least the 'outcome' column, representing the observed outcomes.
    expID : str or int, optional
        Experiment identifier, defaults to the value specified in the configuration.
    b_0 : float, optional
        Prior belief about the mean, used for certain tasks, defaults to the value specified in the configuration.
    tau_0 : float, optional
        Prior belief about the variance, used for certain tasks, defaults to the value specified in the configuration.
    std_dev_pos : float, optional
        Standard deviation of the observations, used for certain tasks, defaults to the value specified in the configuration.
    return_MSE : bool, optional
        If True, the function returns the mean squared error. If False, it returns the model predictions.

    Returns
    -------
    float or np.ndarray
        The mean MSE if return_MSE is True, otherwise an array of predictions.
    """
    if expID != 3:
        raise ValueError("Error: expID has to be 3 for the moment.")
    
    MSEs = []

    outcomes = data['outcome']

    if len(outcomes) == 0:
        raise ValueError("Error: 'outcomes' array is empty")

    estimates = HMM(p_c,outcomes,expID)
        
    if return_MSE:
        mse = np.mean((estimates - outcomes) ** 2)
        MSEs.append(mse)
        return np.mean(MSEs)
    else:
        return estimates

def predict_sequences_with_HMM(train_sequences, test_sequences, subj_idx=0, p_c_optimal = False):
    """
    Fits an HMM model on training sequences and predicts outcomes for test sequences.

    This function uses training data to fit an HMM model and then applies the fitted model to predict outcomes
    for new, unseen test data. It supports both subject-specific and optimal parameter configurations.

    Parameters
    ----------
    train_sequences : pd.DataFrame
        Training data containing sequences of outcomes and other relevant features.
    test_sequences : pd.DataFrame
        Test data for which outcomes need to be predicted.
    subj_idx : int, optional
        Subject index, used for subject-specific modeling, defaults to 0.
    p_c_optimal : bool, optional
        If True, uses the optimal probability for the change parameter. If False, fits the model to determine it.

    Returns
    -------
    np.ndarray
        Array of predictions for the test sequences.
    """
    data_outcome_level_preprocessed_bis = pd.read_csv(data_outcome_level_preprocessed_path)
    subjects = data_outcome_level_preprocessed_bis['subject'].unique()
    subjects_id = {subject: idx for idx, subject in enumerate(subjects)}

    sessions = train_sequences['session_idx'].unique()
    subjs = train_sequences['subject'].unique()
    subj_idx = [subjects_id[subj] for subj in subjs]

    if p_c_optimal == False:
        p_c_min = fit_model(expID, model, subj_idx, sessions)[1]
    elif p_c_optimal == True:
        p_c_min = change_prob_prob

    # Predict outcomes on test set
    predictions = HMM_prediction(
        p_c=p_c_min,
        data=test_sequences,
        expID=expID,
        b_0=b_0,
        tau_0=tau_0,
        std_dev_pos=std_dev_pos,
        return_MSE=False
    )

    return predictions