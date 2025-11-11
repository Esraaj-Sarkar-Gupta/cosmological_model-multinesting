#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loglike.py

Created Oct 2025
@author: Esraaj Sarkar Gupta

This file contains the mathematical tools used in this project.

Functions:
    > Guassian (diagonal) likelihood
    > Gaussian (covariance matrix) likelihood -- as of now does not use Cholesky matrices
    
    > Uniform prior transform -- Mirrors the function in the `model` module.
"""

import numpy as np
import time as t

import LCDM as model
import ccData as data

# ---- Load data ---- #
data_struct = data.getCCData() # Expect verbose
# ---- Unpack data structure ----
free_parameter = data_struct.z
observed_data = data_struct.H
data_std_deviation = data_struct.std_err
covmat = data_struct.covmat

# ---- Gaussian LogLike Constants ---- #
log_norm_diag = -0.5 * np.sum(np.log(2 * np.pi * data_std_deviation**2))

# ---- Covariance Matrix Handling ---- #

try:
    inv_cov = np.linalg.inv(covmat)
    sign, logdet = np.linalg.slogdet(covmat)
    if sign <= 0:
        raise ValueError("Covmat must be positive definite.")
            
except np.linalg.LinAlgError:
    # Handle ill-conditioned covmats with a tiny jitter
    print(f"[Loglike]: LinAlg error encountered during covmat inversion. Adding jitter and trying again...")
    diag_mean = np.mean(np.diag(covmat))
    jitter = 1e-12 * diag_mean
    try:
        inv_cov = np.linalg.inv(covmat + jitter * np.eye(covmat.shape[0]))
        sign, logdet = np.linalg.slogdet(covmat + jitter * np.eye(covmat.shape[0]))
        if sign <= 0:
            raise ValueError("Covmat must be positive definite.")
        print("[Loglike]: OK")
    except np.linalg.LinAlgError:
        print(f"[Loglike]: Failed to invert covmat. What kind of matrix did you feed into me?")
    



# ========================
# Gaussian log likelihoods
# ========================

# ---- Diagonal log-likelihood ---- #

def gaussian_loglike(
        cube,       # Data structure
        ndim,       # Dimensionality of parameter space
        nparams,    # Number of parameters
        ) -> float:
    parameter_list = cube[0:nparams]
    
    predicted_data = model.Hz(
        free_parameter,     # Free parameter in the function (here z)
        parameter_list,     # Constraint parameter list
        )
    
    chi2 = np.sum(((predicted_data - observed_data) / data_std_deviation) ** 2)
    return float(- 0.5 * chi2 + log_norm_diag)

# ---- Full-covariance Gaussian log-likelihood (NON - Cholesky) ---- #


    
def gaussian_loglike_covmat(
        cube,
        ndim,
        nparams,
        ) -> float:
    parameter_list = cube[0:nparams]
    
    predicted_data = model.Hz(
        free_parameter,     # Free parameter in the function (here z)
        parameter_list,     # Constraint parameter list
        )
    
    delta = predicted_data - observed_data
    
    chi2 = delta.T @ inv_cov @ delta
    loglike = -0.5 * (chi2 + logdet + len(observed_data) * np.log(2 * np.pi))
    
    return float(loglike)
    
    

# ================
# Prior Transform
# ================
def prior_transform(cube, ndim, nparams):
    """
    In-place mapping from unit cube to physical parameters via
    model.prior_transform(). Nothing to return
    """
    model.prior_transform(cube)
    
    















