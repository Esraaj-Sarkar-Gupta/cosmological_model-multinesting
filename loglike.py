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
    
    > Planck Constraint Loglike
    > Planck Constraint Gaussian Loglike (uses Planck constraint loglike)
    > Planck Constraint Gaussian Loglike (with covariance matrix) -- Uses diagonal covmat in parameter space for Planck Constraint
    
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
    
# ===========================
# Planck 2018 Fiducial Model
# ===========================

PLANCK_H0_MEAN   = 67.36     # km s^-1 Mpc^-1
PLANCK_H0_SIGMA  = 0.54

PLANCK_OM_MEAN   = 0.3158
PLANCK_OM_SIGMA  = 0.00738

_planck_mu      = np.array([PLANCK_H0_MEAN, PLANCK_OM_MEAN])
_planck_sigma   = np.array([PLANCK_H0_SIGMA, PLANCK_OM_SIGMA])

# ---- Planck Diagonal Likelihood ---- #
def planck_loglike(parameter_list : list) -> float :
    """
    This function assumes that the parameters passed are relevant
    only in the context of the fiducial cosmological model given in
    Planck 2018.
    """
    
    H0 : float = parameter_list[0]
    Om : float = parameter_list[1]
    
    term_H0 : float = (H0 - _planck_mu[0])**2 / _planck_sigma[0]**2
    term_Om : float = (Om - _planck_mu[1])**2 / _planck_sigma[1]**2
    
    return - 0.5 * (term_H0 + term_Om)


# ---- Gaussian Likelihood + Planck Constraint ---- #

def planck_gaussian_loglike(
        cube,
        ndim,
        nparams
        ) -> float :
    if (nparams != 2):
        raise Exception("[LogLike]: The planck likelihood expects parameters (2) H0 and Om to constrain your model to the fiducial cosmological model.")
    return gaussian_loglike(cube, ndim, nparams) + planck_loglike(cube[0:nparams])
 
def planck_gaussian_loglike_covmat(
        cube,
        ndim,
        nparams
    ) -> float :
    if (nparams != 2):
        raise Exception("[LogLike]: The planck likelihood expects parameters (2) H0 and Om to constrain your model to the fiducial cosmological model.")
    return gaussian_loglike_covmat(cube, ndim, nparams) + planck_loglike(cube[0:nparams])

# ================
# Prior Transform
# ================
def prior_transform(cube, ndim, nparams):
    """
    In-place mapping from unit cube to physical parameters via
    model.prior_transform(). Nothing to return
    """
    model.prior_transform(cube)
    
    















