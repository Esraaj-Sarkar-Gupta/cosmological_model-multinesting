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
import ccData
import desibaoData

# ================================
# Planck 2018 Fiducial Constraint
# ================================

PLANCK_H0_MEAN   = 67.36     # km s^-1 Mpc^-1
PLANCK_H0_SIGMA  = 0.54

PLANCK_OM_MEAN   = 0.3158
PLANCK_OM_SIGMA  = 0.00738

_planck_mu      = np.array([PLANCK_H0_MEAN, PLANCK_OM_MEAN])
_planck_sigma   = np.array([PLANCK_H0_SIGMA, PLANCK_OM_SIGMA])

def planck_loglike(parameter_list : list) -> float :
    """
    Computes the log-likelihood of the parameters given the Planck 2018 fiducial values.
    """
    H0 : float = parameter_list[0]
    Om : float = parameter_list[1]
    
    term_H0 : float = (H0 - _planck_mu[0])**2 / _planck_sigma[0]**2
    term_Om : float = (Om - _planck_mu[1])**2 / _planck_sigma[1]**2
    
    return - 0.5 * (term_H0 + term_Om)

# =========================
# Cosmic Chronometer Data 
# =========================

cc_struct = ccData.getCCData()

cc_z = cc_struct.z
cc_obs = cc_struct.H
cc_std = cc_struct.std_err
cc_covmat = cc_struct.covmat

# ---- CC Gaussian LogLike Constants ---- #
cc_log_norm_diag = -0.5 * np.sum(np.log(2 * np.pi * cc_std**2))

# ---- CC Covariance Matrix Handling ---- #
try:
    cc_inv_cov = np.linalg.inv(cc_covmat)
    sign, cc_logdet = np.linalg.slogdet(cc_covmat)
    if sign <= 0:
        raise ValueError("CC Covmat must be positive definite.")
            
except np.linalg.LinAlgError:
    # Handle ill-conditioned covmats with a tiny jitter
    print(f"[Loglike]: LinAlg error encountered during CC covmat inversion. Adding jitter and trying again...")
    diag_mean = np.mean(np.diag(cc_covmat))
    jitter = 1e-12 * diag_mean
    try:
        cc_inv_cov = np.linalg.inv(cc_covmat + jitter * np.eye(cc_covmat.shape[0]))
        sign, cc_logdet = np.linalg.slogdet(cc_covmat + jitter * np.eye(cc_covmat.shape[0]))
        if sign <= 0:
            raise ValueError("CC Covmat must be positive definite.")
        print("[Loglike]: OK")
    except np.linalg.LinAlgError:
        print(f"[Loglike]: Failed to invert CC covmat. What kind of data are you feeding me?")


# ----  Gaussian Likelihoods ---- #

def gaussian_loglike_CC(
        cube,       
        ndim,       
        nparams,    
        ) -> float:
    """ Diagonal Gaussian Likelihood for CC Data """
    parameter_list = cube[0:nparams]
    
    predicted_data = model.Hz(
        cc_z,               # CC redshifts
        parameter_list,     
        )
    
    chi2 = np.sum(((predicted_data - cc_obs) / cc_std) ** 2)
    return float(- 0.5 * chi2 + cc_log_norm_diag)


def gaussian_loglike_covmat_CC(
        cube,
        ndim,
        nparams,
        ) -> float:
    """ Full Covariance Gaussian Likelihood for CC Data """
    parameter_list = cube[0:nparams]
    
    predicted_data = model.Hz(
        cc_z,               # CC redshifts
        parameter_list,     
        )
    
    delta = predicted_data - cc_obs
    
    chi2 = delta.T @ cc_inv_cov @ delta
    loglike = -0.5 * (chi2 + cc_logdet + len(cc_obs) * np.log(2 * np.pi))
    
    return float(loglike)


# ---- CC + Planck ---- #

def planck_gaussian_loglike_CC(
        cube,
        ndim,
        nparams
        ) -> float :
    if (nparams != 2):
        raise Exception("[LogLike]: The planck likelihood expects parameters (2) H0 and Om.")
    return gaussian_loglike_CC(cube, ndim, nparams) + planck_loglike(cube[0:nparams])
 
def planck_gaussian_loglike_covmat_CC(
        cube,
        ndim,
        nparams
    ) -> float :
    if (nparams != 2):
        raise Exception("[LogLike]: The planck likelihood expects parameters (2) H0 and Om.")
    return gaussian_loglike_covmat_CC(cube, ndim, nparams) + planck_loglike(cube[0:nparams])

# ==============
# DESI BAO Data
# ==============

desi_struct = desibaoData.getDESIData()

desi_z = desi_struct.z
desi_obs = desi_struct.H
desi_inv_cov = desi_struct.inv_cov

# Handle log determinant for DESI
# (Assuming desibaoData handles inversion, we calculate logdet here or use the struct if available)
try:
    sign, desi_logdet = np.linalg.slogdet(desi_struct.covmat)
    if sign <= 0:
         print("[LogLike]: Warning! DESI Covariance matrix determinant is not positive.")
except Exception:
    desi_logdet = 0.0 # Fallback if needed


def desi_loglike(
        cube,
        ndim,
        nparams
        ) -> float:
    """ Full Covariance Gaussian Likelihood for DESI BAO Data """
    parameter_list = cube[0:nparams]
    
    predicted_data = model.Hz(
        desi_z,             # DESI redshifts
        parameter_list
    )
    
    delta = predicted_data - desi_obs
    
    chi2 = delta.T @ desi_inv_cov @ delta
    loglike = -0.5 * (chi2 + desi_logdet + len(desi_obs) * np.log(2 * np.pi))
    
    return float(loglike)

# ---- DESI + Planck ---- #

def planck_desi_loglike(
        cube,
        ndim,
        nparams
    ) -> float:
    if (nparams != 2):
        raise Exception("[LogLike]: The planck likelihood expects parameters (2) H0 and Om.")
    return desi_loglike(cube, ndim, nparams) + planck_loglike(cube[0:nparams])


# ================
# Prior Transform
# ================
def prior_transform(cube, ndim, nparams):
    """
    In-place mapping from unit cube to physical parameters via
    model.prior_transform().
    """
    model.prior_transform(cube)