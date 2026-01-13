#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loglike.py

Created Oct 2025
@author: Esraaj Sarkar Gupta

This file contains the mathematical tools used in this project.

Functions:
    ** Planck 2018 Constraints **
    > Planck Constraint Loglike (Gaussian priors on H0 and Omega_m)

    ** Cosmic Chronometers (CC) **
    > Gaussian (diagonal) likelihood
    > Gaussian (covariance matrix) likelihood
    > CC + Planck Constraint (Diagonal & Covariance versions)

    ** DESI BAO **
    > DESI Likelihood (Full Covariance) -- Projects D_H/r_d data onto H(z)
    > DESI + Planck Constraint

    ** Pantheon+ Supernovae **
    > SN Likelihood (Full Covariance) -- Forward models Luminosity Distance to Magnitude.
      Includes marginalization/fitting of Nuisance Parameter 'M'.
    > SN + Planck Constraint

    ** Combined Likelihoods (The Holy Trinity) **
    > CC + DESI + Planck
    > CC + SN + Planck
    > DESI + SN + Planck
    > ALL + Planck

    ** Utilities ** 
    > Uniform prior transform -- Mirrors the function in the `model` module.
"""

import numpy as np

# Models
import LCDM as model

# Datasets
import ccData
import desibaoData
import supernovaData as snData

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


# ----  CC Likelihoods ---- #

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


# ---- CC + Planck LogLikelihood ---- #

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

# ---- DESI BAO  LogLikelihood ---- #
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

# ---- DESI + Planck LogLikelihood ---- #

def planck_desi_loglike(
        cube,
        ndim,
        nparams
    ) -> float:
    if (nparams != 2):
        raise Exception("[LogLike]: The planck likelihood expects parameters (2) H0 and Om.")
    return desi_loglike(cube, ndim, nparams) + planck_loglike(cube[0:nparams])

# ============================
# Supernova (Pantheon+) Data
# ============================

# ---- Load Data ---- #

sn_struct = snData.getPantheonPlusData()
sn_z = sn_struct.z
sn_mb = sn_struct.mb
sn_covmat = sn_struct.covmat
sn_inv_covmat = sn_struct.inv_covmat
sn_logdet = getattr(sn_struct, 'logdet', 0.0) # Some gaurdrails :3

# ---- SN LogLikelihood ---- #
def sn_loglike(
        cube,
        ndim,
        nparams
) -> float :
    if (nparams != 3):
        raise Exception("[LogLike]: The planck likelihood expects parameters (3) H0, Om and M.")
    
    # Cosmological Parameter
    params_cosmo = cube[0:2]

    # Nuisance Parameter
    M = cube[2]

    # Compute model magnitude
    """
    The physics:
    
    (observed peak magnitude) m
    
    m = 5.0 * log10(dL) + M + 25.0
    """
    dL = model.dL(sn_z, params_cosmo)
    m_model = 5.0 * np.log10(dL) + M + 25.0

    # Residuals
    delta = sn_mb - m_model
    
    #Chi2
    chi2 = delta.T @ sn_inv_covmat @ delta

    loglike = -0.5 * (chi2 + sn_logdet + len(sn_mb) * np.log(2 * np.pi))
    return float(loglike)

# ----- SN + Planck LogLikelihood ---- #

def planck_sn_loglike(
        cube,
        ndim,
        nparams
) -> float:
    if (nparams != 3):
        raise Exception("[LogLike]: The planck likelihood expects parameters (3) H0, Om and M.")
    
    return sn_loglike(cube, ndim, nparams) + planck_loglike(cube[0:nparams - 1])
    
# ==========================================
# Combined Likelihoods (Planck Constrained)
# ==========================================

# ---- CC + DESI + Planck ---- #
def planck_cc_desi_loglike(cube, ndim, nparams) -> float:
    if nparams != 2:
        raise Exception("[LogLike]: CC + DESI expects 2 parameters (H0, Om).")
        
    return (
        gaussian_loglike_covmat_CC(cube, ndim, nparams) +
        desi_loglike(cube, ndim, nparams) +
        planck_loglike(cube[0:nparams])
    )

# ---- CC + SN + Planck ---- #
def planck_cc_sn_loglike(cube, ndim, nparams) -> float:
    if nparams != 3:
        raise Exception("[LogLike]: CC + SN expects 3 parameters (H0, Om, M).")
    
    # CC and Planck only use the first 2 parameters (H0, Om)
    return (
        gaussian_loglike_covmat_CC(cube, ndim, 2) + 
        sn_loglike(cube, ndim, nparams) + 
        planck_loglike(cube[0:2])
    )

# ---- DESI + SN + Planck ---- #
def planck_desi_sn_loglike(cube, ndim, nparams) -> float:
    if nparams != 3:
        raise Exception("[LogLike]: DESI + SN expects 3 parameters (H0, Om, M).")
        
    return (
        desi_loglike(cube, ndim, 2) + 
        sn_loglike(cube, ndim, nparams) + 
        planck_loglike(cube[0:2])
    )

# ---- ALL (CC + DESI + SN) + Planck ---- #
def planck_all_loglike(cube, ndim, nparams) -> float:
    if nparams != 3:
        raise Exception("[LogLike]: Combined Analysis expects 3 parameters (H0, Om, M).")
        
    return (
        gaussian_loglike_covmat_CC(cube, ndim, 2) + 
        desi_loglike(cube, ndim, 2) + 
        sn_loglike(cube, ndim, nparams) + 
        planck_loglike(cube[0:2])
    )


# ================
# Prior Transform
# ================
def prior_transform(cube, ndim, nparams):
    """
    In-place mapping from unit cube to physical parameters via
    model.prior_transform().
    """
    model.prior_transform(cube)