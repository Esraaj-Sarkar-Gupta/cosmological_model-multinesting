"""
cc_likelihoods.py

Created February 2026
@author: Esraaj Sarkar Gupta

Cosmic Chronometer likelihoods (loglikes)

    > Gaussian (diagonal) Loglike
    > Gaussian (covmat) Loglike

    > Planck Constraint diagonal loglike
    > Planck Constraint covmat loglike

Model Support: LCDM, LsCDM

"""

import numpy as np
from likelihoods import planck_likelihoods

from Models import LsCDM as model

# ---- Data Imports ---- #

from Datasets import ccData, desibaoData, supernovaData

# =========================
# Cosmic Chronometer Data 
# =========================

struct = ccData.getCCData()

z       = struct.z
obs     = struct.H
std     = struct.std_err
covmat  = struct.covmat

# ---- CC Gaussian LogLike Constants ---- #
log_norm_diag = -0.5 * np.sum(np.log(2 * np.pi * std**2))


# ---- CC Covariance Matrix Handling ---- #
try:
    cc_inv_cov = np.linalg.inv(covmat)
    sign, cc_logdet = np.linalg.slogdet(covmat)
    if sign <= 0:
        raise ValueError("CC Covmat must be positive definite.")
            
except np.linalg.LinAlgError:
    print(f"[Loglike]: LinAlg error encountered during CC covmat inversion. Adding jitter and trying again...")
    diag_mean = np.mean(np.diag(covmat))
    jitter = 1e-12 * diag_mean
    try:
        cc_inv_cov = np.linalg.inv(covmat + jitter * np.eye(covmat.shape[0]))
        sign, cc_logdet = np.linalg.slogdet(covmat + jitter * np.eye(covmat.shape[0]))
        if sign <= 0:
            raise ValueError("CC Covmat must be positive definite.")
        print("[Loglike]: OK")
    except np.linalg.LinAlgError:
        print(f"[Loglike]: Failed to invert CC covmat. What kind of data are you feeding me?")


# =========================
# LogLikelihoods
# =========================

def gaussian_diagonal(
        cube,       
        ndim,       
        nparams,    
        ) -> float:
    """ Diagonal Gaussian Likelihood for CC Data """
    parameter_list = cube[0:nparams]
    
    predicted_data = model.Hz(z, parameter_list)
    
    chi2 = np.sum(((predicted_data - obs) / std) ** 2)
    return float(-0.5 * chi2 + log_norm_diag)

def gaussian_covmat(
        cube,
        ndim,
        nparams,
        ) -> float:
    """ Full Covariance Gaussian Likelihood for CC Data """
    parameter_list = cube[0:nparams]
    
    predicted_data = model.Hz(z, parameter_list)
    delta = predicted_data - obs
    
    chi2 = delta.T @ cc_inv_cov @ delta
    loglike = -0.5 * (chi2 + cc_logdet + len(obs) * np.log(2 * np.pi))
    
    return float(loglike)

# =========================
# Planck + CC Joint Loglikes
# =========================

def planck_gaussian_diagonal(
        cube,
        ndim,
        nparams
        ) -> float:
    if nparams != 2:
        raise Exception("[LogLike]: The Planck likelihood expects parameters (H0, Ωm).")
    return gaussian_diagonal(cube, ndim, nparams) + planck_likelihoods.planck_loglike(cube[0:nparams])

def planck_gaussian_covmat(
        cube,
        ndim,
        nparams
        ) -> float:
    if nparams != 2:
        raise Exception("[LogLike]: The Planck likelihood expects parameters (H0, Ωm).")
    return gaussian_covmat(cube, ndim, nparams) + planck_likelihoods.planck_loglike(cube[0:nparams])


