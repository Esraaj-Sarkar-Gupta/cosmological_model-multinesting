"""
desibao_likelihoods.py

Created February 2026
@author: Esraaj Sarkar Gupta

DESI BAO likelihoods (loglikes)

    > Gaussian (covmat) Loglike
    > Planck constraint Gaussian Loglike

Model Support: LCDM, LsCDM
"""

import numpy as np
from likelihoods import planck_likelihoods
from Models import LsCDM as model

# ---- Data Imports ---- #
from Datasets import desibaoData

# =========================
# DESI BAO Data
# =========================

struct = desibaoData.getDESIData()

z       = struct.z
obs     = struct.H

# Prefer whatever the dataset object provides
covmat  = getattr(struct, "covmat", None)
inv_cov = getattr(struct, "inv_cov", None)

# ---- DESI Covariance Matrix Handling ---- #

if covmat is None:
    raise ValueError("[Loglike]: DESI struct is missing covmat (unexpected).")

covmat = 0.5 * (covmat + covmat.T)

def _compute_inv_and_logdet(C: np.ndarray):
    invC = np.linalg.inv(C)
    sign, ld = np.linalg.slogdet(C)
    if sign <= 0:
        raise np.linalg.LinAlgError("Covariance matrix not positive definite.")
    return invC, float(ld)

try:
    if inv_cov is None:
        inv_cov, logdet = _compute_inv_and_logdet(covmat)
    else:
        # inv_cov exists -> just compute logdet robustly from covmat
        sign, logdet = np.linalg.slogdet(covmat)
        if sign <= 0:
            raise np.linalg.LinAlgError("Covariance matrix not positive definite.")
        logdet = float(logdet)

except np.linalg.LinAlgError:
    print("[Loglike]: Issue with DESI covmat (inv/logdet). Adding jitter and trying again...")

    diag_mean = np.mean(np.diag(covmat))
    jitter = 1e-12 * diag_mean
    covmat_j = covmat + jitter * np.eye(covmat.shape[0])

    # Symmetrize again after jitter (paranoia, but harmless)
    covmat_j = 0.5 * (covmat_j + covmat_j.T)

    inv_cov, logdet = _compute_inv_and_logdet(covmat_j)
    covmat = covmat_j
    print("[Loglike]: OK")


# =========================
# LogLikelihoods
# =========================

def gaussian_covmat(
        cube,
        ndim,
        nparams,
        ) -> float:
    """ Full Covariance Gaussian Likelihood for DESI BAO Data """
    parameter_list = cube[0:nparams]

    predicted_data = model.Hz(z, parameter_list)
    delta = predicted_data - obs

    chi2 = delta.T @ inv_cov @ delta
    loglike = -0.5 * (chi2 + logdet + len(obs) * np.log(2 * np.pi))

    return float(loglike)

# =========================
# Planck + DESI Joint Loglikes
# =========================

def planck_gaussian_covmat(
        cube,
        ndim,
        nparams
        ) -> float:

    return gaussian_covmat(cube, ndim, nparams) + planck_likelihoods.planck_loglike(cube[0:nparams - 1])
