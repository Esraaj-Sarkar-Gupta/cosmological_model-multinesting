"""
supernova_likelihoods.py

Created February 2026
@author: Esraaj Sarkar Gupta

Supernova (Pantheon+) likelihoods (loglikes)

    > Gaussian (covmat) Loglike
    > Planck Constraint Gaussian Loglike

Model Support: LCDM
"""

import numpy as np
from likelihoods import planck_likelihoods
from Models import LCDM as model

# ---- Data Imports ---- #
from Datasets import supernovaData

# ============================
# Supernova (Pantheon+) Data
# ============================

struct = supernovaData.getPantheonPlusData()

z       = struct.z
obs     = struct.mb

covmat  = getattr(struct, "covmat", None)
inv_cov = getattr(struct, "inv_covmat", None)  # your struct uses inv_covmat naming
logdet  = getattr(struct, "logdet", None)

# ---- SN Covariance Matrix Handling ---- #
try:
    if inv_cov is None:
        if covmat is None:
            raise ValueError("[Loglike]: SN struct must provide either inv_covmat or covmat.")
        inv_cov = np.linalg.inv(covmat)

    if logdet is None:
        if covmat is None:
            logdet = 0.0
        else:
            sign, logdet = np.linalg.slogdet(covmat)
            if sign <= 0:
                raise ValueError("SN covmat must be positive definite.")
    else:
        logdet = float(logdet)

except np.linalg.LinAlgError:
    print("[Loglike]: LinAlg error during SN covmat inversion. Adding jitter and trying again...")
    if covmat is None:
        raise ValueError("[Loglike]: SN covmat missing; cannot jitter-invert.")
    diag_mean = np.mean(np.diag(covmat))
    jitter = 1e-12 * diag_mean
    inv_cov = np.linalg.inv(covmat + jitter * np.eye(covmat.shape[0]))
    sign, logdet = np.linalg.slogdet(covmat + jitter * np.eye(covmat.shape[0]))
    if sign <= 0:
        raise ValueError("SN covmat must be positive definite.")
    print("[Loglike]: OK")

# =========================
# LogLikelihoods
# =========================

def gaussian_covmat(
        cube,
        ndim,
        nparams
) -> float:
    """
    Full Covariance Gaussian Likelihood for Pantheon+ SN.

    Parameters expected: (H0, Ωm, M)
    """
    if nparams != 3:
        raise Exception("[LogLike]: Supernova likelihood expects parameters (H0, Ωm, M).")

    # Cosmological params
    params_cosmo = cube[0:2]

    # Nuisance param
    M = cube[2]

    # Model apparent magnitude:
    # m = 5 log10(dL) + M + 25
    dL = model.dL(z, params_cosmo)
    m_model = 5.0 * np.log10(dL) + M + 25.0

    delta = obs - m_model

    chi2 = delta.T @ inv_cov @ delta
    loglike = -0.5 * (chi2 + logdet + len(obs) * np.log(2 * np.pi))

    return float(loglike)

# =========================
# Planck + SN Joint Loglikes
# =========================

def planck_gaussian_covmat(
        cube,
        ndim,
        nparams
) -> float:
    if nparams != 3:
        raise Exception("[LogLike]: Planck+SN expects parameters (H0, Ωm, M).")

    # Planck uses only (H0, Ωm)
    return gaussian_covmat(cube, ndim, nparams) + planck_likelihoods.planck_loglike(cube[0:2])
