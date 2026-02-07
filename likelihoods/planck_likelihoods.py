"""cc_likelihoods.py

Created February 2026
@author: Esraaj Sarkar Gupta

Planck Likelihoods
    > Planck Constraint Gaussian Likelihood

"""

import numpy as np

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
    if len(parameter_list) != 2:
        raise Exception(f"[LogLike]: The Planck likelihood expects parameters (H0, Ωm).  Tried to pass {len(parameter_list)} parameters.")
    H0 : float = parameter_list[0]
    Om : float = parameter_list[1]
    
    term_H0 : float = (H0 - _planck_mu[0])**2 / _planck_sigma[0]**2
    term_Om : float = (Om - _planck_mu[1])**2 / _planck_sigma[1]**2
    
    return - 0.5 * (term_H0 + term_Om)