"""
Models.LsCDM

Created February, 2026
By Esraaj Sarkar Gupta

Cosmological model multinesting: LsCDM.py holds the equations
to the abrupt sign-switching LCDM and smooth sign-switching LCDM
models of the universe.

References:
    - Relaxing cosmological tensions with a sign switching cosmological constant
        by Akarsu et. al.
    - LsCDM cosmology from a type-II minimally modified gravity
        by Akarsu  et. al.

Supported Datasets:
    ccData
    DESI BAO

Model updates must be made in
    main.py
    loglike.py
    ESG_analyzer.py
"""

import numpy as np
from scipy.integrate import quad

#  Physical Constants
c = 299792.458 # km/s

# Model Name
_MODEL_NAME = str("Matter Dominated Sign Switching LsCDM")

# ====================
# Governing Equations
# ====================

# ---- Hubble Parameter ---- #

def Hz(free_parameter, constraint_parameters):
    z = free_parameter
    H0, Omega_m, z_shift = constraint_parameters

    lambda_s0 = 1.0 - Omega_m # Usual cosmological constant

    # Vectorized sign function
    sgn_function = np.where(z_shift >= z, 1.0, -1.0)

    lambda_s = lambda_s0 * sgn_function # Abrupt sign switching cosmological constant

    Hz_val = H0 * np.sqrt(Omega_m * (1.0 + z)**3 + lambda_s)

    # Return scalar if scalar input
    if np.isscalar(free_parameter):
        return Hz_val.item()
    return Hz_val

_PARAM_ORDER = ("H0", "Omega_m", "z_shift")
_PRIORS= dict({
    "H0" : (50.0, 80.0),
    "Omega_m" : (0.01, 0.8),
    "z_shift" : (1.0, 3.0) # Free
    })

"""
NOTE: We give z_shift a free prior here.
"""

def prior_transform(cube):
    """
    In-place map: X ~ U(0,1) -> a + x(b-a).
    Applied to first `n` entries of `cube` in order.
    """
    for i, name in enumerate(_PARAM_ORDER):
        a,b = _PRIORS[name]
        cube[i] = a + cube[i] * (b - a)

# ===============
# Data Structure
# ===============
"""
Model parameters are packaged into a callable object.

This object is used in generating figures in the
analysis module ESG_analyzer.py
"""
class data:
    def __init__(self, model_name, parameters, prior_dict):
        self.model_name = model_name
        self.param_names = parameters
        self.prior_dict = prior_dict
    
    def get_model_name(self) -> str :
        return self.model_name
    
    # ---- Parameter Data ---- #
    
    def get_param_names(self) -> list :
        return self.param_names
    
    def get_priors(self) -> list :
        prior_limits = list([])
        for key in self.prior_dict:
            prior_limits.append(tuple(self.prior_dict[key]))
        return prior_limits
    
    # ---- ----
    
    def get_number_of_parameters(self) -> int :
        return len(self.param_names)
    

# ---- Call Data Structure ---- # 
def getData() -> data:
    return data(_MODEL_NAME, _PARAM_ORDER, _PRIORS)

# ---- Generic Function Nomenclature ---- #

def primary_function(free_parameter, constraint_parameters):
    Hz(free_parameter, constraint_parameters)
    
def secondary_function() -> None:
    return None # No secondary funtion defined in this module
    