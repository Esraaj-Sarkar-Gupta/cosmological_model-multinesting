#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LCDM.py

Created Oct 2025
@author: Esraaj Sarkar Gupta

Cosmological model multinesting: LCDM.py
This file holds relavant functions for the Lambda CDM
model of the universe.

Cosmological Functions:
    > Hubble parameter from the matter-dominated epoch
    
Module Functions:
    > Uniform prior transform limits (dict) -- exported to pyMultiNest via loglike.py
    > data class -- Callable object to export data
    
All variables of the form _VARIABLE_NAME are editable.
"""

import numpy as np
from scipy.integrate import quad

#  Physical Constants
c = 299792.458 # km/s

# Model Name
_MODEL_NAME = str("Matter Dominated LCDM")

# ====================
# Governing Equations
# ====================

# ---- Hubble Parameter ---- #

def Hz(free_parameter, constraint_parameters):
    z = free_parameter
    H0, Omega_m = constraint_parameters
    return H0 * np.sqrt(Omega_m * (1.0 + z)**3 + (1.0 - Omega_m))

# ---- Luminosity Distance ---- #
"""
Supernova datasets require forward modelling 
"""

def dL(z_array : np.ndarray, constraint_parameters) -> np.ndarray:
    H0, Omega_m = constraint_parameters[0], constraint_parameters[1]

    # Precompute c/H0 
    hubble_distance = c/H0

    # Physics
    """
    dL = (1+z) c/H0 int (0, z) 1 / E(z) dz
    """
    def E_inv(z : float) -> float :
        return 1 / np.sqrt(Omega_m * (1.0 + z)**3 + (1.0 - Omega_m))
    
    dL_results : list = list([])
    for z in z_array:
        integral, err = quad(E_inv, 0, z)
        dL_results.append(
            (1.0 + z) * hubble_distance * integral
        )

    return np.array(dL_results)

# ==================
# Prior Transforms
# ==================

_PARAM_ORDER = ("H0", "Omega_m", "M")
_PRIORS= dict({
    "H0" : (40.0, 100.0),
    "Omega_m" : (0.01, 0.8),
    "M" : (-20.0, -18.0) # Absolute magnitude for supernova data
    })

"""
 NOTE: The prior for the absolute magnitude M here is chosen arbitrarily 
 around the value -19.25  which is the value used by the SH0ES team in their 
 2022 paper.

 Cite: A Comprehensive Measurement of the Local Value of the Hubble Constant with 
 1 km s−1 Mpc−1 Uncertainty from the Hubble Space Telescope and the SH0ES Team
 by Adam G. Riess et. al.
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
    