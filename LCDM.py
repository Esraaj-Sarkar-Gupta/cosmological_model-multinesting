#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 2025
@author: Esraaj Sarkar Gupta

Cosmological model multinesting: LCDM.py
This file holds relavant functions for the Lambda CDM
model of the universe.

Functions available:
    > Hubble parameter from the matter-dominated epoch
    > Uniform prior transform limits (dict)
    
All variables of the form _VARIABLE_NAME are editable.
"""

import numpy as np

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

# ==================
# Prior Transforms
# ==================

_PARAM_ORDER = ("H0", "Omega_m")
_PRIORS= dict({
    "H0" : (40.0, 100.0),
    "Omega_m" : (0.01, 0.8)
    })

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
class data:
    def __init__(self, model_name, parameters, prior_dict):
        self.model_name = model_name
        self.param_names = parameters
        self.prior_dict = prior_dict
    
    def get_model_name(self) -> str :
        return self.model_name
    
    # ---- Parameter Data ---- #
    
    def get_param_names(self) -> list :
        return self.parameters
    
    def get_priors(self) -> list :
        prior_limits = list([])
        for key in self.prior_dict:
            prior_limits.append(tuple(self.prior_dict[key]))
        return prior_limits
    
    # ---- ----
    
    def get_number_of_parameters(self) -> int :
        return len(self.parameters)
        
    
    
# ---- Call Data Structure ---- # 
def getData() -> data:
    return data(_MODEL_NAME, _PARAM_ORDER, _PRIORS)

# ---- Generic Function Nomenclature ---- #

def primary_function(free_parameter, constraint_parameters):
    Hz(free_parameter, constraint_parameters)
    