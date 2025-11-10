#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 2025
@author: Esraaj Sarkar Gupta

Cosmological model multinesting: data.py

data.py imports data from cosmological databases.
"""
import numpy as np
import os

# ---- Find and Read Data Files ---- #

pwd = os.getcwd()
path_data = pwd + "/ccData/HzTable_MM_BC32.txt"
path_bias = pwd + "/ccData/data_MM20.dat"

_DATA = np.genfromtxt(path_data) # Extract CC data
zmod, imf, slib, sps, spsooo = np.genfromtxt(path_bias, comments='#', usecols=(0,1,2,3,4), unpack=True) # Extract bias data


#z = data[:,0]
#Hz = data[:,1]
#sig = data[:,2]

# ---- Covariance Matrices ---- #
"""
Each CC measurement has its own statistical noise -- these
are handled by the diagonal component of the covariance matrix.

The systemic uncertainty is encoded for in the off-diaginal
components. The covariance matrix (covmat) encodes for four distinct
systemic biases:
    Initial Mass Function bias -- (imf),
    Stellar Library bias -- (slib),
    Stellar Population Synthesis Model bias -- (sps)
    SPS Model (odd-one-out) -- spsooo

NOTE: The SPSOOO systemic is similar to the SPS systemic, however it computes
variations between SPS codes excluding the most discrepant model (unlike SPS).
It thus yields a more conservative systemic.

Cite: Setting the Stage for Cosmic Chronometers II. Impact of Stellar Population
Synthesis Models Systematics and Full Covariance Matrix -- Moresco et. al. 2020
"""

N = len(_DATA[:,0])

# Init and populate diagonal component of the covmat
covmat_diag = np.zeros((N, N), dtype='float64')
for i in range(N):
    covmat_diag[i,i] = _DATA[i,2] **2
    
# Init and populate systemic covmats
z = _DATA[:, 0]
H = _DATA[:, 1]

# Convert percentages to fractions
imf_intp    = np.interp(z, zmod, imf)/100
slib_intp   = np.interp(z, zmod, slib)/100
sps_intp    = np.interp(z, zmod, sps)/100
spsooo_intp = np.interp(z, zmod, spsooo)/100

# Initialise covmats
covmat_imf     = np.zeros((N,N), dtype='float64')
covmat_slib    = np.zeros((N,N), dtype='float64')
covmat_sps     = np.zeros((N,N), dtype='float64')
covmat_spsooo  = np.zeros((N,N), dtype='float64')

# Populate covmats
for i in range(len(z)):
	for j in range(len(z)):
		covmat_imf[i,j]      = H[i] * imf_intp[i]     * H[j] * imf_intp[j]
		covmat_slib[i,j]     = H[i] * slib_intp[i]    * H[j] * slib_intp[j]
		covmat_sps[i,j]      = H[i] * sps_intp[i]     * H[j] * sps_intp[j]
		covmat_spsooo[i,j]   = H[i] * spsooo_intp[i]  * H[j] * spsooo_intp[j]
        
# Build the final covariance matrix
_COVMAT = covmat_diag + covmat_imf + covmat_spsooo

# ---- Data Object ---- #
class CC:
    def __init__(self, data: np.ndarray, covmat: np.ndarray):
        self.data    = np.array(data, dtype=float)
        self.z          = self.data[:, 0]
        self.H          = self.data[:, 1]
        self.std_err    = self.data[:, 2]
        self.covmat     = covmat

def getCCData() -> CC:
    print(f"[MM_DAT]: 32CC Data loaded into object CC.")
    return CC(_DATA, _COVMAT)
    

