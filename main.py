#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 2025
@author: Esraaj

Flat ΛCDM constraints from 32 CC H(z) data using PyMultiNest.
"""

import time
import numpy as np
import pymultinest
from pathlib import Path

import ccData as data
import LCDM as model
import loglike as loglike
import ESG_analyzer as anal # oopsie


def main():
    # ---- Load Data ---- #
    cc = data.getCCData()
    z, Hz_obs, sig = cc.z, cc.H, cc.std_err # Unpack cc object
    print(f"\n[MM]: Loaded {len(z)} CC points (z in interval [{z.min():.3f}, {z.max():.3f}])")
    
    # ---- Check for output directory ---- #
    outdir = Path("chains")
    outdir.mkdir(parents = True, exist_ok = True)
    basename = str(outdir / "mn_cc_")

    # ---- Run MultiNest ---- #
    print("\n[MM]: Running PyMultiNest... this may take a bit.\n")
    pymultinest.run(
        loglike.gaussian_loglike,       # Log likelihood
        loglike.prior_transform,        # Prior transform
        n_dims=2,                       # Dimensionality of parameter space
        outputfiles_basename=basename,  # Basename for output files
        n_live_points=1000,             # Number of live points
        verbose=True,                   # Verbose
        seed=-1,                        # Sampling seed
        resume=False,                   # Resume (continue from a previous sampling question mark)
    )
    print(f"[MM]: Finished running PyMultiNest.")
    
    query = input("Continue with analyzer? [Y/n]: ")
    if (query.lower() != 'n'):
        pass
    else:
        return
    
    # ---- Run Native Analyzer ---- #
    analyzer = pymultinest.Analyzer(n_params=2, outputfiles_basename=basename)  # Create analysis object -- here ndims = nparams = 2
    
    posterior_samples = analyzer.get_equal_weighted_posterior() # Get equally weighted posteriors
    posterior_chains = analyzer.get_data()                      # Get posterior chains from data
    
    # ---- Run ESG Analyzer ---- #
    anal.contour_plot(posterior_samples) # Generate contour plot
    anal.contour_plot_3D(posterior_samples)
    
    anal.corner_plot(posterior_samples)
    anal.one_d_marginals(posterior_samples)
    
# ---- Run Main ---- #
if __name__ == "__main__":
    main()