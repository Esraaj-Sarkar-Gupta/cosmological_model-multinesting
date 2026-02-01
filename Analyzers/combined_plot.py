#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mainMulti.py

Created Oct 2025
@author: Esraaj

Multi-Probe Cosmology Analysis:
Runs 4 distinct chains (CC, DESI, SN, Combined) all constrained by Planck Priors.
Finally, generates a joint contour plot comparing all results.
"""

import numpy as np
import pymultinest
from pathlib import Path

import loglike as loglike
import ESG_analyzer as anal 

# Global variable to be accessed by ESG_analyzer (Backward compatibility)
DATASETNAME = "Comparison"

def getDatasetName() -> str:
    return DATASETNAME

def main():    
    # ---- Output Directory ---- #
    outdir = Path("chains")
    outdir.mkdir(parents = True, exist_ok = True)
    
    # ---- Configuration List ---- #
    # Format: (Name, Likelihood Function, Dimensions)
    run_configs = [
        ("CC+Planck",   loglike.planck_gaussian_loglike_covmat_CC, 2),
        ("DESI+Planck", loglike.planck_desi_loglike,               2),
        ("SN+Planck",   loglike.planck_sn_loglike,                 3),
        ("Combined",    loglike.planck_all_loglike,                3)
    ]
    
    # Dictionary to store results
    all_posteriors = {}

    print(f"\n[MM]: Starting Multi-Probe Analysis Pipeline...")
    print(f"[MM]: {len(run_configs)} jobs to run.")

    # ---- Main Loop ---- #
    for name, likelihood, dims in run_configs:
        
        global DATASETNAME
        DATASETNAME = name
        
        basename = str(outdir / f"mn_{name}_")
        
        print(f"\n" + "="*40)
        print(f"[MM]: Job Start: {name} (Dims={dims})")
        print("="*40)
        
        
        # -- Run Sampler -- #
        pymultinest.run(
            likelihood,
            loglike.prior_transform,
            n_dims=dims,
            outputfiles_basename=basename,
            n_live_points=800, # Slightly reduced for speed, increase for final run
            verbose=False,     # Reduce clutter
            seed=42,
            resume=False
        )
        
        # -- Extract Data -- #
        analyzer = pymultinest.Analyzer(n_params=dims, outputfiles_basename=basename)
        posterior = analyzer.get_equal_weighted_posterior()
        
        # 3. Save to dict
        all_posteriors[name] = posterior
        
        # 4. Generate Individual Corner Plot (Optional, good for sanity check)
        print(f"[MM]: Generating individual plots for {name}...")
        anal.corner_plot(posterior)
    
    # ---- Final Comparison ---- #
    print("\n" + "="*40)
    print(f"[MM]: All jobs finished. Generating Joint Plot...")
    print("="*40)
    
    # Set global name for the final folder creation
    DATASETNAME = "Combined_Comparison" 
    
    # Call the new comparison function
    anal.comparison_plot_hist2d(all_posteriors)
    
    print("\n[MM]: Analysis Complete.")

# ---- Run Main ---- #
if __name__ == "__main__":
    main()