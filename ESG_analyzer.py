#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2025
@author: esraaj

Non-native pyMultiNest Analyzer.

Functions available:
    ** Statistical Values **
    > posterior_mean @return posterior mean
    > posterior_std_deviation (Posterior standard deviation) @return 
    > goodness_of_fit @returns best fit vector
    ** Data Visualisation **
    > contour_plot (for 2D parameter space)
    > contour_plot_3D (3D contour plot -- I would say this is a very useless function)
    >
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import corner
import os

import LCDM as model
import ccData as data

# ---- Results Directory ----

_ANAL_RESULT_DIR = "./ESG_analyzer_nocovmat/" # <-- Do not forget these slashes

# ---- Helpers ---- #
def saveDir(dir_name : str) -> str:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"[ESG_ANALYSIS]: Result directory {dir_name} created.")
    else:
        pass
    return dir_name

# =========================
# Useful Statistical Values
# =========================

def posterior_means(posterior: np.ndarray) -> np.ndarray :
    """
    @returns: a numpy array of the posterior means of each parameter sampled.
    """
    model_data = model.getData()
    nparam = model_data.get_number_of_parameters()
    
    means = list([])
    
    for n in range(nparam):
        means.append(np.mean(posterior[:,n]))
        
    return np.array(means)

def posterior_std_deviations(posterior : np.ndarray) -> np.ndarray :
    """
    @returns: a numpy array of the posterior standard deviations of each parameter sampled.
    """
    model_data = model.getData()
    nparam = model_data.get_number_of_parameters()
    
    stds = list([])
    
    for n in range(nparam):
        stds.append(np.std(posterior[:,n]))
        
    return np.array(stds)

def goodness_of_fit(posterior : np.ndarray, verbose : bool = True) -> np.ndarray :
    """
    @returns: a numpy array of the best fit values of each parameter, along with the highiest likelihood
              among the posterior samples.
    """
    loglikes = posterior[:,-1]
    best_index = np.argmax(loglikes)
    best_fit = posterior[best_index, :-1]
    chi2_best = -2 * loglikes[best_index]
    
    best_fit_array = np.append(best_fit, chi2_best)
    
    if (verbose == True):
        print(f"[ESG_ANALYSIS]: Best fit array = {best_fit_array}.")
    
    return best_fit_array

# ==================
# Data Visualisation
# ==================

# ---- Contour Plot ---- #
def contour_plot(posterior : np.ndarray) -> bool :
    if (posterior.shape[1] != 3):
        raise IndexError("[ESG_ANALYSIS]: Contour plot failed to generate -- posterior vector shape mismatch.\n\nContour plots can only be generated for 2D parameter spaces.")
    
    # --- Get data from model ---- #
    model_data = model.getData()
    parameter_names = model_data.param_names
    if (len(parameter_names) != 2):
        raise IndexError("[ESG_ANALYSIS]: Contour plot failed to generate -- parameter list must be two-dimensional!")
    model_name = model_data.model_name
    
    param1, param2, logL = posterior[:,0], posterior[:,1], posterior[:,2]
    
    best_fit = goodness_of_fit(posterior, False)
    
    mean1, mean2 = posterior_means(posterior)

    
    
    x_lims = (min(param1) - mean1/2, max(param1) + mean1/2)
    y_lims = (min(param2) - mean2/2, max(param2) + mean2/2)
    
    print(mean1, mean2)
    
    
    # ---- Begin Plot ---- #
    plt.figure(figsize=(10,12))
   
    sc = plt.scatter(param1, param2, c=logL, cmap='viridis', s=12, alpha=0.8, edgecolors='none')
    
    plt.scatter(mean1, mean2, c='black', s=20, alpha=0.5)
    plt.scatter(best_fit[0], best_fit[1], c='red', s=25, alpha=0.8)
    
    plt.plot([x_lims[0], x_lims[1]], [mean2, mean2], color='black', alpha=0.5, linestyle='--')
    plt.plot([mean1, mean1], [y_lims[0], y_lims[1]], color='black', alpha=0.5, linestyle='--')
    
    plt.plot([x_lims[0], x_lims[1]], [best_fit[1], best_fit[1]], color='red', alpha=0.8)
    plt.plot([best_fit[0], best_fit[0]], [y_lims[0], y_lims[1]], color='red', alpha=0.8)
    
    
    
    plt.xlabel(parameter_names[0])
    plt.ylabel(parameter_names[1])
    
    plt.xlim(x_lims[0], x_lims[1])
    plt.ylim(y_lims[0], y_lims[1])
        
    plt.title(f"Posterior samples contour plot for {model_name} model of the universe.")
    plt.colorbar(sc, label="log-likelihood")
    
    # === Annotate coordinates on axes ===
    # Format to 3 significant figures
    fmt = lambda x: f"{x:.3f}"
    
    # Text offsets (in data units or relative fraction)
    x_offset = 0.02 * (x_lims[1] - x_lims[0])
    y_offset = 0.02 * (y_lims[1] - y_lims[0])
    
    # Label mean position
    plt.text(
        mean1 + x_offset, y_lims[0] + 0.02*(y_lims[1]-y_lims[0]),
        f"mean = ({fmt(mean1)}, {fmt(mean2)})",
        color="black", fontsize=11, weight="bold",
        ha="left", va="bottom"
    )
    
    # Label best-fit position
    plt.text(
        best_fit[0] + x_offset, y_lims[0] + 0.08*(y_lims[1]-y_lims[0]),
        f"best = ({fmt(best_fit[0])}, {fmt(best_fit[1])})",
        color="red", fontsize=11, weight="bold",
        ha="left", va="bottom"
    )
        
    plt.tight_layout()
    
    # Save Plot
    plt.savefig(saveDir(_ANAL_RESULT_DIR) + "Contour_plot.png")
    plt.close()
    
    print(f"[ESG_ANALYSIS]: Contour plot generated. {len(posterior)} points plotted.")
    
    return 0

def contour_plot_3D(posterior: np.ndarray) -> bool:
    if posterior.shape[1] != 3:
        raise IndexError("[ESG_ANALYSIS]: 3D contour plot failed to generate — posterior vector shape mismatch.\n\nPlots can only be generated for 2D parameter spaces + 1D log-likelihood.")

    # --- Get data from model ---- #
    model_data = model.getData()
    parameter_names = model_data.param_names
    if len(parameter_names) != 2:
        raise IndexError("[ESG_ANALYSIS]: 3D contour plot failed to generate — parameter list must be two-dimensional!")
    model_name = model_data.model_name

    # ---- Begin 3D Plot ---- #
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    param1, param2, logL = posterior[:, 0], posterior[:, 1], posterior[:, 2]

    surf = ax.plot_trisurf(
        param1, param2, logL,
        cmap='viridis',
        linewidth=0.2,
        antialiased=True,
        alpha=0.9
    )

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="log-likelihood")

    # Labels and title
    ax.set_xlabel(parameter_names[0])
    ax.set_ylabel(parameter_names[1])
    ax.set_zlabel("log-likelihood", labelpad=10)
    ax.set_title(f"3D posterior surface for {model_name} model of the universe")

    ax.view_init(elev=35, azim=230)

    plt.tight_layout()
    
    # Save Plot
    plt.savefig(saveDir(_ANAL_RESULT_DIR) + "Contour_plot_3D.png", dpi=300)
    plt.close(fig)
    
    print(f"[ESG_ANALYSIS]: 3D contour plot generated. {len(posterior)} points plotted.")

    return 0
    
    
def corner_plot(posterior: np.ndarray) -> bool :
    model_data = model.getData()
    parameter_names = model_data.param_names
    model_name = model_data.model_name
    
    best_fit = goodness_of_fit(posterior, False)
    
    fig = corner.corner(
        posterior[:,:-1],
        labels=parameter_names,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        quantiles=[0.16, 0.5, 0.84],
        truths=[best_fit[0], best_fit[1]],
        truth_color='red',
        plot_datapoints=True,
        fill_contours=True,
        levels=[0.68, 0.95],
        color='blue',
        alpha=0.5,
        smooth=1.0,
        bins=30
    )

    # --- Annotate best-fit values on the diagonal panels (minimal change) ---
    try:
        D = posterior.shape[1] - 1  # number of parameters you plotted
        axes = np.array(fig.axes).reshape(D, D)
        k = min(D, len(best_fit))
        for i in range(k):
            ax = axes[i, i]
            ax.text(
                0.02, 0.92, f"best = {best_fit[i]:.4g}",
                transform=ax.transAxes,
                color='red',
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, lw=0)
            )
    except Exception as _:
        # Don’t kill the plot if corner's axes layout changes; just skip annotation.
        pass

    fig.suptitle(f"Flat {model_name} constraints from 32 CC $H(z)$ data", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save Plot
    plt.savefig(saveDir(_ANAL_RESULT_DIR) + f"corner_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[ESG_ANALYSIS]: corner plot generated. {len(posterior)} points plotted.")


def one_d_marginals(posterior: np.ndarray) -> bool:
    """
    Uses median and 68% credible interval (16th–84th percentiles).
    """
    # Pull names & sanity checks from model 
    model_data = model.getData()
    parameter_names = getattr(model_data, "param_names", None)
    model_name = getattr(model_data, "model_name", "Model")
    if parameter_names is None:
        raise AttributeError("[ESG_ANALYSIS]: parameter names not found on model_data.")
    n_params = len(parameter_names)

    if posterior.ndim != 2 or posterior.shape[1] < n_params:
        raise IndexError("[ESG_ANALYSIS]: 1D marginals failed — posterior array shape "
                         "must have at least the parameter columns (logL may be extra as last column).")

    # Only take parameter columns (drop logL if present)
    samples = posterior[:, :n_params]

    # Figure setup (1 row, n_params columns) 
    fig, axes = plt.subplots(1, n_params, figsize=(12, 4))
    if n_params == 1:
        axes = np.array([axes])

    # Loop over parameters
    for i, label in enumerate(parameter_names):
        ax = axes[i]
        xi = samples[:, i]

        # Histogram
        ax.hist(xi, bins=30, density=True, alpha=0.6,
                color='skyblue', edgecolor='k')

        # KDE (optional)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(xi)
            x_vals = np.linspace(np.min(xi), np.max(xi), 500)
            ax.plot(x_vals, kde(x_vals), color='blue', lw=2)
        except ImportError:
            pass

        # Median and 68% credible interval
        median = np.median(xi)
        lo, hi = np.percentile(xi, [16, 84])

        ax.axvline(median, color='red', linestyle='--', label='Median')
        ymin, ymax = ax.get_ylim()
        ax.fill_betweenx([ymin, ymax], lo, hi, color='red', alpha=0.2, label='68% CI')
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel(label, fontsize=14)
        ax.set_ylabel("Probability density", fontsize=12)
        ax.legend(fontsize=10)

    fig.suptitle(f"1D marginalized distributions for Flat {model_name}", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save Plot
    plt.savefig(saveDir(_ANAL_RESULT_DIR) + f"1D_distributions_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[ESG_ANALYSIS]: 1D marginals generated for {n_params} parameters "
          f"({len(posterior)} samples). Saved.")

    return 0


"""
Things to be done:
    > Need to add label support instead of name support
    > Better memory management.
    > UnGPT the last function
"""