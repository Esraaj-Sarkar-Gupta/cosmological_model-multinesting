"""
ESG_analyzer.py

Created Nov 2025
@author: esraaj

Non-native pyMultiNest Analyzer.
Updated Jan 2026: Added support for N-dimensional parameter spaces.

Functions available:
    ** Statistical Values **
    > posterior_mean @return posterior means vector
    > posterior_std_deviation (Posterior standard deviation) @return posterior standard deviations vector
    > goodness_of_fit @returns best fit vector
    
    ** Data Visualisation **
    > contour_plot (Generates 2D slice of the first two parameters)
    > corner_plot -- Generate corner plots (Dynamic for N parameters)
    > one_d_marginals -- Marginal plots for each parameter
    > comparison_plot -- Overlays multiple posteriors on one H0-Omegam graph
    
Markup label support for figures has not been added yet.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import corner
import os
import matplotlib.lines as mlines
from getdist import plots, MCSamples

# --- Model Name --- #
from Models import LCDM as model

# ---- Helper for Dataset Name (Avoids Circular Import) ---- #
def get_dataset_name_safe():
    try:
        from main import getDatasetName
        return getDatasetName()
    except ImportError:
        return "Unknown_Data"

# ---- Results Directory ----

_ANAL_RESULT_DIR = f"./{get_dataset_name_safe()}_ESG_analyzer/" 

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
    # Calculate params from data shape (cols - 1 for logL)
    nparam = posterior.shape[1] - 1
    
    means = list([])
    
    for n in range(nparam):
        means.append(np.mean(posterior[:,n]))
        
    return np.array(means)

def posterior_std_deviations(posterior : np.ndarray) -> np.ndarray :
    """
    @returns: a numpy array of the posterior standard deviations of each parameter sampled.
    """
    nparam = posterior.shape[1] - 1
    
    stds = list([])
    
    for n in range(nparam):
        stds.append(np.std(posterior[:,n]))
        
    return np.array(stds)

def goodness_of_fit(posterior : np.ndarray, verbose : bool = True) -> np.ndarray :
    """
    @returns: a numpy array of the best fit values of each parameter, along with the highiest likelihood
              among the posterior samples.
    """
    loglikes = posterior[:,-1] # Last column is logL
    best_index = np.argmax(loglikes)
    
    best_fit = posterior[best_index, :-1] # All params excluding logL
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
    """
    Generates a 2D contour plot of the first two parameters.
    """
    # --- Get data from model ---- #
    model_data = model.getData()
    parameter_names = model_data.param_names
    model_name = model_data.model_name
    
    # We only take the first two parameters for the 2D slice
    param1, param2 = posterior[:,0], posterior[:,1]
    logL = posterior[:,-1]
    
    best_fit_full = goodness_of_fit(posterior, False)
    best_fit = best_fit_full[0:2] # Slice first 2
    
    mean_full = posterior_means(posterior)
    mean1, mean2 = mean_full[0], mean_full[1]
    
    x_lims = (min(param1) - mean1/4, max(param1) + mean1/4) # Adjusted scaling slightly
    y_lims = (min(param2) - mean2/4, max(param2) + mean2/4)
    
    print(f"[ESG_ANALYSIS]: Contour stats for first 2 params: {mean1:.3f}, {mean2:.3f}")
    
    # ---- Begin Plot ---- #
    plt.figure(figsize=(10,12))
   
    sc = plt.scatter(param1, param2, c=logL, cmap='viridis', s=12, alpha=0.8, edgecolors='none')
    
    plt.scatter(mean1, mean2, c='black', s=20, alpha=0.5, label="Mean")
    plt.scatter(best_fit[0], best_fit[1], c='red', s=50, alpha=0.8, marker='*', label="Best Fit")
    
    # Crosshairs
    plt.plot([x_lims[0], x_lims[1]], [mean2, mean2], color='black', alpha=0.3, linestyle='--')
    plt.plot([mean1, mean1], [y_lims[0], y_lims[1]], color='black', alpha=0.3, linestyle='--')
    
    plt.plot([x_lims[0], x_lims[1]], [best_fit[1], best_fit[1]], color='red', alpha=0.5)
    plt.plot([best_fit[0], best_fit[0]], [y_lims[0], y_lims[1]], color='red', alpha=0.5)
    
    plt.xlabel(parameter_names[0])
    plt.ylabel(parameter_names[1])
    
    plt.xlim(x_lims[0], x_lims[1])
    plt.ylim(y_lims[0], y_lims[1])
        
    plt.title(f"Posterior samples (2D Slice) for {model_name} from {get_dataset_name_safe()} data.")
    plt.colorbar(sc, label="log-likelihood")
    plt.legend()
    
    # === Annotate coordinates on axes ===
    fmt = lambda x: f"{x:.3f}"
    
    # Text offsets
    x_offset = 0.02 * (x_lims[1] - x_lims[0])
    
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
    plt.savefig(saveDir(_ANAL_RESULT_DIR) + "Contour_plot_2D.png")
    plt.close()
    
    print(f"[ESG_ANALYSIS]: Contour plot generated. {len(posterior)} points plotted.")
    
    return False


def corner_plot(posterior: np.ndarray) -> bool :
    model_data = model.getData()
    # Get ALL potential names from model
    all_param_names = model_data.param_names
    model_name = model_data.model_name
    
    # ---- DYNAMIC DETECTION ----
    # Determine N parameters from the DATA, ignoring extra config names
    # Posterior has (N_params + LogL) columns usually
    n_params_data = posterior.shape[1] - 1
    
    # Slice names to match data dimensions
    current_param_names = all_param_names[0:n_params_data]
    
    # Slice only the parameter columns
    samples = posterior[:, 0:n_params_data]
    
    # Get truths (best fit)
    best_fit_full = goodness_of_fit(posterior, False)
    truths = best_fit_full[0:n_params_data]
    
    print(f"[ESG_ANALYSIS]: Detecting {n_params_data} active parameters for Corner Plot.")
    
    # Plot using corner
    fig = corner.corner(
        samples,
        labels=current_param_names, # Use sliced names
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        quantiles=[0.16, 0.5, 0.84],
        truths=truths,
        truth_color='red',
        plot_datapoints=False, # Cleaner for high N
        fill_contours=True,
        levels=[0.68, 0.95],
        color='blue',
        alpha=0.5,
        smooth=1.0,
        bins=30
    )

    # --- Annotate best-fit values on the diagonal panels ---
    try:
        D = n_params_data
        axes = np.array(fig.axes).reshape(D, D)
        for i in range(D):
            ax = axes[i, i]
            ax.text(
                0.02, 0.92, f"best = {truths[i]:.4g}",
                transform=ax.transAxes,
                color='red',
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, lw=0)
            )
    except Exception as _:
        pass

    fig.suptitle(f"{model_name} constraints from {get_dataset_name_safe()} data.", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save Plot
    plt.savefig(saveDir(_ANAL_RESULT_DIR) + f"corner_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[ESG_ANALYSIS]: Corner plot generated for {n_params_data} parameters.")
    return False


def one_d_marginals(posterior: np.ndarray) -> bool:
    """
    Uses median and 68% credible interval (16th–84th percentiles).
    Dynamically adjusts subplots for N parameters.
    """
    # Pull names & sanity checks from model 
    model_data = model.getData()
    all_param_names = getattr(model_data, "param_names", None)
    model_name = getattr(model_data, "model_name", "Model")
    
    if all_param_names is None:
        raise AttributeError("[ESG_ANALYSIS]: parameter names not found on model_data.")
    
    # ---- DYNAMIC DETECTION ----
    n_params_data = posterior.shape[1] - 1
    current_param_names = all_param_names[0:n_params_data]

    # Only take parameter columns
    samples = posterior[:, :n_params_data]

    # Figure setup (1 row, n_params columns) 
    fig, axes = plt.subplots(1, n_params_data, figsize=(4*n_params_data, 4))
    
    # Ensure axes is iterable if n_params == 1
    if n_params_data == 1:
        axes = np.array([axes])

    # Loop over parameters
    for i, label in enumerate(current_param_names):
        ax = axes[i]
        xi = samples[:, i]

        # Histogram
        ax.hist(xi, bins=30, density=True, alpha=0.6,
                color='skyblue', edgecolor='k')

        # KDE
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
        
        # Add a title with the constraints
        ax.set_title(f"{median:.3f} +{hi-median:.3f} / -{median-lo:.3f}")
        
        if (i == 0):
             ax.legend(fontsize=10)

    fig.suptitle(f"1D marginalized distributions for Flat {model_name}", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save Plot
    plt.savefig(saveDir(_ANAL_RESULT_DIR) + f"1D_distributions_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[ESG_ANALYSIS]: 1D marginals generated for {n_params_data} parameters from {get_dataset_name_safe()} data.")

    return False


# ---- Comparison Plot (Multiple Datasets) ---- #

def comparison_plot_hist2d(posteriors_dict : dict) -> bool:
    """
    Overlays H0-Omega_m contours for multiple datasets.
    posteriors_dict format: {"Label": posterior_array, ...}
    """
    model_data = model.getData()
    labels = model_data.param_names
    
    print(f"\\n[ESG_ANALYSIS]: Generating Comparison Plot for {len(posteriors_dict)} datasets...")
    
    # Define colors for different runs
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    
    # Setup Figure (Just 2D H0-Om)
    fig, ax = plt.subplots(figsize=(9, 9))
    
    legend_handles = []
    
    for i, (name, samples) in enumerate(posteriors_dict.items()):
        color = colors[i % len(colors)]
        
        # Slice only H0 and Omega_m
        data_2d = samples[:, 0:2]
        
        corner.hist2d(
            data_2d[:,0], 
            data_2d[:,1], 
            ax=ax,
            color=color,
            plot_datapoints=False,
            plot_density=False,
            levels=[0.10, 0.68, 0.95], # 1 and 2 sigma
            smooth=1.0,
            bins=100,
            alpha=0.20,
            fill_contours=True
        )
        
        # Create a custom legend handle
        line = mlines.Line2D([], [], color=color, label=name)
        legend_handles.append(line)
        
        # Print stats to console
        h0_mean = np.mean(data_2d[:,0])
        om_mean = np.mean(data_2d[:,1])
        print(f"   > {name}: H0={h0_mean:.2f}, Om={om_mean:.3f}")

    ax.set_xlabel(labels[0], fontsize=14)
    ax.set_ylabel(labels[1], fontsize=14)
    ax.legend(handles=legend_handles, fontsize=12, loc='upper right')
    ax.set_title("Planck-Constrained Parameter Comparison", fontsize=16)
    
    # Save
    out_dir = saveDir("./Combined_Analysis/")
    plt.savefig(out_dir + "Comparison_H0_Om_Contours.png", dpi=150)
    plt.close()
    
    print("[ESG_ANALYSIS]: Comparison plot saved to ./Combined_Analysis/")
    return False

def comparison_plot(posteriors_dict: dict) -> bool:
    """
    Overlays H0-Omega_m contours using GetDist.
    Corrected to fix 'add_legend' TypeError.
    """
    print(f"\n[ESG_ANALYSIS]: Generating Comparison Plot (GetDist) for {len(posteriors_dict)} datasets...")

    # Convert numpy chains into GetDist MCSamples objects
    samples_list = []
    
    # Define colors explicitly
    colors = ['red', 'blue', 'green', 'purple'] 
    
    # Store labels to ensure they are passed correctly
    legend_labels = []

    for name, samples in posteriors_dict.items():
        # MCSamples requires the raw chain data
        # names/labels sets the axis internal names and LaTeX labels
        mc_sample = MCSamples(samples=samples[:, 0:2], 
                              names=['H0', 'Omega_m'], 
                              labels=['H_0', '\\Omega_m'], 
                              label=name)
        
        mc_sample.updateSettings({
            'smooth_scale_2D': 0.7,
            'contours': [0.10, 0.68, 0.95]
            }) 
        
        samples_list.append(mc_sample)
        legend_labels.append(name)

    # Create the plotter instance
    g = plots.get_single_plotter(width_inch=8)
    
    # Plot 2D contours
    # legend_loc='upper right' is passed HERE, not in a separate call
    g.plot_2d(samples_list, 'H0', 'Omega_m', 
              filled=True, 
              colors=colors, 
              alpha=0.3,
              legend_loc='upper right')

    # 4. Customizations
    # We add the title manually using standard matplotlib on the subplot
    g.subplots[0,0].set_title('Planck-Constrained Parameter Comparison', fontsize=16)
    
    # 5. Save
    out_dir = saveDir("./Combined_Analysis/")
    g.export(out_dir + "Comparison_H0_Om_Contours_GetDist.png")
    
    print("[ESG_ANALYSIS]: GetDist Comparison plot saved.")
    return True