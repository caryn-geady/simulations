import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
np.random.seed(3)

def generate_synthetic_raw_volumes(N=90, seed=3):
    """
    Generates synthetic pre- and post-treatment lesion volumes for a cohort of patients.

    Parameters
    ----------
    N : int, optional
        Number of patients to generate data for. Default is 90.
    seed : int, optional
        Random seed for reproducibility. Default is 3.

    Returns
    -------
    labels : np.ndarray
        Array of RECIST group labels ("PR", "SD", "PD") for each patient.
    pre_vols : list of np.ndarray
        List of arrays containing pre-treatment lesion volumes for each patient.
    post_vols : list of np.ndarray
        List of arrays containing post-treatment lesion volumes for each patient.

    Notes
    -----
    - Each patient is assigned a response group label with probabilities: PR (35%), SD (40%), PD (25%).
    - Each patient has between 2 and 8 lesions with random pre-treatment volumes between 10 and 100 cc.
    - Post-treatment volumes are simulated using group-specific percent changes drawn from normal distributions.
    """
    np.random.seed(seed)
    labels = np.random.choice(["PR", "SD", "PD"], size=N, p=[0.35, 0.40, 0.25])

    pre_vols = []
    post_vols = []
    for i in range(N):
        k = np.random.randint(2, 9)
        # Pre volumes: random between 10 and 100 cc
        pre = np.random.uniform(10, 100, k)
        # Simulate group-specific percent change
        if labels[i] == "PR":
            pct_change = np.random.normal(-230, 60, k)
        elif labels[i] == "SD":
            pct_change = np.random.normal(10, 80, k)
        else:
            pct_change = np.random.normal(320, 120, k)
        post = pre * (1 + pct_change / 100)
        pre_vols.append(pre)
        post_vols.append(post)
    return labels, pre_vols, post_vols

def custom_waterfall_raw(labels, pre_vols, post_vols, pd_thresh=73):
    """
    Plots a custom waterfall chart visualizing raw lesion volume changes for a cohort of patients.

    Parameters
    ----------
    labels : array-like of str
        List or array of response category labels for each patient (e.g., "PR", "SD", "PD").
    pre_vols : list of array-like
        List where each element is an array of pre-treatment lesion volumes for a patient.
    post_vols : list of array-like
        List where each element is an array of post-treatment lesion volumes for a patient.
    pd_thresh : float, optional
        Threshold for progressive disease (PD) in percent volume change. Default is 73.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object of the plot.

    Notes
    -----
    - The function computes percent change in total lesion volume per patient and sorts patients accordingly.
    - Each bar represents the cumulative percent change in lesion volume for a patient, colored by response category.
    - Individual lesion responses are overlaid as scatter points.
    - A horizontal dashed line indicates the PD threshold.
    - The legend explains color coding and markers.
    """
    # Lesion-level percent changes
    lesion_pct_changes = [100 * (post - pre) / pre for pre, post in zip(pre_vols, post_vols)]
    # Patient-level cumulative percent change
    total_pre = np.array([pre.sum() for pre in pre_vols])
    total_post = np.array([post.sum() for post in post_vols])
    total_pct_change = 100 * (total_post - total_pre) / total_pre

    order = np.argsort(total_pct_change)
    v_sorted = total_pct_change[order]
    l_sorted = labels[order]
    lesions_sorted = [lesion_pct_changes[i] for i in order]

    cmap = {"PR": "green", "SD": "yellow", "PD": "red"}

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(v_sorted))
    bar_colors = [cmap[g] for g in l_sorted]
    ax.bar(x, v_sorted, width=0.85, color=bar_colors, edgecolor="none", zorder=2)

    for i, lc in enumerate(lesions_sorted):
        ax.scatter(np.full(lc.shape, i), lc, s=16, color="k", alpha=0.65, zorder=3)

    ax.axhline(pd_thresh, color="k", linestyle="--", linewidth=1)
    ax.set_ylabel("Volume Change (%)")
    ax.set_xlim(-0.5, len(v_sorted) - 0.5)
    ax.set_ylim(min(-100, v_sorted.min()-50), max(580, v_sorted.max()+30))
    ax.tick_params(axis='x', labelbottom=False)
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    handles = [
        Patch(facecolor=cmap["PR"], label="PR"),
        Patch(facecolor=cmap["SD"], label="SD"),
        Patch(facecolor=cmap["PD"], label="PD"),
        Line2D([0],[0], color="k", linestyle="--", label="RECIST PD Threshold (20%)"),
        Line2D([0],[0], marker='o', color='k', linestyle='None', markersize=5, label='Individual Lesion Response')
    ]
    legend_labels = [
        "PR",
        "SD",
        "PD",
        "RECIST PD Threshold (20%)",
        "Individual Lesion Response"
    ]
    ax.legend(handles, legend_labels, loc="upper left", frameon=True)
    plt.tight_layout()
    return fig, ax

def custom_boxplot_raw(labels, pre_vols, post_vols):
    """
    Plots a boxplot of cumulative percent volume change per patient, grouped by RECIST response category.

    Parameters
    ----------
    labels : array-like of str
        List or array of response category labels for each patient (e.g., "PR", "SD", "PD").
    pre_vols : list of array-like
        List where each element is an array of pre-treatment lesion volumes for a patient.
    post_vols : list of array-like
        List where each element is an array of post-treatment lesion volumes for a patient.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object of the plot.

    Notes
    -----
    - Computes percent change in total lesion volume per patient.
    - Groups patients by RECIST response category ("PR", "SD", "PD").
    - Displays boxplots for each group, colored accordingly.
    """
    total_pre = np.array([pre.sum() for pre in pre_vols])
    total_post = np.array([post.sum() for post in post_vols])
    total_pct_change = 100 * (total_post - total_pre) / total_pre
    groups = ["PR", "SD", "PD"]
    data = [total_pct_change[labels == g] for g in groups]
    colors = ["green", "yellow", "red"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, labels=groups)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("black")
    for element in ['whiskers','caps','medians']:
        for line in bp[element]:
            line.set_color("black")
    ax.set_ylabel("Volume Change (%)")
    ax.set_ylim(-300, 600)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig, ax

# ---- Generate pretty plots ----
labels, pre_vols, post_vols = generate_synthetic_raw_volumes()
fig1, ax1 = custom_waterfall_raw(labels, pre_vols, post_vols, pd_thresh=20)
fig2, ax2 = custom_boxplot_raw(labels, pre_vols, post_vols)
plt.show()
