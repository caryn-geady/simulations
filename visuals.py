import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
np.random.seed(3)

def generate_synthetic_data(N=90, seed=3):
    """
    Generate a synthetic dataset for RECIST groups, patient-level volume changes,
    and per-patient lesion-level changes.

    Parameters:
        N (int): Number of patients.
        seed (int): Random seed for reproducibility.

    Returns:
        labels (np.ndarray): RECIST group labels for each patient.
        vol_change (np.ndarray): Patient-level % volume change.
        lesion_changes (list of np.ndarray): Lesion-level changes per patient.
    """
    np.random.seed(seed)
    # assign RECIST groups
    labels = np.random.choice(["PR", "SD", "PD"], size=N, p=[0.35, 0.40, 0.25])

    # patient-level % volume change distribution by group
    vol_change = (
        np.where(labels == "PR", np.random.normal(-230, 60, N),
        np.where(labels == "SD", np.random.normal(  10, 80, N),
                                   np.random.normal( 320, 120, N)))
    )

    # per-patient lesion-level changes: list of arrays (each patient has 2–8 lesions)
    lesion_changes = []
    for i in range(N):
        k = np.random.randint(2, 9)
        # lesions vary around the patient-level change with added heterogeneity
        lc = np.random.normal(vol_change[i], 100, k)
        lesion_changes.append(lc)

    return labels, vol_change, lesion_changes

# Generate synthetic data
labels, vol_change, lesion_changes = generate_synthetic_data()
    

def custom_waterfall(labels, lesion_changes, pd_thresh=73):
    """
    Plots a custom waterfall chart visualizing total and individual lesion volume changes, colored by RECIST status.

    Parameters
    ----------
    labels : array-like of str
        List or array of RECIST status labels for each subject (e.g., "PR", "SD", "PD").
    lesion_changes : list of np.ndarray
        List where each element is an array of individual lesion volume changes (%) for a subject.
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
    - Bars represent total volume change per subject, colored by RECIST status.
    - Dots overlay individual lesion responses.
    - A dashed line indicates the PD threshold.
    - Includes a custom legend for RECIST groups, threshold, and lesion responses.
    """
    
    # Use total volume change for bar heights
    total_vol_change = np.array([lc.sum() for lc in lesion_changes])
    order = np.argsort(total_vol_change)  # ascending
    v_sorted   = total_vol_change[order]
    l_sorted   = labels[order]
    lesions_sorted = [lesion_changes[i] for i in order]


    cmap = {"PR": "green", "SD": "yellow", "PD": "red"}

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(v_sorted))

    # bars colored by RECIST status
    bar_colors = [cmap[g] for g in l_sorted]
    ax.bar(x, v_sorted, width=0.85, color=bar_colors, edgecolor="none", zorder=2)

    # overlay individual-lesion dots
    for i, lc in enumerate(lesions_sorted):
        ax.scatter(np.full(lc.shape, i), lc, s=16, color="k", alpha=0.65, zorder=3)

    # PD threshold line
    ax.axhline(pd_thresh, color="k", linestyle="--", linewidth=1)

    # pretty plot stuff
    ax.set_ylabel("Volume Change (%)")
    ax.set_xlim(-0.5, len(v_sorted) - 0.5)
    ax.set_ylim(min(-100, v_sorted.min()-50), max(580, v_sorted.max()+30))
    ax.tick_params(axis='x', labelbottom=False)
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    # legend
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

def custom_boxplot(vol_change, labels):
    """
    Creates a customized boxplot for volume change data grouped by response categories.

    Parameters
    ----------
    vol_change : array-like
        Array or sequence containing volume change values.
    labels : array-like
        Array or sequence of group labels corresponding to each value in `vol_change`.
        Expected values are "PR", "SD", and "PD".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object with the boxplot.

    Notes
    -----
    - The boxplot displays three groups: "PR" (green), "SD" (yellow), and "PD" (red).
    - The y-axis is labeled "Volume Change (%)" and limited to the range [-300, 600].
    - The top and right plot spines are removed for a cleaner appearance.
    """
    groups = ["PR", "SD", "PD"]
    data = [vol_change[labels == g] for g in groups]
    colors = ["green", "yellow", "red"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, labels=groups)

    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("black")
    for element in ['whiskers','caps','medians']:
        for line in bp[element]:
            line.set_color("black")

    # pretty plot stuff
    ax.set_ylabel("Volume Change (%)")
    ax.set_ylim(-300, 600)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig, ax



# ---- Generate pretty plots ----
fig1, ax1 = custom_waterfall(labels, lesion_changes, pd_thresh=20)
fig2, ax2 = custom_boxplot(vol_change, labels)
plt.show()
