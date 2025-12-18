import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.colors as mcolors
import numpy as np


def plot_random_search_metrics_twinx(df: pd.DataFrame, x_axes=["learning_rate", "learning_rate_energy", "lamb_ortho"], metrics=["Loss_DE", "Loss_Orthogonality"], s=5, alpha=0.3, xscale="log", yscale="log"):
    ''' Function for plotting the result of chosen metrics as a function of the parameters along the x axis from a random search.'''

    fig, axes = plt.subplots(1, len(x_axes), figsize=(6*len(x_axes), 5))
    if not hasattr(axes, "__getitem__"):
        axes = [axes]

    legend_handles = []

    for i, x_axis in enumerate(x_axes):
        base_ax = axes[i]
        base_ax.set_xscale(xscale)
        for j, metric in enumerate(metrics):
            # Legend handles
            if i == 0:
                legend_handles.append(Line2D([0], [0], marker='o', color='w', label=metric,markerfacecolor=f"C{j}", markersize=8))

            # Create twin axes
            ax = base_ax if j == 0 else base_ax.twinx()
            
            ax.set_xlabel(x_axis)

            ax.scatter(df[x_axis], df[metric], alpha=alpha, s=s, color=f"C{j}")
            ax.set_yscale(yscale)

            # --- Color ticks *and* tick marks *and* axis spine ---
            ax.tick_params(axis='y', colors=f"C{j}", which="both")  # colors tick labels + marks (if visible)

            # Ensure tick MARK LINES are colored (sometimes the above fails)
            for tickline in ax.yaxis.get_ticklines():
                tickline.set_color(f"C{j}")

            # Color the visible y-axis spine
            spine = 'left' if j == 0 else 'right'
            ax.spines[spine].set_color(f"C{j}")

        
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(metrics),
        frameon=True,
        bbox_to_anchor=(0.5, 1.0)  # adjust vertical spacing
    )

    plt.tight_layout(rect=(0, 0.0, 1, 1))  # reserve extra margin below


def plot_colormap_plot(df: pd.DataFrame, metrics=["Loss_DE", "Loss_Orthogonality"], x="learning_rate", y="learning_rate_energy", s=20, alpha=0.5, color_scale="log", vmax=None):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7, 5*len(metrics)))
    if not hasattr(axes, "__getitem__"):
        axes = [axes]
    for i, metric in enumerate(metrics):
            
        ax = axes[i]
        vmin = df[metric].min()
        if vmin <= 0:
            vmin = df[df[metric]!=0][metric].min() # LogNorm requires positive values
            
        log_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax if vmax else df[metric].max())
        scatter = ax.scatter(df[x], df[y], alpha=alpha, s=s, c=df[metric], cmap="viridis", norm=log_norm if color_scale=="log" else None)
        fig.colorbar(scatter, ax=ax, label=f"{metric}")
        ax.set_xscale("log")
        ax.set_yscale("log")

        if i == len(metrics)-1:
            ax.set_xlabel(x)
            ax.set_ylabel(y)
    plt.tight_layout()

def plot_3dscatter(df, params, metric, s=20, alpha=0.5):
    x, y, z = params
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        np.log10(df[x]), np.log10(df[y]), np.log10(df[z]),
        c=np.log10(df[metric]), cmap="viridis",
        s=s, alpha=alpha
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label(metric)
    return fig, ax