from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation


def PlotClustering(X, Y_final, pi_star, figsize=(6, 4), save_path=None):
    N = X.shape[0]
    M = Y_final.shape[0]
    plt.figure(figsize=figsize, facecolor="#FFFFFF", edgecolor="#000000")

    # Create a colormap with M distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, M))  # Use tab10 for distinct colors

    # For each data point, compute its color based on cluster assignment probabilities
    data_colors = np.zeros((N, 4))  # RGBA colors
    for i in range(N):
        for j in range(M):
            data_colors[i] += pi_star[i, j] * colors[j]
    data_colors = np.clip(data_colors, 0, 1, out=data_colors)
    # Plot data points with their weighted colors
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=data_colors,
        marker="o",
        s=30,  # Slightly larger points
        edgecolors="black",
        linewidths=0.5,  # Thinner edges
        alpha=0.7,
        label="Data points",
    )

    # Plot centroids with distinct markers and colors
    for j in range(M):
        plt.scatter(
            Y_final[j, 0],
            Y_final[j, 1],
            color=colors[j],
            marker="*",
            s=300,  # Larger centroids
            edgecolors="black",
            linewidths=1.5,
            label=f"Centroid {j+1}" if j == 0 else "",
        )

    # Add a legend for the first centroid only to avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    plt.gca().set_aspect("equal", "box")
    plt.axis("off")
    if save_path is not None:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.show()
