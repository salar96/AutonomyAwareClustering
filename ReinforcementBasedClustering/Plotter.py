from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation
import distinctipy

import random


def PlotClustering(
    X, Y_final, pi_star,
    figsize=(6, 4), cmap="tab10",
    point_size=30, centroid_size=300,
    alpha=0.7,
    data_edge_color="black",
    cluster_edge_color="black",
    save_path=None
):  
    SEED = 123 
    random.seed(SEED)
    rng = random.Random(SEED)
    # --- sort by angle around (0.5,0.5) ---
    ref = np.array([0.5, 0.5])
    angles = np.arctan2(Y_final[:,1] - ref[1], Y_final[:,0] - ref[0])
    angles = (angles + 2*np.pi) % (2*np.pi)  # 0..2π
    order = np.argsort(angles)
    Y_final = Y_final[order]
    pi_star = pi_star[:, order]
    # ---------------------------------------

    N, M = X.shape[0], Y_final.shape[0]
    plt.figure(figsize=figsize, facecolor="white")
    if cmap is not None:
        cmap_ = plt.cm.get_cmap(cmap, M)
        colors = [cmap_(i) for i in range(M)]
    else:
        colors = distinctipy.get_colors(M)

    data_colors = np.clip(pi_star @ colors, 0, 1)

    plt.scatter(X[:,0], X[:,1], c=data_colors,
                s=point_size, edgecolors=data_edge_color,
                linewidths=0.5, alpha=alpha)

    for j in range(M):
        plt.scatter(Y_final[j,0], Y_final[j,1],
                    color=colors[j], marker="*",
                    s=centroid_size, edgecolors=cluster_edge_color, linewidths=1.5)

    plt.gca().set_aspect("equal", "box")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # --- test ---
    N = 300
    M = 5
    X = np.random.rand(N, 2)
    Y_final = np.random.rand(M, 2)
    pi_star = np.random.dirichlet(alpha=np.ones(M), size=N)

    PlotClustering(X, Y_final, pi_star, figsize=(6, 6), cmap=None, point_size=20, centroid_size=200)