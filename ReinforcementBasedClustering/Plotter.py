from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation


def PlotClustering(
    X, Y_final, pi_star,
    figsize=(6, 4), cmap="tab10",
    point_size=30, centroid_size=300,
    save_path=None
):
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

    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, M))
    data_colors = np.clip(pi_star @ colors, 0, 1)

    plt.scatter(X[:,0], X[:,1], c=data_colors,
                s=point_size, edgecolors="black",
                linewidths=0.5, alpha=0.7)

    for j in range(M):
        plt.scatter(Y_final[j,0], Y_final[j,1],
                    color=colors[j], marker="*",
                    s=centroid_size, edgecolors="black", linewidths=1.5)

    plt.gca().set_aspect("equal", "box")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.show()