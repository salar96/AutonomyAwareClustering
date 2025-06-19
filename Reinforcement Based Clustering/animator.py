import matplotlib.animation as animation
import numpy as np
from matplotlib import pyplot as plt

def animate_Clusters(
    nodes,
    Facilities_over_time,
    assignment_list,
    figuresize=(6, 5),
    interval=200,
    save_path="facility_movement.gif",
):
    facecolor = "#CACACA"
    edgecolor = "#08D3D6"

    # Create a categorical colormap
    num_clusters = max(np.max(a) for a in assignment_list) + 1
    cmap = plt.cm.get_cmap("Accent", num_clusters)

    # Set up the figure
    fig = plt.figure(figsize=figuresize, facecolor=facecolor, edgecolor=edgecolor)
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    # Initialize scatter with correct colors
    initial_colors = cmap(assignment_list[0])
    node_scatter = ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        c=initial_colors,
        marker="o",
        alpha=0.6,
        edgecolors='black',
        label="S",
    )

    f_scatter = ax.scatter([], [], color="#d805ef", marker="x", label="F")
    ax.axis("off")

    # Set axis limits
    all_x = np.concatenate(
        [nodes[:, 0], nodes[:, 0]] + [f[:, 0] for f in Facilities_over_time]
    )
    all_y = np.concatenate(
        [nodes[:, 1], nodes[:, 1]] + [f[:, 1] for f in Facilities_over_time]
    )
    ax.set_xlim(min(all_x) - 0.2, max(all_x) + 0.2)
    ax.set_ylim(min(all_y) - 0.2, max(all_y) + 0.2)

    # Animation update function
    def update(frame):
        cluster_ids = assignment_list[frame]
        node_colors = cmap(cluster_ids)
        node_scatter.set_facecolor(node_colors)
        f_scatter.set_offsets(Facilities_over_time[frame])
        return (node_scatter, f_scatter)

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(Facilities_over_time), interval=interval, blit=True
    )

    # Save the animation
    anim.save(save_path, writer="pillow")
    plt.close()

    return anim