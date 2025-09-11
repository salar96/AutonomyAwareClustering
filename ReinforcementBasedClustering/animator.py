import matplotlib.animation as animation
import numpy as np
from matplotlib import pyplot as plt

def animate_Clusters(
    nodes,
    Facilities_over_time,
    assignment_probabilities,  # Now takes probabilities instead of hard assignments
    figuresize=(6, 5),
    interval=200,
    save_path="facility_movement.gif",
):
    facecolor = "#CACACA"
    edgecolor = "#08D3D6"

    # Get number of clusters (facilities)
    num_clusters = Facilities_over_time[0].shape[0]

    # Create a categorical colormap using tab10 for distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))

    # Set up the figure
    fig = plt.figure(figsize=figuresize, facecolor=facecolor, edgecolor=edgecolor)
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    # Calculate initial data colors based on weighted probabilities
    initial_data_colors = np.zeros((len(nodes), 4))  # RGBA colors
    for i in range(len(nodes)):
        for j in range(num_clusters):
            initial_data_colors[i] += assignment_probabilities[0][i, j] * colors[j]
    # normalize colors to ensure they are valid RGBA
    initial_data_colors = np.clip(initial_data_colors, 0, 1)
    # Initialize data scatter plot
    node_scatter = ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        c=initial_data_colors,
        marker="o",
        s=30,
        edgecolors='black',
        linewidths=0.5,
        alpha=0.7,
        label="Data points",
    )

    # Initialize facility markers (will be updated in each frame)
    f_scatters = []
    for j in range(num_clusters):
        f_scatter = ax.scatter(
            [], [], 
            color=colors[j],
            marker="*",
            s=300,
            edgecolors='black',
            linewidths=1.5,
            label=f"Centroid {j+1}" if j == 0 else "",
        )
        f_scatters.append(f_scatter)

    ax.axis("off")

    # Set axis limits based on all data
    all_x = np.concatenate(
        [nodes[:, 0]] + [f[:, 0] for f in Facilities_over_time]
    )
    all_y = np.concatenate(
        [nodes[:, 1]] + [f[:, 1] for f in Facilities_over_time]
    )
    margin = 0.1 * (max(all_x) - min(all_x))
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    

    # Animation update function
    def update(frame):
        # Update data colors based on assignment probabilities
        data_colors = np.zeros((len(nodes), 4))  # RGBA colors
        for i in range(len(nodes)):
            for j in range(num_clusters):
                data_colors[i] += assignment_probabilities[frame][i, j] * colors[j]
        # normalize colors to ensure they are valid RGBA
        data_colors = np.clip(data_colors, 0, 1)
        
        # Set the color while preserving black edgecolor
        node_scatter.set_facecolor(data_colors)
        node_scatter.set_edgecolor('black')  # Explicitly maintain black edgecolor
        
        # Update facility positions
        for j in range(num_clusters):
            if j < len(Facilities_over_time[frame]):
                f_scatters[j].set_offsets([Facilities_over_time[frame][j]])
            else:
                f_scatters[j].set_offsets([])
        
        return [node_scatter] + f_scatters

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(Facilities_over_time), interval=interval, blit=True
    )

    # Save the animation
    anim.save(save_path, writer="pillow")
    plt.close()

    return anim