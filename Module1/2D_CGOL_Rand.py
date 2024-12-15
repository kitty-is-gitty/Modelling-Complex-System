import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# Parameters
grid_size = 100  
frames = 800  # Number of frames for the animation
f_values = [0.6, 0.7, 0.9]  # Three different values of f (fraction of live cells)
colors = ['blue', 'green', 'red']  # Colors for different f values

# Initialize an empty list to store the mean liveness for each f value
mean_liveness_data = {f: [] for f in f_values}

def initialize_grid(f):
    """Initialize the grid with the specified fraction of live cells (f)."""
    num_live_cells = int(f * grid_size * grid_size)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    live_indices = np.random.choice(grid_size * grid_size, num_live_cells, replace=False)
    grid[np.unravel_index(live_indices, (grid_size, grid_size))] = 1
    return grid

def update_grid(grid):
    """Apply the rules of Conway's Game of Life to update the grid."""
    new_grid = grid.copy()

    # Calculate the number of live neighbors for each cell
    neighbors = sum(np.roll(np.roll(grid, i, 0), j, 1)
                    for i in (-1, 0, 1) for j in (-1, 0, 1)
                    if (i != 0 or j != 0))

    # Apply the rules of Conway's Game of Life
    new_grid[(grid == 1) & ((neighbors < 2) | (neighbors > 3))] = 0
    new_grid[(grid == 0) & (neighbors == 3)] = 1

    return new_grid

def animate(frame_num):
    """Update the grids and mean liveness values for all f values."""
    for i, f in enumerate(f_values):
        grid = grids[f]  # Get the grid for the current f value
        new_grid = update_grid(grid)  # Update the grid using the Game of Life rules
        grids[f][:] = new_grid  # Update the grid in place

        # Calculate mean liveness and store it
        mean_liveness = np.mean(new_grid)
        mean_liveness_data[f].append(mean_liveness)

        # Update the grid plot
        im_list[i].set_array(new_grid)

    # Update the live plot
    for i, f in enumerate(f_values):
        mean_lines[i].set_data(range(len(mean_liveness_data[f])), mean_liveness_data[f])

    return mean_lines + im_list

# Set up the figure and the axes using GridSpec
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 3, height_ratios=[2, 1])

# Create subplots for the three grids
ax_grids = [fig.add_subplot(gs[0, i]) for i in range(3)]

# Create the subplot for the mean liveness plot
ax_liveness = fig.add_subplot(gs[1, :])
ax_liveness.set_title('Mean Liveness vs Generation for Different f Values')
ax_liveness.set_xlabel('Generation')
ax_liveness.set_ylabel('Mean Liveness')

# Initialize grids and image objects for each f value
grids = {f: initialize_grid(f) for f in f_values}
# Switch the colormap to 'binary_r' to reverse the color of live (black) and dead (white) cells
im_list = [ax_grids[i].imshow(grids[f], cmap='binary_r', vmin=0, vmax=1, animated=True) for i, f in enumerate(f_values)]

# Set titles for the grid plots
for i, f in enumerate(f_values):
    ax_grids[i].set_title(f'f = {f}')
    ax_grids[i].axis('off')  # Turn off axes for the grids

# Initialize mean liveness lines for the live plot
mean_lines = [ax_liveness.plot([], [], color=colors[i], label=f'f = {f}')[0] for i, f in enumerate(f_values)]

# Initialize the plot limits
def init():
    ax_liveness.set_xlim(0, frames)
    ax_liveness.set_ylim(0, 1)
    return mean_lines + im_list

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=frames, interval=50, blit=True, init_func=init)

# Add a legend to the live plot
ax_liveness.legend()

plt.tight_layout()
plt.show()
