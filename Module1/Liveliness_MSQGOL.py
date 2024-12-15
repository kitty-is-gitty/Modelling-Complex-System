import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# Parameters
grid_size = 100
frames = 200  # Number of generations to animate
f_values = [0.1, 0.2, 0.35, 0.5, 0.8]  # Different fractions of live cells for the five simulations
mean_liveness_values = [[] for _ in f_values]  # To store mean liveness for each f
colors = ['blue', 'green', 'red', 'purple', 'orange']  # Colors for different f plots

# Initialize five 100x100 grids, one for each fraction f
grids = []
for f in f_values:
    grid = np.zeros((grid_size, grid_size))
    num_live_cells = int(f * grid_size * grid_size)  # Total number of live cells
    live_indices = np.random.choice(grid_size * grid_size, num_live_cells, replace=False)
    grid[np.unravel_index(live_indices, (grid_size, grid_size))] = np.random.rand(num_live_cells)
    grids.append(grid)

# Function to update the state of the grid
def update_grid(grid):
    new_grid = np.copy(grid)
    for i in range(grid_size):
        for j in range(grid_size):
            neighborhood = grid[max(0, i-1):min(i+2, grid_size), max(0, j-1):min(j+2, grid_size)]
            A = np.sum(neighborhood)  # Liveness sum of neighbors
            
            if A >= 0 and A <= 1:
                new_grid[i, j] = 0  # Dead
            elif A > 1 and A <= 2:
                new_grid[i, j] = np.sqrt(2 + 1) / 2  # Semi-live (gray)
            elif A > 2 and A <= 3:
                new_grid[i, j] = np.sqrt(2 + 1) / 3  # Semi-live (gray)
            elif A > 3 and A <= 4:
                new_grid[i, j] = 1  # Live
            else:
                new_grid[i, j] = 0  # Dead
    return new_grid

# Function to plot the grid
def plot_grid(grid, ax):
    ax.clear()
    # Plot the grid as an image
    ax.imshow(grid, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])

# Animation function
def animate(frame):
    global grids, mean_liveness_values
    # Update each grid
    for idx, (grid, ax) in enumerate(zip(grids, axs)): 
        grids[idx] = update_grid(grid)  # Update the grid
        mean_liveness = np.mean(grid)
        mean_liveness_values[idx].append(mean_liveness)  # Update mean liveness
        
        # Plot the updated grid
        plot_grid(grid, ax)
    
    # Update the live plot with the new mean liveness values
    for idx, line in enumerate(lines):
        line.set_data(range(len(mean_liveness_values[idx])), mean_liveness_values[idx])
    
    # Adjust the limits of the live plot dynamically
    ax_live.relim()
    ax_live.autoscale_view()

# Create figure and gridspec for animations (top row) and live plot (bottom row)
fig = plt.figure(figsize=(20, 20))
gs = GridSpec(3, 3, height_ratios=[2, 2, 1])  # Adjusted for 5 animations and live plot

# Create subplots for the five animations
axs = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(5)]

# Create the live plot for the mean liveness (bottom row, spanning all columns)
ax_live = fig.add_subplot(gs[2, :])
ax_live.set_title('Mean Liveness')
ax_live.set_xlabel('Generation')
ax_live.set_ylabel(r'$\langle a \rangle$')
lines = [ax_live.plot([], [], color=colors[i], label=f'f = {f_values[i]}')[0] for i in range(len(f_values))]
ax_live.legend(loc='upper right')
ax_live.set_xlim(0, frames)
ax_live.set_ylim(0, 1)

# Set titles for each animation subplot to indicate the value of f
for idx, f in enumerate(f_values):
    axs[idx].set_title(f'f = {f}')

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=frames, interval=100, repeat=False)

# Show the animation with live plot
plt.tight_layout()
plt.show()
