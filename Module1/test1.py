import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from IPython.display import HTML

# Parameters
grid_size = 100
frames = 500  # Increased number of generations to animate
speed_up_factor = 50  # Lower value for faster animation

# Function to initialize the beacon pattern in the center of the grid
def initialize_beacon():
    grid = np.zeros((grid_size, grid_size))
    beacon = [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ]
    center = grid_size // 2  # Place it in the center of the grid
    grid[center-2:center+2, center-2:center+2] = beacon
    return grid

# Function to update the grid based on Conway's Game of Life rules
def update_conway(grid):
    new_grid = np.copy(grid)
    for i in range(grid_size):
        for j in range(grid_size):
            neighborhood = grid[max(0, i-1):min(i+2, grid_size), max(0, j-1):min(j+2, grid_size)]
            A = np.sum(neighborhood) - grid[i, j]  # Live neighbors
            
            if grid[i, j] == 1:  # Live cell
                if A < 2 or A > 3:
                    new_grid[i, j] = 0  # Dies
                else:
                    new_grid[i, j] = 1  # Survives
            else:  # Dead cell
                if A == 3:
                    new_grid[i, j] = 1  # Becomes alive
    return new_grid

# Function to update the grid based on modified Quantum Game of Life rules
def update_quantum(grid):
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
    ax.imshow(grid, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])

# Animation function to update both Conway's and Quantum beacon side by side
def animate_beacon(frame):
    global conway_grid_beacon, quantum_grid_beacon
    conway_grid_beacon = update_conway(conway_grid_beacon)
    quantum_grid_beacon = update_quantum(quantum_grid_beacon)
    plot_grid(conway_grid_beacon, ax_conway_beacon)
    plot_grid(quantum_grid_beacon, ax_quantum_beacon)

# Create figure and gridspec for the animations (side-by-side Beacon animations)
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(1, 2)  # 1 row, 2 columns for Conway's and Quantum Beacon animations

# Create the subplots for Conway's and Quantum Game of Life beacon animations
ax_conway_beacon = fig.add_subplot(gs[0, 0])
ax_quantum_beacon = fig.add_subplot(gs[0, 1])
ax_conway_beacon.set_title("Conway's Game of Life - Beacon")
ax_quantum_beacon.set_title("Quantum Game of Life - Beacon")

# Initialize grids with beacon pattern
conway_grid_beacon = initialize_beacon()
quantum_grid_beacon = initialize_beacon()

# Create the combined animation with repeat=True for continuous animation
ani_beacon = animation.FuncAnimation(fig, animate_beacon, frames=frames, interval=speed_up_factor, repeat=True)

# Convert the combined animation to JSHTML format and display in Jupyter
HTML(ani_beacon.to_jshtml())
