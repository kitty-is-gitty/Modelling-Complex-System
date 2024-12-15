import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# Parameters
grid_size = 100
frames = 200  # Number of generations to animate
speed_up_factor = 50  # Lower value for faster animation

# Function to initialize the glider pattern in the top-left corner
def initialize_glider():
    grid = np.zeros((grid_size, grid_size))
    glider = [
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ]
    grid[1:4, 1:4] = glider  # Positioning glider in the top-left corner
    return grid

# Function to initialize the Blinker oscillator pattern in the center of the grid
def initialize_blinker():
    grid = np.zeros((grid_size, grid_size))
    blinker = [
        [1],
        [1],
        [1]
    ]
    start_row = grid_size // 2 - 1
    start_col = grid_size // 2
    grid[start_row:start_row+3, start_col:start_col+1] = blinker  # Positioning blinker in the center
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

# Function to update the grid based on Quantum Game of Life rules
def update_quantum(grid):
    new_grid = np.copy(grid)
    # Iterate over each cell and apply the rules based on the neighboring cells
    for i in range(grid_size):
        for j in range(grid_size):
            # Get the Moore neighborhood (8 surrounding cells)
            neighborhood = grid[max(0, i-1):min(i+2, grid_size), max(0, j-1):min(j+2, grid_size)]
            A = np.sum(neighborhood)  # Liveness sum of neighbors
            
            # Clamp grid values between -1 and 1 to prevent invalid sqrt results
            clamped_value = np.clip(grid[i, j], -1, 1)
            l=(np.sqrt(2))
            # Apply the rules from Table I
            if A >= 0 and A <= 1:
                new_grid[i, j] = 0  # Dead
            elif A > 1 and A <= 2:
                new_grid[i, j] = (A-1)*clamped_value  # Semi-live (gray)
            elif A > 2 and A <= 3:
                new_grid[i, j] = l*clamped_value + (A-2)*(np.sqrt(1-clamped_value**2))  # Semi-live (gray)
            elif A > 3 and A <= 4:
                new_grid[i, j] = l*(-A+4)*(clamped_value) + l*(-A+4)*(np.sqrt(1-clamped_value**2))  # Live
            else:
                new_grid[i, j] = 0  # Dead
        
    return new_grid

# Function to plot the grid
def plot_grid(grid, ax):
    ax.clear()
    ax.imshow(grid, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])

# Animation functions for glider
def animate_conway_glider(frame):
    global conway_grid_glider
    conway_grid_glider = update_conway(conway_grid_glider)
    plot_grid(conway_grid_glider, ax_conway_glider)

def animate_quantum_glider(frame):
    global quantum_grid_glider
    quantum_grid_glider = update_quantum(quantum_grid_glider)
    plot_grid(quantum_grid_glider, ax_quantum_glider)

# Animation functions for blinker
def animate_conway_blinker(frame):
    global conway_grid_blinker
    conway_grid_blinker = update_conway(conway_grid_blinker)
    plot_grid(conway_grid_blinker, ax_conway_blinker)

def animate_quantum_blinker(frame):
    global quantum_grid_blinker
    quantum_grid_blinker = update_quantum(quantum_grid_blinker)
    plot_grid(quantum_grid_blinker, ax_quantum_blinker)

# Create figure and gridspec for the animations
fig = plt.figure(figsize=(12, 8))  # Adjust size for fewer plots
gs = GridSpec(2, 2)  # 2 rows, 2 columns

# Create the subplots for Conway's Game of Life animations
ax_conway_glider = fig.add_subplot(gs[0, 0])
ax_conway_blinker = fig.add_subplot(gs[1, 0])
ax_conway_glider.set_title("Conway's Game of Life - Glider")
ax_conway_blinker.set_title("Conway's Game of Life - Blinker")

# Create the subplots for Quantum Game of Life animations
ax_quantum_glider = fig.add_subplot(gs[0, 1])
ax_quantum_blinker = fig.add_subplot(gs[1, 1])
ax_quantum_glider.set_title("Quantum Game of Life - Glider")
ax_quantum_blinker.set_title("Quantum Game of Life - Blinker")

# Initialize grids with respective patterns
conway_grid_glider = initialize_glider()
quantum_grid_glider = initialize_glider()

conway_grid_blinker = initialize_blinker()
quantum_grid_blinker = initialize_blinker()

# Create animations for Glider
ani_conway_glider = animation.FuncAnimation(fig, animate_conway_glider, frames=frames, interval=speed_up_factor, repeat=False)
ani_quantum_glider = animation.FuncAnimation(fig, animate_quantum_glider, frames=frames, interval=speed_up_factor, repeat=False)

# Create animations for Blinker
ani_conway_blinker = animation.FuncAnimation(fig, animate_conway_blinker, frames=frames, interval=speed_up_factor, repeat=False)
ani_quantum_blinker = animation.FuncAnimation(fig, animate_quantum_blinker, frames=frames, interval=speed_up_factor, repeat=False)

# Show the animations
plt.tight_layout()
plt.show()
