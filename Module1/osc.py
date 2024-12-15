import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
grid_size = 100  # Smaller grid size for better visualization
frames = 200  # Number of generations to animate
speed_up_factor = 50  # Lower value for faster animation

# Function to initialize the qutub pattern with semi-quantum values
def initialize_qutub():
    grid = np.zeros((grid_size, grid_size))
    
    # Define semi-quantum values between 0 and 1
    a1 = a2=0.55
    a3=a4=0.5 # Top-left corner




    
    grid[50, 51] = 1  # Above a1
    grid[50, 49] = 1  # Above a2
    grid[51, 50] = 1  # Left of a1
    grid[49, 50] = 1  # Left of a3
 

    # Central cell remains dead (0)
    grid[49, 49] = a1  # a1 (alive)
    grid[49, 51] = a2  # a2 (alive)
    grid[51, 49] = a3  # a3 (alive)
    grid[51, 51] = a4  # a4 (alive)
    grid[50, 50] = 0 # Central cell (dead)
    
    return grid

# Function to update the grid based on the given rules
def update_grid(grid):
    new_grid = np.copy(grid)
    for i in range(grid_size):
        for j in range(grid_size):
            neighborhood = grid[max(0, i-1):min(i+2, grid_size), max(0, j-1):min(j+2, grid_size)]
            A = np.sum(neighborhood)  # Liveness sum of neighbors
            
            # Clamp grid values between -1 and 1 to prevent invalid sqrt results
            clamped_value = np.clip(grid[i, j], -1, 1)
            l = np.sqrt(2) +1
            
            # Apply the rules
            if A >= 0 and A <= 1:
                new_grid[i, j] = 0  # Dead
            elif A > 1 and A <= 2:
                new_grid[i, j] = (A - 1) * clamped_value  # Semi-live (gray)
            elif A > 2 and A <= 3:
                new_grid[i, j] = l * clamped_value + (A - 2) * (np.sqrt(1 - clamped_value ** 2))  # Semi-live (gray)
            elif A > 3 and A <= 4:
                new_grid[i, j] = l * (-A + 4) * (clamped_value) + l * (-A + 4) * (np.sqrt(1 - clamped_value ** 2))  # Live
            else:
                new_grid[i, j] = 0  # Dead
        
    return new_grid

# Function to plot the grid
def plot_grid(grid, ax):
    ax.clear()
    ax.imshow(grid, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])

# Animation functions for qutub
def animate_qutub(frame):
    global qutub_grid
    qutub_grid = update_grid(qutub_grid)
    plot_grid(qutub_grid, ax_qutub)

# Create figure for the animations
fig = plt.figure(figsize=(10, 10))
ax_qutub = fig.add_subplot(111)
ax_qutub.set_title("Semi-Quantum Game of Life - Qutub")

# Initialize grid with qutub pattern
qutub_grid = initialize_qutub()

# Create animations for Qutub
ani_qutub = animation.FuncAnimation(fig, animate_qutub, frames=frames, interval=speed_up_factor, repeat=False)

# Show the animations
plt.tight_layout()
plt.show()
