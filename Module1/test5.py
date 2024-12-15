import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the quantum operators
B = np.array([[1, 0], [0, 0]])  # Birth operator
D = np.array([[0, 0], [0, 1]])  # Death operator
S = np.array([[0, 1], [1, 0]])  # Survival operator

# Parameters
grid_size = 100  # 100x100 grid as in the example
f = 0.2  # Fraction of cells initially alive
generations = 100  # Number of generations to run

# Initialize the grid with the fraction f of cells alive, liveness parameter a randomly distributed
def initialize_grid(grid_size, f):
    grid = np.zeros((grid_size, grid_size, 2))  # (a, b) components
    num_alive = int(f * grid_size * grid_size)  # Number of alive cells

    # Randomly select cells to be alive and assign random liveness 'a'
    alive_indices = np.random.choice(grid_size * grid_size, num_alive, replace=False)
    
    for idx in range(grid_size * grid_size):
        i, j = divmod(idx, grid_size)
        if idx in alive_indices:
            a = np.random.uniform(0, 1)  # Random liveness 'a' between 0 and 1
            b = np.sqrt(1 - a**2)  # Ensure normalization with a^2 + b^2 = 1
            grid[i, j] = np.array([a, b])
        else:
            grid[i, j] = np.array([0, 1])  # Dead (a=0, b=1)
    
    return grid

# Normalize quantum states in each cell
def normalize(cell):
    norm = np.linalg.norm(cell)
    if norm == 0:
        return cell
    return cell / norm

# Apply operator based on liveness A
def apply_operator(cell, A):
    if A <= 1:
        G = D
    elif 1 < A <= 2:
        G = (2 + 1)*(2 - A) * D + (A - 1) * S
    elif 2 < A <= 3:
        G = (2 + 1)*(3 - A) * S + (A - 2) * B
    elif 3 < A <= 4:
        G = (2 + 1)*(4 - A) * B + (A - 3) * D
    else:
        G = D
    return np.dot(G, cell)

# Compute liveness A of Moore neighborhood
def compute_liveness(grid, i, j):
    neighbors = grid[max(i-1, 0):min(i+2, grid_size), max(j-1, 0):min(j+2, grid_size), 0]
    A = np.sum(neighbors)  # Exclude the cell itself
    return A

# Update the grid for each generation
def update_grid(grid):
    new_grid = np.zeros_like(grid)
    for i in range(grid_size):
        for j in range(grid_size):
            A = compute_liveness(grid, i, j)
            new_grid[i, j] = normalize(apply_operator(grid[i, j], A))
    return new_grid

# Calculate the mean liveness (mean value of 'a')
def calculate_mean_liveness(grid):
    return np.mean(grid[:, :, 0])

# Initialize the grid with the specified fraction of alive cells
grid = initialize_grid(grid_size, f)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Subplot 1: Display the grid (evolution of liveness)
img = ax1.imshow(grid[:, :, 0], cmap='gray', vmin=0, vmax=1)  # Display 'a' component as grayscale (0=black, 1=white)
ax1.set_title('Grid Evolution')

# Subplot 2: Plot mean liveness over time
mean_liveness = []
line, = ax2.plot([], [], lw=2)
ax2.set_xlim(0, generations)
ax2.set_ylim(0, 1)
ax2.set_title('Mean Liveness over Generations')
ax2.set_xlabel('Generation')
ax2.set_ylabel('Mean Liveness')

# Animation update function
def animate(frame):
    global grid, mean_liveness
    grid = update_grid(grid)  # Update the grid for the next generation
    img.set_data(grid[:, :, 0])  # Update the image with the new 'a' values

    # Calculate and store the mean liveness
    mean_liveness.append(calculate_mean_liveness(grid))

    # Update the liveness plot
    line.set_data(np.arange(len(mean_liveness)), mean_liveness)
    
    return [img, line]

# Generate the animation
ani = FuncAnimation(fig, animate, frames=generations, interval=200, blit=True)

# Save or display the animation
plt.show()
