import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize grid size
grid_size = 100
state_grid = np.random.rand(grid_size, grid_size)  # Random initial liveliness between 0 and 1

# Operators B, D, S as defined in the paper
B = np.array([[1, 0], [1, 0]])  # Birth operator
D = np.array([[0, 1], [0, 1]])  # Death operator
S = np.array([[1, 0], [0, 1]])  # Survival operator

# Function to calculate operator based on value of A
def G_hat(A):
    if 0 <= A < 1:
        return D  # Death
    elif 1 <A <= 2:
        return (np.sqrt(2+1)*(2-A)*D+(A-1)*S)  # Birth
    elif 2 < A <= 3:
        return (np.sqrt(2+1)*(3-A)*S+(A-1)*B)  # Survival
    elif 3 <A <= 4:

        return (np.sqrt(2+1)*(4-A)*B+(A-1)*D)  # Identity matrix (equivalent to doing nothing)
    else:
        return D  # Identity matrix for A >= 4

# Calculate liveliness A for each cell based on Moore neighborhood
def compute_liveness(state_grid):
    padded_grid = np.pad(state_grid, pad_width=1, mode='constant', constant_values=0)
    liveness = np.zeros(state_grid.shape)
    for i in range(1, state_grid.shape[0] + 1):
        for j in range(1, state_grid.shape[1] + 1):
            neighborhood = padded_grid[i-1:i+2, j-1:j+2]
            liveness[i-1, j-1] = np.sum(neighborhood) - padded_grid[i, j]
    return liveness

# Function to update the grid based on liveness and operators
def update_grid(state_grid):
    liveness_grid = compute_liveness(state_grid)
    new_state = np.zeros(state_grid.shape)
    for i in range(state_grid.shape[0]):
        for j in range(state_grid.shape[1]):
            A = liveness_grid[i, j]
            G = G_hat(A)
            # The state update depends on the matrix form of G, specifically G[0, 0] or similar
            new_state[i, j] = G[0, 0] * state_grid[i, j]  # Update with G operator's effect
    return new_state

# Initialize the grid
def init():
    grid.set_data(state_grid)
    return [grid]

# Update function for animation
def animate(i):
    global state_grid
    new_state = update_grid(state_grid)
    grid.set_data(new_state)
    state_grid = new_state
    return [grid]

# Plot setup
fig, ax = plt.subplots()
grid = ax.imshow(state_grid, cmap='viridis', vmin=0, vmax=1)

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=50, init_func=init, blit=True)

# Show the animation
plt.show()
