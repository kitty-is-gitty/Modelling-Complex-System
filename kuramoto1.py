import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters for the Kuramoto model
N = 50  # Number of oscillators
K = 10.0  # Strong coupling strength to promote synchronization
time_steps = 500  # Extended time steps to observe full synchronization
dt = 0.05  # Smaller time step for smoother convergence

# Initial random phases and intrinsic frequencies
np.random.seed(0)
phases = 2 * np.pi * np.random.rand(N)
natural_frequencies = 0.05 * np.random.randn(N)  # Reduced frequency variation

# Calculate phase differences
def phase_diffs(phases):
    return np.sin(phases[:, None] - phases)

# Update function for each time step in the Kuramoto model
def update_phases(phases):
    global K, dt
    diff_matrix = phase_diffs(phases)
    return phases + dt * (natural_frequencies + (K / N) * np.sum(diff_matrix, axis=1))

# Setting up the figure layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_aspect('equal')
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax2.set_xlim(0, time_steps * dt)
ax2.set_ylim(0, 1)

# Plotting elements for oscillators and order parameter
points, = ax1.plot([], [], 'o', markersize=5, color='tab:blue')
order_param_line, = ax2.plot([], [], label='Sync. Parameter', color='tab:blue')
order_vector, = ax1.plot([], [], color='purple', linewidth=2, label='Order Parameter')
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax1.set_xlabel('Re')
ax1.set_ylabel('Im')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel(r'$r$')

# Data arrays for the order parameter (r) over time
sync_params = []

# Initialize function
def init():
    points.set_data([], [])
    order_param_line.set_data([], [])
    order_vector.set_data([], [])
    return points, order_param_line, order_vector

# Animation function
def animate(i):
    global phases
    phases = update_phases(phases)

    # Plot oscillators on unit circle
    x = np.cos(phases)
    y = np.sin(phases)
    points.set_data(x, y)

    # Calculate the order parameter
    order_parameter = np.mean(np.exp(1j * phases))
    sync_params.append(abs(order_parameter))
    order_param_line.set_data(np.linspace(0, i * dt, i + 1), sync_params)

    # Plot the order parameter vector
    order_vector.set_data([0, order_parameter.real], [0, order_parameter.imag])

    return points, order_param_line, order_vector

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=time_steps, init_func=init, blit=True)

# Show the animation
plt.show()
