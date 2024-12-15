import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Kuramoto model parameters
N = 50  # Number of oscillators
K = 1.75  # Coupling constant
dt = 0.05  # Time step
T = 20  # Total time for simulation
steps = int(T / dt)

# Natural frequencies with a narrow distribution for better synchronization
np.random.seed(0)
omega = 2 * np.pi * (np.random.normal(0, 0.1, N))  # Narrower frequency distribution

# Initialize phases randomly
theta = 2 * np.pi * np.random.rand(N)

# Function to update the phases based on the Kuramoto model
def kuramoto_step(theta, omega, K, dt):
    dtheta = omega + (K / N) * np.sum(np.sin(theta - theta[:, None]), axis=1)
    return theta + dtheta * dt

# Function to compute the synchronization order parameter R(t) as a complex number
def compute_order_parameter(theta):
    return np.sum(np.exp(1j * theta)) / N

# Set up the figure and axes
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Kuramoto Model: Correlation Matrix, Polar Plot, and Synchronization Order", fontsize=16)

# Correlation matrix plot
corr_matrix_plot = ax1.imshow(np.zeros((N, N)), vmin=-1, vmax=1, cmap='coolwarm')
ax1.set_title("Correlation Matrix")

# Polar plot setup
ax2 = plt.subplot(132, projection='polar')
ax2.set_title("Oscillator Phases")
points, = ax2.plot([], [], 'o', markersize=4)

# Arrow annotation for the synchronization order parameter R(t)
arrow = ax2.annotate('', xy=(0, 0), xytext=(0, 0),
                     arrowprops=dict(facecolor='blue', edgecolor='blue', shrink=0, width=2))

# Line to trace the trajectory of R(t) in polar plot
R_trail, = ax2.plot([], [], color='blue', lw=1)

# Line plot for the synchronization order parameter R(t)
ax3.set_xlim(0, T)
ax3.set_ylim(0, 1)
ax3.set_title("Synchronization Order Parameter R(t)")
ax3.set_xlabel("Time")
ax3.set_ylabel("R(t)")
order_parameter_line, = ax3.plot([], [], lw=2)

# Initialize arrays to store R(t) over time
time_data = np.linspace(0, T, steps)
R_data = np.zeros(steps)
R_trail_data = np.zeros((steps, 2))  # Stores trail of R(t) for the polar plot

# Animation update function
def update(frame):
    global theta
    
    # Update the phases using the Kuramoto step
    theta = kuramoto_step(theta, omega, K, dt)

    # Update the correlation matrix
    corr_matrix = np.cos(theta[:, None] - theta)
    corr_matrix_plot.set_array(corr_matrix)

    # Update the polar plot
    points.set_data(theta, np.ones(N))
    
    # Update R(t) as a complex number and get its magnitude and phase
    R_complex = compute_order_parameter(theta)
    R_magnitude = np.abs(R_complex)
    R_phase = np.angle(R_complex)

    # Update the arrow to represent R(t)
    arrow.xy = (R_phase, R_magnitude)  # Arrow points to the current R vector
    arrow.xytext = (0, 0)  # Start at the center
    
    # Update the R(t) trail
    R_trail_data[frame] = (R_phase, R_magnitude)
    R_trail.set_data(R_trail_data[:frame+1, 0], R_trail_data[:frame+1, 1])
    
    # Update the order parameter plot
    R_data[frame] = R_magnitude
    order_parameter_line.set_data(time_data[:frame+1], R_data[:frame+1])
    
    return corr_matrix_plot, points, order_parameter_line, arrow, R_trail

# Create animation
ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=True)

# Show the animation
plt.show()
