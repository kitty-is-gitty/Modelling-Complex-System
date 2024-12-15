import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
N = 15          # Number of oscillators (nodes)
K = 2.0         # Coupling strength
T = 200         # Number of time steps
dt = 0.05       # Time step
adapt_rate = 0.1  # Adaptation rate for edge weights

# Create an initial graph
G = nx.erdos_renyi_graph(N, 0.4)  # Random graph
pos = nx.spring_layout(G, seed=42)  # Fixed positions for visualization

# Initialize random phases and natural frequencies
phases = np.random.uniform(0, 2 * np.pi, N)
natural_frequencies = np.random.uniform(-1.0, 1.0, N)  # Random natural frequencies

# Initialize edge weights (coupling strength between nodes)
weights = {edge: 1.0 for edge in G.edges()}

# Function to compute phase updates
def compute_phase_update(phases, weights):
    dphase = np.zeros(N)
    for i, j in G.edges():
        weight = weights[(i, j)]
        dphase[i] += weight * np.sin(phases[j] - phases[i])
        dphase[j] += weight * np.sin(phases[i] - phases[j])  # Symmetric interaction
    return dphase

# Function to adapt edge weights
def adapt_weights(phases, weights):
    new_weights = {}
    for (i, j), weight in weights.items():
        phase_diff = np.abs(np.sin(phases[j] - phases[i]))
        new_weights[(i, j)] = weight + adapt_rate * (1 - phase_diff - weight) * dt
        new_weights[(i, j)] = max(0, new_weights[(i, j)])  # Ensure weights are non-negative
    return new_weights

# Animation update function
def update(frame):
    global phases, weights

    # Compute phase dynamics
    dphase = compute_phase_update(phases, weights)
    phases += (natural_frequencies + K * dphase) * dt
    phases %= 2 * np.pi  # Keep phases within [0, 2π]

    # Adapt edge weights
    weights = adapt_weights(phases, weights)

    # Update node colors based on phases
    node_colors = [plt.cm.hsv(phase / (2 * np.pi)) for phase in phases]

    # Clear and redraw the network
    ax1.clear()
    nx.draw(
        G, pos, node_color=node_colors, with_labels=True,
        node_size=800, edge_color="gray", ax=ax1,
        width=[weights[edge] * 3 for edge in G.edges()]
    )
    ax1.set_title("Dynamic Kuramoto Network", fontsize=16)
    ax1.axis("off")

    # Update the coherence plot
    ax2.clear()
    r = compute_order_parameter(phases)
    r_values.append(r)
    ax2.plot(r_values, color='blue', label='Coherence (r)')
    ax2.set_ylim(0, 1.1)
    ax2.set_xlim(0, T)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Coherence (r)")
    ax2.set_title("Synchronization Over Time", fontsize=14)
    ax2.legend(loc="upper right")

    return ax1, ax2

# Function to compute the order parameter (r)
def compute_order_parameter(phases):
    mean_field = np.mean(np.exp(1j * phases))  # Complex order parameter
    return np.abs(mean_field)  # Magnitude represents coherence

# Initialize order parameter storage
r_values = []

# Set up the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.tight_layout()

# Run the animation
ani = FuncAnimation(fig, update, frames=T, interval=50, blit=False, repeat=False)
plt.show()
