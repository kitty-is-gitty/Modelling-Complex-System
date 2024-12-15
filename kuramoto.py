import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the parameters
N = 1000  # Number of oscillators
omega_mean = 0.0  # Mean natural frequency
omega_std = 1.0  # Std deviation of natural frequencies
K_c = 2.0  # Critical coupling constant (example value)

# Initialize natural frequencies from a Gaussian distribution
omega = np.random.normal(omega_mean, omega_std, N)

# Define the Kuramoto model
def kuramoto(theta, t, K):
    r = np.abs(np.sum(np.exp(1j * theta)) / N)  # Order parameter r
    psi = np.angle(np.sum(np.exp(1j * theta)) / N)  # Mean phase
    dtheta_dt = omega + K * r * np.sin(psi - theta)  # Kuramoto equation
    return dtheta_dt

# Time variables
t = np.linspace(0, 100, 500)  # Time array for integration

# Coupling constants for the two graphs
K_values = [1.5, 2.5]  # Below and above K_c

# Simulation and plotting
plt.figure(figsize=(10, 8))

# First plot: r(t) for K < K_c and K > K_c
for i, K in enumerate(K_values):
    # For K < K_c, start with a more coherent initial phase distribution
    if K < K_c:
        theta0 = np.random.normal(0, 0.5, N)  # Coherent initial condition
    else:
        theta0 = np.random.uniform(0, 2 * np.pi, N)  # Random initial condition
    
    # Integrate the Kuramoto model
    theta_t = odeint(kuramoto, theta0, t, args=(K,))
    
    # Compute the order parameter r(t) over time
    r_t = np.abs(np.sum(np.exp(1j * theta_t), axis=1) / N)
    
    # Plot r(t) for the two values of K
    label = r'$K > K_c$' if K > K_c else r'$K < K_c$'
    plt.plot(t, r_t, label=label)

plt.xlabel(r'$t$')
plt.ylabel(r'$r$')
plt.legend()
plt.title(r'Order Parameter $r$ vs Time $t$')
plt.grid()
plt.show()

# Second plot: r_infinity vs K
K_values = np.linspace(0, 4, 50)  # Range of K values
r_inf = []

for K in K_values:
    theta0 = np.random.uniform(0, 2 * np.pi, N)
    theta_t = odeint(kuramoto, theta0, t, args=(K,))
    r_t = np.abs(np.sum(np.exp(1j * theta_t[-1])) / N)  # r at t -> infinity
    r_inf.append(r_t)

plt.figure(figsize=(8, 6))
plt.plot(K_values, r_inf, label=r'$r_\infty$', color='black')
plt.axvline(K_c, color='gray', linestyle='--', label=r'$K_c$')
plt.axhline(1, color='gray', linestyle='--')
plt.xlabel(r'$K$')
plt.ylabel(r'$r_\infty$')
plt.title(r'Order Parameter $r_\infty$ vs Coupling $K$')
plt.legend()
plt.grid()
plt.show()
