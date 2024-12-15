import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create figure and axis
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Set the center
center = (0, 0)

# Length of the main edges and the V-shaped edges
main_length = 4
v_length = main_length  # Length of the V-shaped edges

# Number of iterations for drawing V-shaped edges
iterations = 3  # Set the number of iterations here

# Initialize the lines list
main_lines = []
v_lines = []

# Function to generate V-shaped edges from a given point and direction
def generate_v(x, y, angle, v_length):
    # Create two V-shaped branches at 45 degrees and -45 degrees from the direction of the edge
    v_edges = []
    for angle_offset in [np.pi / 4, -np.pi / 4]:  # V shape (two branches)
        v_angle = angle + angle_offset
        x_v = x + v_length * np.cos(v_angle)
        y_v = y + v_length * np.sin(v_angle)
        v_edges.append(((x, x_v), (y, y_v)))
    return v_edges

# Function to draw V shapes recursively
def draw_recursive_v(x, y, angle, v_length, depth):
    if depth == 0:
        return []
    
    v_edges = generate_v(x, y, angle, v_length)
    
    # Draw the V-shaped edges
    for line in v_edges:
        x_start, x_end = line[0]
        y_start, y_end = line[1]
        ax.plot([x_start, x_end], [y_start, y_end], 'k-')
        
        # Recurse to draw V shapes from the new V endpoints
        draw_recursive_v(x_end, y_end, np.arctan2(y_end - y_start, x_end - x_start), v_length, depth - 1)
    
    return v_edges

def draw_lines(frame):
    # Clear previous lines
    ax.clear()


    # Draw the 4 perpendicular main edges
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        x_end = center[0] + main_length * np.cos(angle)
        y_end = center[1] + main_length * np.sin(angle)
        main_lines.append(((center[0], x_end), (center[1], y_end)))
    
    # Draw the main lines
    for line in main_lines:
        x_start, x_end = line[0]
        y_start, y_end = line[1]
        ax.plot([x_start, x_end], [y_start, y_end], 'k-')
        
        # Generate V-shaped edges from the end of the main edge
        x, y = x_end, y_end
        angle = np.arctan2(y_end - y_start, x_end - x_start)
        draw_recursive_v(x, y, angle, v_length, iterations)

# Create the animation
ani = FuncAnimation(fig, draw_lines, frames=1, interval=500, repeat=False)

plt.show()
