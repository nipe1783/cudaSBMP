import csv
import numpy as np
import matplotlib.pyplot as plt

# File paths
controls_path = "/home/nic/dev/research/cudaSBMP/build/controls.csv"
samples_path = "/home/nic/dev/research/cudaSBMP/build/samples.csv"

# Read controls from CSV
controls = []

with open(controls_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # Filter out empty strings and convert remaining values to float
        filtered_row = [float(x) for x in row if x]
        controls.append(filtered_row)

# Initial state: [x, y, psi, v]
initial_state = [0, 0, 0, 0]

# Time step (assuming a fixed time step for simplicity)
dt = 0.1
L = 1.0  # Example car length

# List to store all trajectories
all_trajectories = []

# Start with the initial state
current_states = [initial_state]

# Iterate through each row of control inputs
for control_row in controls:
    new_states = []
    for state in current_states:
        for i in range(0, len(control_row), 2):
            steeringAngle = control_row[i]
            acceleration = control_row[i + 1]

            x, y, psi, v = state

            x_new = x + v * np.cos(psi) * dt
            y_new = y + v * np.sin(psi) * dt
            psi_new = psi + (v / L) * np.tan(steeringAngle) * dt
            v_new = v + acceleration * dt

            new_state = [x_new, y_new, psi_new, v_new]
            new_states.append(new_state)

    # Store the new states and prepare for the next iteration
    all_trajectories.append(new_states)
    current_states = new_states

# Convert all trajectories to a single numpy array for plotting
all_points = np.array([state for sublist in all_trajectories for state in sublist])

# Plot the paths of the car
plt.figure(figsize=(10, 6))
plt.plot(all_points[:, 0], all_points[:, 1], marker='o')
plt.title('Car Paths from Multiple Control Inputs')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
plt.show()
