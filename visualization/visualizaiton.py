import csv
import numpy as np
import matplotlib.pyplot as plt

# File paths
controls_path = "/home/nic/dev/research/cudaSBMP/build/controls.csv"
samples_path = "/home/nic/dev/research/cudaSBMP/build/samples.csv"

# Read samples from CSV
samples = []
with open(samples_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        samples.append([float(x) for x in row])

samples = np.array(samples)

# Extract x and y coordinates
x_coords = samples[:, 0]
y_coords = samples[:, 1]

# Plot the trajectory
plt.figure(figsize=(10, 6))
plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b')
plt.title('Trajectory of Samples')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()
