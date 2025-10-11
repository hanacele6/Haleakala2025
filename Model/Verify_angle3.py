import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Lambertian (Cosine) Distribution Sampling ---
# Number of samples (reduced for better 3D visualization)
N_SAMPLES = 5000

# Generate two sets of uniform random numbers between 0 and 1
u1 = np.random.rand(N_SAMPLES)
u2 = np.random.rand(N_SAMPLES)

# For a Lambertian distribution, the probability of emission is proportional to cos(θ).
# Using the inverse transform sampling method:
# 1. The azimuthal angle φ is uniformly distributed from 0 to 2π.
phi = 2 * np.pi * u1

# 2. The polar angle θ is sampled such that cos(θ) = sqrt(U2).
#    This makes samples more likely to be near the normal (θ=0).
cos_theta = np.sqrt(u2)
sin_theta = np.sqrt(1 - cos_theta**2)

# --- Convert Spherical Coordinates to 3D Cartesian Coordinates ---
# x = r * sin(θ) * cos(φ)
# y = r * sin(θ) * sin(φ)
# z = r * cos(θ)
# Assuming a unit hemisphere (r=1)
x = sin_theta * np.cos(phi)
y = sin_theta * np.sin(phi)
z = cos_theta

# --- Plotting the Result in 3D ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
fig.suptitle('3D Visualization of Lambertian Distribution', fontsize=16)

# Scatter plot of the sampled points
ax.scatter(x, y, z, s=5, alpha=0.6)

# Draw an arrow representing the surface normal vector
#ax.quiver(0, 0, 0, 0, 0, 1.2, color='red', arrow_length_ratio=0.1, label='Surface Normal')

# Set plot limits and labels
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1.3])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend()
ax.set_aspect('equal') # Ensure the hemisphere is not distorted
ax.grid(True)

# Set the viewing angle
ax.view_init(elev=30, azim=45)

plt.show()