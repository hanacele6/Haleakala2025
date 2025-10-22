import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Simulation Parameters ---
N_SAMPLES = 500000  # Number of samples

# --- Generate samples following a Lambertian distribution ---
# Generate zenith angle θ
# Sampled such that the PDF is P(θ) = sin(2θ)/2
u2 = np.random.random(N_SAMPLES)
sin_theta = np.sqrt(u2)
cos_theta = np.sqrt(1 - sin_theta**2)

# Generate azimuthal angle φ (uniformly distributed from 0 to 2π)
phi_rad = 2 * np.pi * np.random.random(N_SAMPLES)

# --- Convert to 3D Cartesian coordinates ---
# Conversion from spherical to Cartesian coordinates
x = sin_theta * np.cos(phi_rad)
y = sin_theta * np.sin(phi_rad)
z = cos_theta

# ==============================================================================
# Graph: 3D Scatter Plot of Lambertian Distribution
# ==============================================================================
print("--- Displaying 3D Scatter Plot of Lambertian Distribution ---")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
fig.suptitle('3D Scatter Plot of Lambertian Distribution', fontsize=16)

# Plot a subset of points because plotting all is slow (e.g., first 5000)
n_plot = 5000
ax.scatter(x[:n_plot], y[:n_plot], z[:n_plot], s=5, alpha=0.6, label=f'Sampled Points (first {n_plot})')

# Draw a wireframe of a hemisphere to make the shape clear
u_sphere = np.linspace(0, 2 * np.pi, 100)
v_sphere = np.linspace(0, np.pi / 2, 50)
sphere_x = np.outer(np.cos(u_sphere), np.sin(v_sphere))
sphere_y = np.outer(np.sin(u_sphere), np.sin(v_sphere))
sphere_z = np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color='gray', alpha=0.2, rstride=5, cstride=5)

# --- Formatting the plot ---
ax.set_title('Distribution of points on a hemisphere', fontsize=14)
ax.set_xlabel('X axis', fontsize=12)
ax.set_ylabel('Y axis', fontsize=12)
ax.set_zlabel('Z axis (Normal)', fontsize=12)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1])

# Adjust the aspect ratio to make it look better
ax.set_box_aspect((1, 1, 0.5))

# Adjust the viewing angle
ax.view_init(elev=20, azim=45)
ax.legend()
ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()