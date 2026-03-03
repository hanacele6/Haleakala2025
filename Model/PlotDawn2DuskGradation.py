import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# --- MIV probability distribution function ---
# Replicates the "Dawn:Dusk = 2:1" ratio
def get_miv_probability(lon):
    """
    Takes longitude (lon) and calculates the relative MIV generation probability.
    lon = +pi/2 (Dawn, +Y) -> Max (4/3)
    lon = -pi/2 (Dusk, -Y) -> Min (2/3)
    """
    return 1.0 - (1.0 / 3.0) * np.sin(lon)

# --- Plotting Setup ---
print("Plotting MIV longitude-dependent gradation...")

# 1. Create longitude (lon) and latitude (lat) grids
# lon (phi): -pi to +pi
# lat (theta): -pi/2 to +pi/2
lon = np.linspace(-np.pi, np.pi, 200)  # Longitude resolution
lat = np.linspace(-np.pi/2, np.pi/2, 100) # Latitude resolution
lon_grid, lat_grid = np.meshgrid(lon, lat)

# 2. Calculate spherical coordinates (X, Y, Z)
# (Assuming a sphere with R=1)
X = np.cos(lat_grid) * np.cos(lon_grid)
Y = np.cos(lat_grid) * np.sin(lon_grid)
Z = np.sin(lat_grid)

# 3. Calculate color values
# Color depends only on longitude (lon_grid)
C = get_miv_probability(lon_grid) / (4/3)

# 4. Set color scaling
# Fix the range from the function's min (2/3) to max (4/3)
vmin = 0.5
vmax = 1.0
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = cm.viridis  # Colormap (e.g., 'viridis', 'coolwarm', 'jet')

# 5. Prepare 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 6. Plot the surface
# Pass the normalized and colormapped 'C' values to facecolors
# Set shade=False to display pure function colors without 3D lighting
surf = ax.plot_surface(X, Y, Z,
                       facecolors=cmap(norm(C)),
                       rstride=1, cstride=1,
                       antialiased=True,
                       shade=False) # Disable lighting

# 7. Set axis labels and title (All in English)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.set_title('MIV Generation Probability Distribution', fontsize=16)

# 8. Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])

# 9. Add a colorbar (legend)
m = cm.ScalarMappable(cmap=cmap, norm=norm)
m.set_array(C)
cbar = fig.colorbar(m, ax=ax, shrink=0.6, aspect=20)
cbar.set_label('Relative Generation Probability P(lon)', fontsize=12)

# 10. Display the plot (Changed from savefig to show)
print("\nPlot window is opening...")
print("-------------------------------------------------")
print("Checkpoints:")
print(f" - Dawn side (+Y) should be brightest (Prob: {vmax:.2f})")
print(f" - Dusk side (-Y) should be darkest (Prob: {vmin:.2f})")
print(f" - Sun (+X) and Anti-Sun (-X) sides are mid-color (Prob: 1.0)")
print(f" - Color is uniform along the Z-axis (latitude)")
print("-------------------------------------------------")

plt.show()