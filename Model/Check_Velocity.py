import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
GM_MERCURY = 2.2032e13  # Gravitational parameter [m^3/s^2]
RM = 2.440e6  # Mercury radius [m]

# --- Parameters to Test ---
v0 = 500.0  # Initial Speed [m/s]
launch_angle_deg = 45.0  # Launch angle from horizontal [degrees]

# --- Grid Cell Size (from main sim) ---
# (GRID_MAX_RM * 2) / GRID_RESOLUTION = (5.0 * 2) / 101 approx 0.1 RM
CELL_SIZE_M = (10.0 * RM) / 101.0

# --- 2D Simulation (Ballistic Trajectory) ---
angle_rad = np.deg2rad(launch_angle_deg)

# Initial state (Cartesian, centered on Mercury)
# Particle starts at (x=0, z=Radius)
pos = np.array([0.0, RM])
# Velocity vector based on launch angle
vel = np.array([v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)])

dt = 0.1  # Time step [s]

pos_history = []
time_history = []

current_altitude = 0.0
max_altitude_m = 0.0

time_elapsed = 0.0

print("Running 2D trajectory simulation...")

# Loop until particle hits the ground (altitude < 0)
while current_altitude >= -1.0:  # (Allow for one step slightly below surface)

    # Calculate gravity vector
    r_mag = np.linalg.norm(pos)
    accel_vec = -GM_MERCURY * pos / (r_mag ** 3)

    # Update state using simple Euler-Cromer integration
    vel += accel_vec * dt
    pos += vel * dt

    # Calculate current altitude
    current_altitude = np.linalg.norm(pos) - RM

    # Store data
    max_altitude_m = max(max_altitude_m, current_altitude)

    # Calculate horizontal distance traveled (arc length)
    # arctan2(x, z) gives angle from vertical Z-axis
    horiz_dist_m = RM * np.arctan2(pos[0], pos[1])

    pos_history.append((horiz_dist_m / 1000.0, current_altitude / 1000.0))
    time_history.append(time_elapsed)

    time_elapsed += dt

    if time_elapsed > 30000:  # Safety break for high-speed (escape)
        print("Safety break: Simulation took too long (particle may have escaped).")
        break

# --- Final Results ---
final_range_m = horiz_dist_m

print(f"\n--- Ballistic Hop Verification (Angle = {launch_angle_deg} deg) ---")
print(f"Initial Speed (v0): {v0:.1f} m/s")
print(f"Max Altitude (h_max): {max_altitude_m / 1000.0:.2f} km")
print(f"Max Range (horizontal): {final_range_m / 1000.0:.2f} km")
print(f"--- Comparison with Grid Size ---")
print(f"Your Grid Cell Size (approx 0.1 RM): {CELL_SIZE_M / 1000.0:.2f} km")

# Check if it left the cubic cell
if max_altitude_m > CELL_SIZE_M or final_range_m > CELL_SIZE_M:
    print(f"\n>>> Verification: Success.")
    print(f"    Particle (h_max={max_altitude_m / 1000.0:.1f} km, range={final_range_m / 1000.0:.1f} km)")
    print(f"    CAN cross the grid cell boundary ({CELL_SIZE_M / 1000.0:.1f} km).")
else:
    print(f"\n>>> !!! WARNING: Verification Failed !!!")
    print(f"    Particle (h_max={max_altitude_m / 1000.0:.1f} km, range={final_range_m / 1000.0:.1f} km)")
    print(f"    CANNOT cross the grid cell boundary ({CELL_SIZE_M / 1000.0:.1f} km).")
    print(f"    The bypass threshold (LOW_SPEED_THRESHOLD_M_S) is too low.")

# --- Plotting ---
plot_data = np.array(pos_history)
plot_x = plot_data[:, 0]
plot_y = plot_data[:, 1]

plt.figure(figsize=(10, 8))
plt.plot(plot_x, plot_y, label=f'v0 = {v0} m/s, angle = {launch_angle_deg} deg')

# Draw the "box" of the launch cell
cell_size_km = CELL_SIZE_M / 1000.0
plt.axhline(y=cell_size_km, color='r', linestyle='--',
            label=f'Grid Cell Boundary (Alt) ({cell_size_km:.1f} km)')
plt.axvline(x=cell_size_km, color='r', linestyle='--',
            label=f'Grid Cell Boundary (Range) ({cell_size_km:.1f} km)')

plt.title('Particle Hop Verification')
plt.xlabel('Horizontal Range (km)')
plt.ylabel('Altitude from Surface (km)')
plt.legend()
plt.grid(True)

# Set axis limits to be equal for a proportional view
max_dim = max(max(plot_x, default=0), max(plot_y, default=0), cell_size_km) * 1.1
if max_dim > 0:
    plt.xlim(0, max_dim)
    plt.ylim(0, max_dim)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()