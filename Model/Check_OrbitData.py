import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- Settings ---
ORBIT_V6_FILE = 'orbit2025_v6.txt' # (v6生成スクリプトの出力ファイル名に合わせる)
ROTATION_PERIOD_SEC = 58.646 * 24 * 3600
# --- End of Settings ---

print(f"Loading '{ORBIT_V6_FILE}' to plot UNWRAPPED physical angle...")

# 1. Load v6 file
try:
    orbit_data = np.loadtxt(ORBIT_V6_FILE)
except Exception as e:
    print(f"Error: Failed to load '{ORBIT_V6_FILE}'.")
    print(e)
    sys.exit(1)

if orbit_data.shape[1] < 6:
    print(f"Error: '{ORBIT_V6_FILE}' must have 6 columns.")
    sys.exit(1)

# 2. Extract base data
taa_deg_axis = orbit_data[:, 0] # X-axis
time_sec = orbit_data[:, 2]     # Time (for calculation)

# 3. Re-calculate the "Correct" (v6) UNWRAPPED physical angle
#    (This is the calculation inside the v6 generator script)
rotation_angle_deg = (time_sec / ROTATION_PERIOD_SEC) * 360.0
subsolar_lon_v6_UNWRAPPED = taa_deg_axis - rotation_angle_deg

# 4. Re-calculate the "Incorrect" (v5) UNWRAPPED physical angle
old_wrong_lon_rad = (-2 * np.pi * time_sec / ROTATION_PERIOD_SEC) % (2 * np.pi)
# (v5はもともとラップされているので、比較のためアンラップする)
# (np.unwrapはラジアンで行う必要がある)
old_wrong_lon_v5_UNWRAPPED = np.rad2deg(np.unwrap(old_wrong_lon_rad, period=2*np.pi))
# -2*pi*t... は負の角度なので、0から-540度へと進む

print("Calculation complete. Displaying UNWRAPPED plot.")

# 5. Plot the graph
plt.figure(figsize=(12, 7))
plt.title("Comparison of UNWRAPPED Subsolar Longitude", fontsize=16)

# Plot 1 (v6 / Correct)
plt.plot(
    taa_deg_axis,
    subsolar_lon_v6_UNWRAPPED,
    label='v6 (Correct / Unwrapped): [Orbit(TAA)] - [Rotation(t)]',
    color='deepskyblue',
    linewidth=3
)

# Plot 2 (v5 / Incorrect)
plt.plot(
    taa_deg_axis,
    old_wrong_lon_v5_UNWRAPPED,
    label='v5 (Incorrect / Unwrapped): -[Rotation(t)] only',
    color='red',
    linestyle='--',
    linewidth=2
)

# Graph decorations
plt.xlabel("TAA (Orbital Angle) [degrees]", fontsize=14)
plt.ylabel("Calculated Subsolar Longitude [degrees] (Unwrapped)", fontsize=14)
plt.xticks(np.arange(0, 361, 45))
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12, loc='upper left')
plt.axvline(x=0, color='gray', linestyle=':', label='Perihelion (TAA=0)')
plt.axvline(x=180, color='gray', linestyle=':', label='Aphelion (TAA=180)')

plt.tight_layout()
plt.show()