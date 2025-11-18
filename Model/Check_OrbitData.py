import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- Settings ---
ORBIT_V6_FILE = 'orbit2025_v6.txt' # (v6生成スクリプトの出力ファイル名に合わせる)
ROTATION_PERIOD_SEC = 58.646 * 24 * 3600
MERCURY_YEAR_DAYS = 87.969
# --- End of Settings ---

print(f"Loading '{ORBIT_V6_FILE}' to visualize orbital and rotational angles...")

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

# 2. Extract data
# X軸: 時間 (秒 -> 日)
time_sec = orbit_data[:, 2]
time_days = time_sec / (24 * 3600)

# Y軸1: TAA (公転の角度)
taa_deg = orbit_data[:, 0]

# Y軸2: 自転の角度 (計算し直すことで、ファイルに依存せず正確に比較)
rotation_angle_deg = (time_sec / ROTATION_PERIOD_SEC) * 360.0

print("Calculation complete. Displaying plot.")

# 3. Plot the graph
plt.figure(figsize=(12, 7))
plt.title("Mercury's Orbital vs Rotational Angle Over Time", fontsize=16)

# Plot TAA (Orbital Angle)
plt.plot(
    time_days,
    taa_deg,
    label='Orbital Angle (TAA) - Pericenter reference',
    color='blue',
    linewidth=3,
    linestyle='-'
)

# Plot Rotational Angle
plt.plot(
    time_days,
    rotation_angle_deg,
    label='Rotational Angle (relative to fixed stars)',
    color='red',
    linewidth=3,
    linestyle='--'
)

# Graph decorations
plt.xlabel("Time [days]", fontsize=14)
plt.ylabel("Angle [degrees]", fontsize=14)
plt.xticks(np.arange(0, MERCURY_YEAR_DAYS * 2 + 1, MERCURY_YEAR_DAYS / 2)) # 0.5年ごとに
plt.yticks(np.arange(0, 360 * 3 + 1, 90)) # 90度ごとに
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12, loc='upper left')

# 1公転の期間を縦線で示す
plt.axvline(x=MERCURY_YEAR_DAYS, color='gray', linestyle=':', label='1 Mercury Year')
plt.axvline(x=MERCURY_YEAR_DAYS * 2, color='gray', linestyle=':', label='2 Mercury Years')


plt.tight_layout()
plt.show()