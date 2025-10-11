import numpy as np
import matplotlib.pyplot as plt

# --- Lambertian (Cosine) Distribution Sampling ---
N_SAMPLES = 100000

# Generate two sets of uniform random numbers
u1 = np.random.random(N_SAMPLES)
u2 = np.random.random(N_SAMPLES)

# Sample cos(θ) according to the Lambertian distribution
sin_theta = np.sqrt(u2)
cos_theta_lambert = np.sqrt(1-sin_theta**2)

# Convert the sampled cos(θ) values to θ in degrees
theta_lambert_rad = np.arccos(cos_theta_lambert)
theta_lambert_deg = np.degrees(theta_lambert_rad)

# Sample the azimuthal angle φ and convert to degrees
phi_rad_lambert = 2 * np.pi * u1
phi_lambert_deg = np.degrees(phi_rad_lambert) - 180.0

# --- Plotting the Results ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
fig.suptitle('Lambertian (Cosine) Distribution', fontsize=16)

# 1. Histogram of θ (Zenith Angle)
ax1.hist(theta_lambert_deg, bins=50, density=True, range=(0, 90), label='Sampled Data')

# Theoretical PDF for θ is P(θ) = sin(2θ)
x_theory_deg = np.linspace(0, 90, 100)
x_theory_rad = np.radians(x_theory_deg)
y_theory_theta = np.sin(2 * x_theory_rad)
ax1.set_xlabel('Zenith Angle θ [degrees]', fontsize="12")
ax1.set_ylabel('Probability Density', fontsize="12")
ax1.grid(True)
ax1.legend()

# 2. Histogram of the azimuthal angle φ (unchanged)
ax2.hist(phi_lambert_deg, bins=50, density=True, range=(-180, 180))
ax2.set_xlabel('Azimuthal Angle φ [degrees]', fontsize="12")
ax2.set_ylabel('Probability Density', fontsize="12")
ax2.grid(True)

plt.show()