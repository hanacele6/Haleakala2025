import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate  # ★ 積分ライブラリをインポート

# --- Parameters ---
U_eV = 0.27  #
E_max_eV = 5.0  #
N_samples = 500000

# --- 1. Perform Rejection Sampling (Same as before) ---

def thompson_sigmund_shape(E, U):
    return E / (E + U)**3  #

E_peak = U_eV / 2.0
f_max = thompson_sigmund_shape(E_peak, U_eV)  #

E_try = np.random.uniform(0, E_max_eV, N_samples)  #
y_try = np.random.uniform(0, f_max, N_samples)     #
f_E_try = thompson_sigmund_shape(E_try, U_eV)  #

accepted_mask = (y_try <= f_E_try)  #
accepted_energies = E_try[accepted_mask]

print(f"Generated {len(accepted_energies)} accepted samples.")

# --- 2. Define the Normalized Theoretical PDF (0-inf normalized) ---

def thompson_sigmund_pdf(E, U):
    """
    Calculates the true PDF, normalized from 0 to Infinity.
    """
    # Integral of f(E) from 0 to inf is 1/(2*U)
    integral_0_inf = 1.0 / (2.0 * U)
    return thompson_sigmund_shape(E, U) / integral_0_inf
    # (This simplifies to 2*U * E / (E + U)**3)

# --- 3. ★★★ NEW STEP: Calculate the area of the PDF in our specific range ★★★ ---

# Calculate the actual area of the theoretical PDF from 0 to E_max_eV
area_under_curve, error = integrate.quad(
    lambda E: thompson_sigmund_pdf(E, U_eV), 0, E_max_eV
)

print(f"Theoretical PDF area from 0 to {E_max_eV} eV: {area_under_curve:.5f}")
print(f"(The histogram density is normalized to 1.0 in this same range)")

# --- 4. Plotting the Histogram and the *Corrected* PDF ---

plt.figure(figsize=(12, 7))

# 4a. Plot the histogram of accepted samples
# density=True normalizes this to Area = 1.0 *within the 0-5.0 eV range*
plt.hist(
    accepted_energies,
    bins=100,
    density=True,
    color='orange',
    alpha=0.7,
    label=f'Histogram of Accepted Samples'
)

# 4b. Plot the theoretical normalized PDF
E_curve = np.linspace(0, E_max_eV, 500)
pdf_curve_0_inf = thompson_sigmund_pdf(E_curve, U_eV)

# 4c. ★★★ CORRECTION ★★★
# Scale the theoretical curve so *its* area from 0-5.0eV also equals 1.0
# We do this by dividing the curve by its actual area (area_under_curve, ~0.9)
corrected_pdf_curve = pdf_curve_0_inf / area_under_curve

plt.plot(
    E_curve, corrected_pdf_curve,  # <-- Plot the corrected curve
    color='blue',
    linewidth=3,
    label=f'Theoretical PDF'
)

# 4d. Graph labels and titles
plt.title(f'Probability Density (U = {U_eV} eV)', fontsize=16)
plt.xlabel('Energy E [eV]', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, E_max_eV)

# 4e. Show the plot
print("\nDisplaying plot...")
plt.show()