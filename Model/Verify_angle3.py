import numpy as np
import matplotlib.pyplot as plt

# --- シミュレーションのパラメータ ---
N_SAMPLES = 500000  # サンプル数を多くすると結果が理論値に近づく

# --- 1. ランバート分布に従うサンプルを生成 ---
u2 = np.random.random(N_SAMPLES)
sin_theta = np.sqrt(u2)
cos_theta = np.sqrt(1 - sin_theta**2)
theta_rad = np.arccos(cos_theta)

# ==============================================================================
# グラフ1: P(θ) = sin(2θ) の検証
# ==============================================================================
print("--- Displaying PDF of θ ---")
fig1, ax1 = plt.subplots(figsize=(8, 6))
fig1.suptitle('Verification of P(θ) = sin(2θ)', fontsize=16)

# P(θ) = sin(2θ) の検証
ax1.hist(theta_rad, bins=50, density=True, range=(0, np.pi/2), label='Sampled Data')
x_theory_rad = np.linspace(0, np.pi/2, 100)
y_theory_rad = np.sin(2 * x_theory_rad)
ax1.plot(x_theory_rad, y_theory_rad, 'r-', lw=2, label='Theory: P(θ) = sin(2θ)')
ax1.set_title('PDF of θ (in Radians)', fontsize=14)
ax1.set_xlabel('Zenith Angle θ [radians]', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.grid(True)
ax1.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ==============================================================================
# グラフ2: P(cosθ) = 2cosθ の検証
# ==============================================================================
print("\n--- Displaying PDF of cos(θ) ---")
fig2, ax2 = plt.subplots(figsize=(8, 6))
fig2.suptitle('Verification of P(cosθ) = 2cosθ', fontsize=16)

# P(cosθ) = 2cosθ の検証
ax2.hist(cos_theta, bins=50, density=True, range=(0, 1), label='Sampled Data')
x_theory_cos = np.linspace(0, 1, 100)
y_theory_cos = 2 * x_theory_cos
ax2.plot(x_theory_cos, y_theory_cos, 'r-', lw=2, label='Theory: P(cosθ) = 2cosθ')
ax2.set_title('PDF of cos(θ)', fontsize=14)
ax2.set_xlabel('cos(θ)', fontsize=12)
ax2.set_ylabel('Probability Density', fontsize=12)
ax2.grid(True)
ax2.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ==============================================================================
# Part 2: 数値積分による確認 (変更なし)
# ==============================================================================
print("\n--- Part 2: Numerical Verification for p(Ω) = cosθ/π ---")

f_values = 1.0 / cos_theta
monte_carlo_result = np.mean(f_values)

print(f"Theoretical expectation value of (1/cosθ): 2.0")
print(f"Monte Carlo result from {N_SAMPLES} samples: {monte_carlo_result:.6f}")

if np.isclose(monte_carlo_result, 2.0, atol=0.01):
    print("✅ The result is very close to the theoretical value.")
else:
    print("❌ The result is not close to the theoretical value. Check the code or increase N_SAMPLES.")