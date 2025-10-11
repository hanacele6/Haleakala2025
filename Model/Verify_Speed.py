import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell

# --- 物理定数 ---
PHYSICAL_CONSTANTS = {
    'MASS_NA': 22.98976928 * 1.66054e-27,  # Na原子の質量 [kg]
    'K_BOLTZMANN': 1.380649e-23,  # ボルツマン定数 [J/K]
}

# --- 元のコードから必要な関数をコピー ---
def sample_maxwellian_speed(mass_kg, temp_k):
    sigma = np.sqrt(PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k / mass_kg)
    vx = np.random.normal(0, sigma)
    vy = np.random.normal(0, sigma)
    vz = np.random.normal(0, sigma)
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    return speed

# --- 検証のメイン処理 ---
N_SAMPLES = 100000
MASS = PHYSICAL_CONSTANTS['MASS_NA']
TEMPERATURE = 1500.0  # 元のコードで使われている温度

speeds = [sample_maxwellian_speed(MASS, TEMPERATURE) for _ in range(N_SAMPLES)]

# --- 結果をプロット ---
plt.figure(figsize=(8, 4))

# ヒストグラム (密度=Trueで正規化)
plt.hist(speeds, bins=100, density=True, label='Sampled Speeds')

# 理論的なマクスウェル分布を重ねてプロット
# scale parameter (a or sigma) for scipy's maxwell distribution
scale_param = np.sqrt(PHYSICAL_CONSTANTS['K_BOLTZMANN'] * TEMPERATURE / MASS)
v_range = np.linspace(0, max(speeds), 500)
pdf_theoretical = maxwell.pdf(v_range, scale=scale_param)

plt.plot(v_range, pdf_theoretical, 'r-', lw=2, label='Theoretical Maxwell PDF')


plt.xlabel('Speed [m/s]' , fontsize = "12")
plt.ylabel('Probability Density' ,  fontsize = "12")
plt.legend()
plt.grid(True)
plt.show()