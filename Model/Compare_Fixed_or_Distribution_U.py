import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import norm

# 物理定数と実験パラメータ
nu = 1e13            # 前指数因子 (s^-1)
beta = 7.0           # 昇温速度 (K/s)
kB = 8.617e-5        # ボルツマン定数 (eV/K)

# 温度範囲
T_start = 300
T_end = 1000
T_span = (T_start, T_end)
T_eval = np.linspace(T_start, T_end, 500)

# ==========================================
# 1. 単一エネルギー (Ed = 1.85 eV) の計算
# ==========================================
Ed_single = 1.85
def dtheta_dT_single(T, theta, Ed):
    return -(nu / beta) * theta * np.exp(-Ed / (kB * T))

res_single = solve_ivp(dtheta_dT_single, T_span, [1.0], t_eval=T_eval, args=(Ed_single,), method='Radau')
rate_single = (nu / beta) * res_single.y[0] * np.exp(-Ed_single / (kB * T_eval))

# ==========================================
# 2. ガウス分布 (平均 1.85 eV, 標準偏差 0.15 eV) の計算
# ==========================================
mu = 1.85
sigma = 0.15
num_bins = 100
Ed_bins = np.linspace(mu - 3*sigma, mu + 3*sigma, num_bins)

theta0_dist = norm.pdf(Ed_bins, mu, sigma)
theta0_dist /= np.sum(theta0_dist)

def dtheta_dT_dist(T, theta_array):
    return -(nu / beta) * theta_array * np.exp(-Ed_bins / (kB * T))

res_dist = solve_ivp(dtheta_dT_dist, T_span, theta0_dist, t_eval=T_eval, method='Radau')

rate_dist_total = np.zeros_like(T_eval)
for i in range(num_bins):
    rate_i = (nu / beta) * res_dist.y[i] * np.exp(-Ed_bins[i] / (kB * T_eval))
    rate_dist_total += rate_i


# 元の配列のすべての要素を、その配列の「最大値」で割り算する
rate_single_norm = rate_single / np.max(rate_single)
rate_dist_norm = rate_dist_total / np.max(rate_dist_total)


# ==========================================
# プロット
# ==========================================
plt.figure(figsize=(10, 6))

plt.plot(T_eval, rate_single_norm, label=f'Single Ed = {Ed_single} eV', color='black', linewidth=1.5)
plt.plot(T_eval, rate_dist_norm, label=f'Gaussian Ed (mean={mu} eV, std={sigma} eV)', color='red', linewidth=2)

plt.xlabel('Temperature (K)', fontsize=12)
plt.ylabel('Normalized Desorption Rate (Peak = 1) [arb. un.]', fontsize=12)
plt.title('Peak Height Normalized TPD Spectra (Width Comparison)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(400, 1000)

plt.tight_layout()
plt.show()