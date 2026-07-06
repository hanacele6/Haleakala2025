import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==========================================
# 共通の物理定数とパラメータ
# ==========================================
E_d_fixed = 1.85 # 単一エネルギー用の固定値 (eV)
E_d_mean = 1.9  # ガウス分布用の最頻値/平均値 (eV)
nu = 1e13        # 頻度因子 (s^-1)
k_B = 8.617e-5   # ボルツマン定数 (eV/K)

# 微分方程式: 被覆率(theta)の温度に対する変化率 d(theta)/dT (単一エネルギー用)
def dtheta_dT_single(theta, T, beta):
    return -(nu / beta) * theta * np.exp(-E_d_fixed / (k_B * T))

# ==========================================
# Figure 1: 昇温速度 (beta) の違いによるピークシフト
# ==========================================
T_range_fig1 = np.linspace(400, 900, 1000)
theta0 = 1.0  # 初期被覆率
betas = [0.1, 1.0, 7.0, 49.0]

plt.figure(figsize=(10, 6))

for beta in betas:
    theta = odeint(dtheta_dT_single, theta0, T_range_fig1, args=(beta,))
    rate = nu * theta[:, 0] * np.exp(-E_d_fixed / (k_B * T_range_fig1))
    
    # 高さを1に揃えて(正規化して)プロット
    plt.plot(T_range_fig1, rate / np.max(rate), label=f'β = {beta} K/s')

plt.xlabel('Temperature T (K)', fontsize=14)
plt.ylabel('Normalized Desorption Rate', fontsize=14)
plt.title(f'Fig 1: Simulated TPD Curves (Fixed Ed = {E_d_fixed} eV)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)


# ==========================================
# Figure 2: 単一エネルギー vs ガウス分布 (Yakshinskiy Fig.16 比較用)
# ==========================================
sigma = 0.25     # 論文の図16の幅(約1.4〜2.7eV)に近づけるための標準偏差
beta_fixed = 7.0 # 論文の実験値に固定
T_range_fig2 = np.linspace(400, 1000, 1000)

# 1. ガウス分布の計算
E_d_arr = np.linspace(1.2, 3.0, 1000) # 1000個のエネルギービンに分割
weights = np.exp(-0.5 * ((E_d_arr - E_d_mean) / sigma)**2)
weights /= np.sum(weights) # 初期被覆率の合計が1になるように重み付け

# 複数エネルギー同時の微分方程式
def dtheta_dT_vec(theta_vec, T, beta, E_arr):
    return -(nu / beta) * theta_vec * np.exp(-E_arr / (k_B * T))

theta_sol = odeint(dtheta_dT_vec, weights, T_range_fig2, args=(beta_fixed, E_d_arr), atol=1e-12, rtol=1e-10)

# 各温度において全ビンの脱離速度を合計
rate_tot = np.zeros(len(T_range_fig2))
for i in range(len(T_range_fig2)):
    rate_tot[i] = np.sum(nu * theta_sol[i, :] * np.exp(-E_d_arr / (k_B * T_range_fig2[i])))

# 2. 比較用の単一エネルギーの計算
theta_single = odeint(dtheta_dT_single, theta0, T_range_fig2, args=(beta_fixed,))
rate_single = nu * theta_single[:, 0] * np.exp(-E_d_fixed / (k_B * T_range_fig2))

# 3. 形状比較のための規格化 (論文の図16の縦軸に合わせてピークを約14.8に揃える)
rate_single_norm = (rate_single / np.max(rate_single)) * 14.8
rate_tot_norm = (rate_tot / np.max(rate_tot)) * 14.8

plt.figure(figsize=(10, 6))

# 単一エネルギー (細い理論線)
plt.plot(T_range_fig2, rate_single_norm, label=f'Single Ed = {E_d_fixed} eV (Theory)', linestyle='--', color='gray', linewidth=2)
# ガウス分布エネルギー (太い実験データ相当)
plt.plot(T_range_fig2, rate_tot_norm, label=f'Gaussian Ed (mean={E_d_mean}eV, σ={sigma}eV)', color='blue', linewidth=3)

plt.xlabel('Temperature T (K)', fontsize=14)
plt.ylabel('TPD Signal (arb. units)', fontsize=14)
plt.title('Fig 2: Shape Comparison (Normalized to Yakshinskiy Fig. 16)', fontsize=16)
plt.xlim(400, 950)
plt.ylim(0, 16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# ==========================================
# 描画の実行 (2つのウィンドウを同時に開く)
# ==========================================
plt.show()