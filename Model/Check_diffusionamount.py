import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 物理定数および軌道パラメータ
# ==========================================
KB_EV_CONST = 8.617333262e-5  # ボルツマン定数 [eV/K]
T_year = 87.969 * 24 * 3600   # 水星の公転周期 [秒]
a = 0.387098                  # 軌道長半径 [AU]
e = 0.205630                  # 離心率
r_p = a * (1 - e)             # 近日点距離 [AU]

# ==========================================
# 2. ユーザー指定パラメータ
# ==========================================
DIFF_REF_FLUX = 1.0e7 * (100.0 ** 2) 
DIFF_REF_TEMP = 700.0         # 基準温度 [K]
LABEL_SIZE = 16               # 軸ラベルの文字サイズ
TICK_SIZE = 14                # 目盛りの数字の文字サイズ
TITLE_SIZE = 18               # グラフタイトルの文字サイズ
xticks_deg = np.arange(0, 361, 60) # 60度刻みの目盛り

# 面積速度の2倍 (ケプラーの第2法則用)
h = 2 * np.pi * a**2 * np.sqrt(1 - e**2) / T_year

# ==========================================
# 3. 計算処理
# ==========================================
theta_rad = np.linspace(0, 2 * np.pi, 500)
theta_deg = np.degrees(theta_rad)

def get_flux_data(e_a):
    pre_factor = DIFF_REF_FLUX / np.exp(-e_a / (KB_EV_CONST * DIFF_REF_TEMP))
    # T = T_p * sqrt(r_p / r)
    r_arr = a * (1 - e**2) / (1 + e * np.cos(theta_rad))
    temp_arr = DIFF_REF_TEMP * np.sqrt(r_p / r_arr)
    flux_m2 = pre_factor * np.exp(-e_a / (KB_EV_CONST * temp_arr))
    flux_cm2 = flux_m2 / (100.0 ** 2)
    
    # 時間積分の計算
    dt_dtheta = r_arr**2 / h
    dt_arr = dt_dtheta * (theta_rad[1] - theta_rad[0])
    cumulative = np.cumsum(flux_cm2 * dt_arr)
    return flux_cm2, cumulative

# 各データの取得
flux_040, cum_040 = get_flux_data(0.40)
flux_075, cum_075 = get_flux_data(0.90)

# 一定モデル (1e7)
flux_const = np.full_like(theta_rad, 1.0e7)
dt_dtheta_const = (a * (1 - e**2) / (1 + e * np.cos(theta_rad)))**2 / h
cum_const = np.cumsum(flux_const * dt_dtheta_const * (theta_rad[1] - theta_rad[0]))

# ==========================================
# 4. ウィンドウ1: 瞬間フラックス
# ==========================================
plt.figure(1, figsize=(8, 6))
plt.plot(theta_deg, flux_040, label='U = 0.40 eV', color='blue', linewidth=2)
plt.plot(theta_deg, flux_075, label='U = 0.90 eV', color='green', linewidth=2)
plt.plot(theta_deg, flux_const, label='Constant (1e7)', color='red', linestyle='--', linewidth=2)

plt.yscale('log')
plt.xlabel('True Anomaly [deg]', fontsize=LABEL_SIZE)
plt.ylabel('Flux [atoms / cm$^2$ / s]', fontsize=LABEL_SIZE)
#plt.title('Instantaneous Diffusion Flux', fontsize=TITLE_SIZE)
plt.xticks(xticks_deg, fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
plt.xlim(0, 360)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

# ==========================================
# 5. ウィンドウ2: 累積放出量
# ==========================================
plt.figure(2, figsize=(8, 6))
plt.plot(theta_deg, cum_040, label='U = 0.40 eV', color='blue', linewidth=2)
plt.plot(theta_deg, cum_075, label='U = 0.90 eV', color='green', linewidth=2)
plt.plot(theta_deg, cum_const, label='Constant (1e7)', color='red', linestyle='--', linewidth=2)

plt.xlabel('True Anomaly [deg]', fontsize=LABEL_SIZE)
plt.ylabel('Cumulative Released Na [atoms / cm$^2$]', fontsize=LABEL_SIZE)
#plt.title('Cumulative Sodium Release', fontsize=TITLE_SIZE)
plt.xticks(xticks_deg, fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
plt.xlim(0, 360)
plt.grid(True, ls="--", alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# 全てのウィンドウを表示
plt.show()