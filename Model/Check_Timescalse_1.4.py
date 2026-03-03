import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ==========================================
# 1. パラメータ設定
# ==========================================
# 物理定数
KB_EV = 8.617e-5
NU = 1.0e13

# 軌道パラメータ (水星)
MERCURY_A_AU = 0.387098
MERCURY_E = 0.205630

# 温度モデルパラメータ
T_BASE = 100.0
T_AMP = 600.0
SCALING_REF_AU = 0.306

# 解析対象
U_TARGET = 1.6
DT_STEP = 100.0
LATITUDES = [0, 30, 45, 60, 70, 80]  # Fig1, 2用のリスト


# ==========================================
# 2. 計算関数
# ==========================================
def get_dist_au(taa_deg):
    rad = np.deg2rad(taa_deg)
    return MERCURY_A_AU * (1 - MERCURY_E ** 2) / (1 + MERCURY_E * np.cos(rad))


def get_noon_temp(r_au, lat_deg):
    scaling = np.sqrt(SCALING_REF_AU / r_au)
    t_ss = T_BASE + T_AMP * scaling
    lat_rad = np.deg2rad(lat_deg)
    # cosが負になる(夜側)場合のガード
    t_local = t_ss * (np.maximum(np.cos(lat_rad), 0.0) ** 0.25)
    return np.maximum(t_local, T_BASE)


def get_residence_time(temp, u_ev):
    if temp <= 0: return 1e50
    exponent = u_ev / (KB_EV * temp)
    if exponent > 100: return 1e50
    return (1.0 / NU) * np.exp(exponent)


# 臨界温度 (tau = dt)
critical_temp = U_TARGET / (KB_EV * np.log(DT_STEP * NU))

# ==========================================
# 3. メイン計算 & プロット (3ウィンドウ)
# ==========================================
taa_list = np.arange(0, 360, 2)
r_au_list = [get_dist_au(taa) for taa in taa_list]
colors = cm.plasma(np.linspace(0, 0.9, len(LATITUDES)))

# --- ウィンドウ設定 ---
# Figure 1: 滞留時間
fig1, ax1 = plt.subplots(figsize=(8, 5))
fig1.canvas.manager.set_window_title('Figure 1: Residence Time')

# Figure 2: 表面温度 (全緯度)
fig2, ax2 = plt.subplots(figsize=(8, 5))
fig2.canvas.manager.set_window_title('Figure 2: Surface Temperature (All)')

# Figure 3: 特定緯度の温度比較 (Max, Lat50, Min)
fig3, ax3 = plt.subplots(figsize=(8, 5))
fig3.canvas.manager.set_window_title('Figure 3: Temperature Comparison (Max/50deg/Min)')


# --- 計算ループ (Fig 1 & 2) ---
for lat, color in zip(LATITUDES, colors):
    tau_list = []
    temp_list_plot = []

    for r in r_au_list:
        t = get_noon_temp(r, lat)
        tau = get_residence_time(t, U_TARGET)

        tau_list.append(tau)
        temp_list_plot.append(t)

    # プロット (ax1: 時間, ax2: 温度)
    ax1.plot(taa_list, tau_list, color=color, linewidth=2, label=f'Lat = {lat}°')
    ax2.plot(taa_list, temp_list_plot, color=color, linewidth=2, label=f'Lat = {lat}°')


# --- 追加計算 (Fig 3) ---
# 要件: 最高温度(Lat 0), Lat 50, 最低温度(Lat 85※)
# ※Lat 90だとCos90=0で常にT_BASE(100K)になるため、変化が見えるよう85度としています
target_lats = [0, 50, 85]
target_labels = ['Max Temp (Lat 0°)', 'Lat 50°', 'Min Temp (Lat 85°)']
target_colors = ['#FF4500', '#32CD32', '#1E90FF'] # 赤, 緑, 青

for lat, label, color in zip(target_lats, target_labels, target_colors):
    temps = [get_noon_temp(r, lat) for r in r_au_list]
    ax3.plot(taa_list, temps, color=color, linewidth=2.5, label=label)


# ==========================================
# 4. グラフ装飾
# ==========================================

# --- Figure 1: 滞留時間 ---
ax1.set_facecolor('#f9f9f9')
ax1.grid(True, which='both', linestyle='--', alpha=0.6)
ax1.axhline(y=DT_STEP, color='black', linestyle=':', linewidth=2, label=f'Step ({DT_STEP}s)')
ax1.set_yscale('log')
ax1.set_xlabel('True Anomaly (TAA) [deg]', fontsize=11)
ax1.set_ylabel('TimeScale [s]', fontsize=11)
ax1.set_ylim(1e-4, 1e10)
ax1.set_xlim(0, 360)
ax1.legend(loc='upper right', fontsize=9)

# --- Figure 2: 全緯度温度 ---
ax2.set_facecolor('#f9f9f9')
ax2.grid(True, which='both', linestyle='--', alpha=0.6)
ax2.axhline(y=critical_temp, color='red', linestyle='-.', linewidth=2, label=f'Critical (~{int(critical_temp)}K)')
ax2.text(180, critical_temp + 10, 'Desorption', color='red', fontsize=9, ha='center')
ax2.text(180, critical_temp - 30, 'Adsorption', color='blue', fontsize=9, ha='center')
ax2.set_xlabel('True Anomaly (TAA) [deg]', fontsize=11)
ax2.set_ylabel('Surface Temp [K]', fontsize=11)
ax2.set_xlim(0, 360)
ax2.set_ylim(100, 800)
ax2.legend(loc='upper right', fontsize=9)

# --- Figure 3: 比較プロット (新規) ---
ax3.set_facecolor('#f9f9f9')
ax3.grid(True, which='both', linestyle='--', alpha=0.6)
ax3.set_xlabel('True Anomaly (TAA) [deg]', fontsize=11)
ax3.set_ylabel('Surface Temp [K]', fontsize=11)
ax3.set_title('Temperature Comparison: 0°, 50°, 85°', fontsize=12)
ax3.set_xlim(0, 360)
ax3.set_ylim(100, 800)

# Lat 50のポイントを強調
ax3.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)

# 臨界温度のラインも一応引いておく（参考用）
ax3.axhline(y=critical_temp, color='gray', linestyle='-.', alpha=0.5, label='Critical Temp')

print(f"Calculated Critical Temperature: {critical_temp:.2f} K")
plt.show()