import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 物理定数と軌道パラメータ
# ==========================================
A_AU = 0.387098          # 軌道長半径 [AU]
E = 0.205630             # 離心率
T_ORBIT = 87.969         # 公転周期 [days]
T_SPIN = 58.6462         # 自転周期 [days]

# 角速度計算 (rad/day)
n_orbit = 2 * np.pi / T_ORBIT
w_spin = 2 * np.pi / T_SPIN

# TAA (真近点角) 配列の作成
taa_deg = np.linspace(0, 360, 500)
taa_rad = np.radians(taa_deg)

# ==========================================
# 2. 物理量の計算 (本コードと同じ解析解)
# ==========================================
# 距離 r (AU)
r_au = A_AU * (1 - E**2) / (1 + E * np.cos(taa_rad))

# [A] 太陽の見かけの移動速度
h = n_orbit * A_AU**2 * np.sqrt(1 - E**2)
dtheta_dt = h / r_au**2
w_app_deg_day = np.degrees(w_spin - dtheta_dt)

# [B] 光電離速度 & 太陽放射圧の変化率 (遠日点基準)
r_aphelion = A_AU * (1 + E)
flux_relative_to_aphelion = (r_aphelion / r_au)**2

# ==========================================
# 3. スライド用プロットの共通設定
# ==========================================
plt.rcParams.update({
    'font.size': 16,             # スライド用に少し大きめ
    'axes.linewidth': 2.0,
    'lines.linewidth': 3.0,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0
})

# ---------------------------------------------------------
# 図1: 太陽の見かけの移動速度 (Apparent Solar Velocity)
# ---------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(taa_deg, w_app_deg_day, color='crimson', label="Apparent Solar Velocity")
ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7) # 逆行の境界線
ax1.axvline(180, color='gray', linestyle=':', linewidth=1.5)

ax1.set_xlabel("True Anomaly Angle (TAA) [deg]", fontweight='bold')
ax1.set_ylabel("Apparent Solar Velocity\n[deg / day]", fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 360)
ax1.set_xticks(np.arange(0, 361, 60))

# 凡例を右上に配置
ax1.legend(loc='upper right') 

fig1.tight_layout()
fig1.savefig("apparent_velocity.png", dpi=300, bbox_inches='tight')
print("Saved: apparent_velocity.png")

# ---------------------------------------------------------
# 図2: 光電離速度 & 太陽放射圧の相対強度 (Photo-ionization & SRP)
# ---------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.plot(taa_deg, flux_relative_to_aphelion, color='dodgerblue', label="Photo-ionization")
ax2.axvline(180, color='gray', linestyle=':', linewidth=1.5)

ax2.set_xlabel("True Anomaly Angle (TAA) [deg]", fontweight='bold')
ax2.set_ylabel("Relative Intensity\n(1.0 at Aphelion)", fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 360)
ax2.set_xticks(np.arange(0, 361, 60))

ax2.legend(loc='upper right')

fig2.tight_layout()
fig2.savefig("photoionization_srp.png", dpi=300, bbox_inches='tight')
print("Saved: photoionization_srp.png")

plt.show()