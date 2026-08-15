import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
from matplotlib.ticker import MultipleLocator

e = 0.20563

# TAAの配列: 0が近日点、180が遠日点
taa_deg = np.linspace(0, 360, 1000)
taa_rad = np.radians(taa_deg)

# ==========================================
# 軌道力学の計算 (3:2共鳴)
# ==========================================
E_rad = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(taa_rad / 2))
E_rad = np.unwrap(E_rad)
M_rad = E_rad - e * np.sin(E_rad)

# 太陽直下点(SSP)の経度
ssp_lon_rad = taa_rad - 1.5 * M_rad

# ==========================================
# プロット
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

# TAA=0時点でのDawn(夜明け)ターミネーター地点の経度
lon_dawn_deg = -90
lon_dawn_rad = np.radians(lon_dawn_deg)

# 太陽天頂角(SZA) = 表面経度 - SSP経度
sza_dawn_rad = lon_dawn_rad - ssp_lon_rad
eff_cos_dawn = np.maximum(0, np.cos(sza_dawn_rad))

ax.plot(taa_deg, eff_cos_dawn, lw=3, color='blue', label='TAA=0でのDawnターミネーター地点 (経度 -90°)')

# 注目TAAの強調
ax.axvline(162.5, color='green', ls='--', alpha=0.8, label='TD単独ピーク (162.5°)')
ax.axvline(180, color='gray', ls=':', alpha=0.8, label='遠日点 (180°)')

ax.set_title('Dawnターミネーター地点の eff_cos の推移\n(夜明けを迎えたはずが、逆行により再び夜側に押し戻される様子)', fontsize=14)
ax.set_xlabel('TAA [deg] (0: 近日点, 180: 遠日点)', fontsize=12)
ax.set_ylabel('eff_cos', fontsize=12)
ax.set_xlim(0, 360)
ax.set_ylim(0, 1.05)

# 横軸を60度刻みに設定
ax.xaxis.set_major_locator(MultipleLocator(60))

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.show()