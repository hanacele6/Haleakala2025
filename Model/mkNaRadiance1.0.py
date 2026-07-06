import numpy as np
import matplotlib.pyplot as plt
import glob
import re

# 物理定数・グリッド設定 (シミュレーション設定と合わせる)
RM = 2.440e6
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0
gmin = -GRID_MAX_RM * RM
gmax = GRID_MAX_RM * RM
dx = (gmax - gmin) / GRID_RESOLUTION

# 抽出したい高度 [m]
ALTITUDE_TARGET = 300e3

# 座標からグリッドインデックスを取得する関数
def get_index(val, gmin, dx, max_idx):
    idx = int((val - gmin) / dx)
    return max(0, min(idx, max_idx - 1))

# --- データ読み込みと抽出 ---
file_list = sorted(glob.glob("SimulationResult_202606/ParabolicHop_72x36_NoEq_DT100_0611_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_RMtest4/density_grid_t*_taa*.npy"))

taa_list = []
radiance_lt12 = []
radiance_lt06 = []
radiance_lt18 = []

for file_path in file_list:
    # ファイル名からTAAを抽出 (例: ..._taa065.npy)
    match = re.search(r'taa(\d{3})\.npy', file_path)
    if not match: continue
    taa = float(match.group(1))
    
    # 3D密度グリッドの読み込み [atoms / m^3]
    dgrid = np.load(file_path)
    
    # 1. Z軸方向(南から北)へ積分し、カラム密度マップを作成 [atoms / m^2]
    column_density_map = np.sum(dgrid, axis=2) * dx
    
    # 2. 各LTの接線高度300kmの座標を計算
    # 太陽方向が +x (LT12) と仮定
    # LT12 (Noon) : X = RM + 300km, Y = 0
    idx_x_lt12 = get_index(RM + ALTITUDE_TARGET, gmin, dx, GRID_RESOLUTION)
    idx_y_lt12 = get_index(0.0, gmin, dx, GRID_RESOLUTION)
    
    # LT06 (Dawn) : X = 0, Y = RM + 300km
    idx_x_lt06 = get_index(0.0, gmin, dx, GRID_RESOLUTION)
    idx_y_lt06 = get_index(RM + ALTITUDE_TARGET, gmin, dx, GRID_RESOLUTION)
    
    # LT18 (Dusk) : X = 0, Y = -(RM + 300km)
    idx_x_lt18 = get_index(0.0, gmin, dx, GRID_RESOLUTION)
    idx_y_lt18 = get_index(-(RM + ALTITUDE_TARGET), gmin, dx, GRID_RESOLUTION)
    
    # 3. カラム密度の抽出
    cd_lt12 = column_density_map[idx_x_lt12, idx_y_lt12]
    cd_lt06 = column_density_map[idx_x_lt06, idx_y_lt06]
    cd_lt18 = column_density_map[idx_x_lt18, idx_y_lt18]
    
    # 4. g-factorを掛けて輝度に変換
    # ※厳密には各TAAで再計算したg-factorを用いるのが理想ですが、ここでは暫定値を適用します。
    g_factor = 1.5 
    rad_lt12_MR = (cd_lt12 * g_factor) / 1e16 # [MR]に変換
    rad_lt06_MR = (cd_lt06 * g_factor) / 1e16
    rad_lt18_MR = (cd_lt18 * g_factor) / 1e16
    
    taa_list.append(taa)
    radiance_lt12.append(rad_lt12_MR)
    radiance_lt06.append(rad_lt06_MR)
    radiance_lt18.append(rad_lt18_MR)

# --- プロット (論文のFigure 8ライクな図) ---
# TAAでソート
sort_idx = np.argsort(taa_list)
taa_list = np.array(taa_list)[sort_idx]
radiance_lt12 = np.array(radiance_lt12)[sort_idx]
radiance_lt06 = np.array(radiance_lt06)[sort_idx]
radiance_lt18 = np.array(radiance_lt18)[sort_idx]

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(taa_list, radiance_lt06, 'k.-')
plt.title('LT06 (Dawn) at 300km')
plt.xlabel('TAA (deg)')
plt.ylabel('Na Emission (MR)')
plt.xlim(0, 360)

plt.subplot(1, 3, 2)
plt.plot(taa_list, radiance_lt12, 'k.-')
plt.title('LT12 (Noon) at 300km')
plt.xlabel('TAA (deg)')
plt.xlim(0, 360)

plt.subplot(1, 3, 3)
plt.plot(taa_list, radiance_lt18, 'k.-')
plt.title('LT18 (Dusk) at 300km')
plt.xlabel('TAA (deg)')
plt.xlim(0, 360)

plt.tight_layout()
plt.show()