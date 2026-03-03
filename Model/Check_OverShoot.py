import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import re
import glob
import sys

# ==============================================================================
# 【設定】
# ==============================================================================
TARGET_DIR = r"./SimulationResult_202510/DynamicGrid72x36_4.0"
TARGET_TAA = 220  # 検証したいTAA (暴れているタイミング)
ORBIT_FILE_PATH = 'orbit2025_v6.txt'

# シミュレーション設定
DT = 500.0  # タイムステップ [s]
BINDING_ENERGY_EV = 1.5  # 結合エネルギー [eV]

# ==============================================================================
# 物理定数・関数
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'K_BOLTZMANN': 1.380649e-23,
    'EV_TO_JOULE': 1.602176634e-19,
}


def find_file_by_taa(directory, taa):
    pattern = os.path.join(directory, f"surface_density_*_taa{taa:03d}.npy")
    files = glob.glob(pattern)
    if not files:
        pattern_alt = os.path.join(directory, f"surface_density_*_taa{taa}.npy")
        files = glob.glob(pattern_alt)
    if not files:
        print(f"エラー: TAA={taa} のファイルが見つかりません。")
        sys.exit(1)
    files.sort()
    return files[-1]


def get_orbital_info(filepath, orbit_path):
    match = re.search(r'_t(\d+)_taa(\d+)\.npy$', os.path.basename(filepath))
    time_h = int(match.group(1))
    relative_time_sec = float(time_h) * 3600.0

    orbit_data = np.loadtxt(orbit_path)
    idx_peri = np.argmin(np.abs(orbit_data[:, 0]))
    t_start = orbit_data[idx_peri, 2]

    current_t_sec = t_start + relative_time_sec
    MERCURY_YEAR_SEC = 87.97 * 24 * 3600
    current_time_in_orbit = current_t_sec % MERCURY_YEAR_SEC

    time_col = orbit_data[:, 2]
    au = np.interp(current_time_in_orbit, time_col, orbit_data[:, 1])
    sub_lon_deg = np.interp(current_time_in_orbit, time_col, orbit_data[:, 5])

    sub_lon_rad = np.deg2rad(sub_lon_deg)
    sub_lon_rad = (sub_lon_rad + np.pi) % (2 * np.pi) - np.pi

    return au, sub_lon_rad, time_h


def calculate_rate(temp_k, u_ev):
    if temp_k <= 10: return 0.0
    exponent = -(u_ev * PHYSICAL_CONSTANTS['EV_TO_JOULE']) / (PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k)
    if exponent < -700: return 0.0
    return 1e13 * np.exp(exponent)


def calculate_temp(lon_fixed, lat, au, sub_lon):
    cos_theta = np.cos(lat) * np.cos(lon_fixed - sub_lon)
    if cos_theta <= 0: return 100.0
    T0 = np.interp(au, [0.307, 0.467], [600.0, 475.0])
    return T0 + 100.0 * (cos_theta ** 0.25)


# ==============================================================================
# メイン検証処理
# ==============================================================================
def verify_amount_and_rate():
    # 1. データ準備
    target_file = find_file_by_taa(TARGET_DIR, TARGET_TAA)
    grid_density = np.load(target_file)  # [atoms/m^2] (在庫量)

    N_LON, N_LAT = grid_density.shape
    AU, sub_lon, time_h = get_orbital_info(target_file, ORBIT_FILE_PATH)

    print(f"--- 検証レポート (TAA={TARGET_TAA}, Time={time_h}h) ---")

    # 2. 全グリッド計算
    overshoot_factor_grid = np.zeros((N_LON, N_LAT))  # Rate * dt
    demand_amount_grid = np.zeros((N_LON, N_LAT))  # 放出要求量 [atoms/m^2]

    lon_edges = np.linspace(-np.pi, np.pi, N_LON + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)

    # 最悪地点の記録用
    max_factor = 0.0
    worst_pos = (0, 0)
    worst_vals = {}  # {density, rate, factor, demand}

    for i in range(N_LON):
        for j in range(N_LAT):
            lon = (lon_edges[i] + lon_edges[i + 1]) / 2
            lat = (lat_edges[j] + lat_edges[j + 1]) / 2

            # 温度とレート
            T = calculate_temp(lon, lat, AU, sub_lon)
            rate = calculate_rate(T, BINDING_ENERGY_EV)

            # 現在の在庫量 (Amount)
            current_amount = grid_density[i, j]

            # 率 (Factor)
            factor = rate * DT
            overshoot_factor_grid[i, j] = factor

            # 要求量 (Demand Amount) = 在庫 * 率
            demand = current_amount * factor
            demand_amount_grid[i, j] = demand

            # 最大オーバーシュート地点を探す
            if factor > max_factor:
                max_factor = factor
                worst_pos = (i, j)
                worst_vals = {
                    'temp': T,
                    'density': current_amount,
                    'rate': rate,
                    'factor': factor,
                    'demand': demand
                }

    # 3. 詳細レポート (量を見る！)
    print(f"\n【最悪のオーバーシュート地点】 インデックス {worst_pos}")
    print(f"  表面温度:      {worst_vals['temp']:.1f} K")
    print(f"  放出レート:    {worst_vals['rate']:.4e} /s")
    print(f"  タイムステップ: {DT} s")
    print("-" * 40)
    print(f"★ オーバーシュート率 (Factor): {worst_vals['factor']:.2f} 倍")
    print("-" * 40)
    print(f"  [量] 現在の在庫量:   {worst_vals['density']:.3e} atoms/m^2")
    print(f"  [量] 放出要求量:     {worst_vals['demand']:.3e} atoms/m^2")
    print("-" * 40)

    if worst_vals['factor'] > 1.0:
        excess = worst_vals['demand'] - worst_vals['density']
        print(f"判定: 在庫が足りません！ {excess:.3e} atoms/m^2 分の計算が破綻しています。")
        print("      -> min()関数により、在庫は強制的に 0 になりました。")
    else:
        print("判定: 正常範囲内です。")

    # 4. 可視化 (率と量を並べて表示)
    # 太陽中心にシフト
    shift = N_LON // 2 - int(np.round((np.rad2deg(sub_lon) + 180) / (360 / N_LON))) % N_LON

    plot_factor = np.roll(overshoot_factor_grid, shift, axis=0).T
    plot_density = np.roll(grid_density, shift, axis=0).T

    # 0や負の値をマスク
    plot_density = np.maximum(plot_density, 1e10)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左：オーバーシュート率 (Rate * dt)
    im1 = axes[0].pcolormesh(np.linspace(-180, 180, N_LON + 1), np.linspace(-90, 90, N_LAT + 1),
                             plot_factor, cmap='RdYlBu_r', vmin=0, vmax=2.0)
    plt.colorbar(im1, ax=axes[0], label='Overshoot Factor (Rate * dt)')
    axes[0].set_title(f"Overshoot Factor ( > 1.0 is BAD)\nMax: {max_factor:.2f}")
    axes[0].set_xlabel("Longitude (Sun-Fixed)")
    axes[0].set_ylabel("Latitude")

    # 右：表面密度 (在庫量)
    im2 = axes[1].pcolormesh(np.linspace(-180, 180, N_LON + 1), np.linspace(-90, 90, N_LAT + 1),
                             plot_density, cmap='inferno', norm=LogNorm(vmin=1e10, vmax=np.max(plot_density)))
    plt.colorbar(im2, ax=axes[1], label='Surface Density [atoms/m^2]')
    axes[1].set_title(f"Surface Density (Inventory Amount)\nLook at the 'depleted' areas")
    axes[1].set_xlabel("Longitude (Sun-Fixed)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    verify_amount_and_rate()