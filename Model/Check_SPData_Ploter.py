# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import glob
import re

# ==============================================================================
# ユーザー設定
# ==============================================================================
# 1. 表面密度ファイル(.npy)が入っているフォルダを指定
#    例: r"C:\Users\hanac\...\SimulationResult_202511\SubCycle_72x36_3.0"
#INPUT_DIR = r"./SimulationResult_202511/SubCycle_144x72_4.0"
INPUT_DIR = r"./SimulationResult_202510/DynamicGrid72x36_10.0"

# 2. 診断したいTAAの範囲 (この範囲内のファイルだけ読み込みます)
TAA_RANGE_MIN = 100
TAA_RANGE_MAX = 140

# ==============================================================================
# 物理定数・設定
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'AU': 1.496e11,
    'K_BOLTZMANN': 1.380649e-23,
    'RM': 2.440e6,
    'EV_TO_JOULE': 1.602176634e-19,
}

N_LON_FIXED, N_LAT = 72, 36


# ==============================================================================
# 物理関数
# ==============================================================================
def calculate_surface_temperature_leblanc(lon_fixed_rad, lat_rad, AU, subsolar_lon_rad):
    T_night = 100.0
    cos_theta = np.cos(lat_rad) * np.cos(lon_fixed_rad - subsolar_lon_rad)

    # 昼側温度
    T0 = np.interp(AU, [0.307, 0.467], [600.0, 475.0])
    T_calc = T0 * (np.maximum(0.0, cos_theta) ** 0.25)
    return np.maximum(T_night, T_calc)


def calculate_thermal_desorption_rate(surface_temp_K):
    VIB_FREQ = 1e13
    BINDING_ENERGY_EV = 1.85
    BINDING_ENERGY_J = BINDING_ENERGY_EV * PHYSICAL_CONSTANTS['EV_TO_JOULE']
    k_B = PHYSICAL_CONSTANTS['K_BOLTZMANN']

    exponent = -BINDING_ENERGY_J / (k_B * np.maximum(surface_temp_K, 10.0))
    rate = np.zeros_like(surface_temp_K)
    mask = exponent > -700
    rate[mask] = VIB_FREQ * np.exp(exponent[mask])
    return rate


def get_orbit_params_from_taa(target_taa, orbit_data):
    # orbit_data: [TAA, AU, Time, ...]
    # TAA列から最も近い行を探す
    taa_col = orbit_data[:, 0]

    # 360度境界の処理は簡易的に無視（310-330なら問題ない）
    idx = np.argmin(np.abs(taa_col - target_taa))

    au = orbit_data[idx, 1]

    # Subsolar Longitude (col 5)
    sub_lon_deg = orbit_data[idx, 5]
    sub_lon_rad = np.deg2rad(sub_lon_deg)

    # サイクル補正（簡易: TAAが大きいときは偶数/奇数サイクルの影響を受けるが、
    # ここではファイル保存時の状況を再現したい。
    # 通常、Subsolar Lonは固定座標系での太陽位置なので、そのまま使う）

    return au, sub_lon_rad


# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    print(f"=== Playback Diagnostic: TAA {TAA_RANGE_MIN} - {TAA_RANGE_MAX} ===")
    print(f"Target Directory: {INPUT_DIR}")

    # 1. ファイル検索とソート
    pattern = os.path.join(INPUT_DIR, "surface_density_*.npy")
    files = glob.glob(pattern)

    if not files:
        print("Error: .npy files not found in the directory.")
        return

    # ファイル名からTAAを抽出してリスト化 (taa, filepath)
    valid_files = []
    for f in files:
        # 正規表現で taaXXX を抽出
        match = re.search(r"taa(\d+)", f)
        if match:
            taa_val = int(match.group(1))
            if TAA_RANGE_MIN <= taa_val <= TAA_RANGE_MAX:
                valid_files.append((taa_val, f))

    # TAA順にソート
    valid_files.sort(key=lambda x: x[0])

    if not valid_files:
        print("No files found within the specified TAA range.")
        return

    # 2. 軌道データ読み込み
    try:
        orbit_data = np.loadtxt('orbit2025_v6.txt')
    except:
        print("Error: orbit2025_v6.txt is missing.")
        return

    # 3. グリッド作成
    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
    LON_GRID, LAT_GRID = np.meshgrid(lon_c, lat_c, indexing='ij')

    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    CELL_AREAS_GRID = np.tile(cell_areas, (N_LON_FIXED, 1))

    # 4. 連続診断ループ
    print(
        f"\n{'TAA (File)':>10} | {'AU':>6} | {'Max Temp(K)':>11} | {'Max Dens(/m2)':>13} | {'Max Rate(/s)':>11} | {'TOTAL FLUX (atoms/s)':>22} | {'Status'}")
    print("-" * 115)

    for taa_val, file_path in valid_files:
        # A. データ読み込み
        density_grid = np.load(file_path)

        # B. 軌道パラメータ取得
        au, sub_lon = get_orbit_params_from_taa(taa_val, orbit_data)

        # C. 物理計算
        temp_grid = calculate_surface_temperature_leblanc(LON_GRID, LAT_GRID, au, sub_lon)
        rate_grid = calculate_thermal_desorption_rate(temp_grid)

        # 放出量 = 密度 × 脱離率 × 面積
        flux_grid = density_grid * rate_grid * CELL_AREAS_GRID
        total_flux = np.sum(flux_grid)

        # D. 統計情報
        max_dens = np.max(density_grid)
        max_temp = np.max(temp_grid)
        max_rate = np.max(rate_grid)

        status = ""
        if total_flux > 1e27:
            status = "!!! EXPLOSION !!!"
        elif total_flux > 1e25:
            status = "! High !"

        print(
            f"{taa_val:10d} | {au:6.3f} | {max_temp:11.2f} | {max_dens:13.2e} | {max_rate:11.2e} | {total_flux:22.2e} | {status}")

        # 異常時の詳細
        if "EXPLOSION" in status:
            # どこで爆発しているか？
            idx = np.unravel_index(np.argmax(flux_grid), flux_grid.shape)
            bad_lon_deg = np.degrees(lon_c[idx[0]])
            bad_lat_deg = np.degrees(lat_c[idx[1]])
            print(f"    -> Cause Location: Lon={bad_lon_deg:.1f}, Lat={bad_lat_deg:.1f}")
            print(
                f"    -> Local Values: Dens={density_grid[idx]:.2e}, Temp={temp_grid[idx]:.1f}K, Rate={rate_grid[idx]:.2e}")


if __name__ == "__main__":
    main()