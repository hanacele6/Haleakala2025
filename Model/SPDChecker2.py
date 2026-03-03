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
INPUT_DIR = r"./SimulationResult_202510/DynamicGrid72x36_10.0"

# 2. 診断したいTAAの範囲
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

# 追加: PSD計算用定数
F_UV_1AU = 1.5e14 * (100.0 ** 2)  # [photons/m2/s]
Q_PSD = 2.0e-20 / (100.0 ** 2)  # [m2]

N_LON_FIXED, N_LAT = 72, 36


# ==============================================================================
# 物理関数 (計算ロジックをシミュレーション本体と合わせる)
# ==============================================================================
def calculate_surface_temperature_leblanc(lon_fixed_rad, lat_rad, AU, subsolar_lon_rad):
    T_night = 100.0
    cos_theta = np.cos(lat_rad) * np.cos(lon_fixed_rad - subsolar_lon_rad)

    # 昼側温度 (600K設定のまま計算)
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


def calculate_psd_rate(lon_fixed_rad, lat_rad, AU, subsolar_lon_rad):
    # フラックスは距離の2乗に反比例
    f_uv_curr = F_UV_1AU / (AU ** 2)

    # cos(SZA)
    cos_theta = np.cos(lat_rad) * np.cos(lon_fixed_rad - subsolar_lon_rad)

    # 昼側のみ
    rate = np.maximum(0.0, f_uv_curr * Q_PSD * cos_theta)
    return rate


def get_orbit_params_from_taa(target_taa, orbit_data):
    taa_col = orbit_data[:, 0]
    # 360度境界処理 (簡易)
    diff = np.abs(taa_col - target_taa)
    idx = np.argmin(diff)
    au = orbit_data[idx, 1]
    sub_lon_deg = orbit_data[idx, 5]
    return au, np.deg2rad(sub_lon_deg)


# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    print(f"=== Deep Dive Diagnostic: TAA {TAA_RANGE_MIN} - {TAA_RANGE_MAX} ===")
    print(f"Target Directory: {INPUT_DIR}")
    print("-" * 140)
    print(
        f"{'TAA':>4} | {'AU':>5} | {'TotFlux':>10} | {' vs Prev':>8} | {'TD(%)':>5} {'PSD(%)':>6} | {'MaxDens':>9} | {'MaxTemp':>7} | {'Dominant Grid Info (Lon, Lat, Rate)'}")
    print("-" * 140)

    # 1. ファイル検索
    pattern = os.path.join(INPUT_DIR, "surface_density_*.npy")
    files = glob.glob(pattern)

    valid_files = []
    for f in files:
        match = re.search(r"taa(\d+)", f)
        if match:
            taa_val = int(match.group(1))
            if TAA_RANGE_MIN <= taa_val <= TAA_RANGE_MAX:
                valid_files.append((taa_val, f))
    valid_files.sort(key=lambda x: x[0])

    if not valid_files:
        print("No files found.")
        return

    # 2. 軌道データ
    try:
        orbit_data = np.loadtxt('orbit2025_v6.txt')
    except:
        print("Error: orbit2025_v6.txt missing.")
        return

    # 3. グリッド計算
    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
    LON_GRID, LAT_GRID = np.meshgrid(lon_c, lat_c, indexing='ij')

    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    CELL_AREAS_GRID = np.tile(cell_areas, (N_LON_FIXED, 1))

    # 変数初期化
    prev_flux = None

    # 4. ループ解析
    for taa_val, file_path in valid_files:
        # Load
        density_grid = np.load(file_path)
        au, sub_lon = get_orbit_params_from_taa(taa_val, orbit_data)

        # Physics Calc
        temp_grid = calculate_surface_temperature_leblanc(LON_GRID, LAT_GRID, au, sub_lon)

        # Rate Calculation (TD & PSD)
        rate_td = calculate_thermal_desorption_rate(temp_grid)
        rate_psd = calculate_psd_rate(LON_GRID, LAT_GRID, au, sub_lon)
        rate_total = rate_td + rate_psd

        # Flux Calculation
        flux_grid_td = density_grid * rate_td * CELL_AREAS_GRID
        flux_grid_psd = density_grid * rate_psd * CELL_AREAS_GRID
        flux_grid_total = flux_grid_td + flux_grid_psd

        sum_flux = np.sum(flux_grid_total)
        sum_td = np.sum(flux_grid_td)
        sum_psd = np.sum(flux_grid_psd)

        # 変化率 (vs Previous Step)
        flux_change_str = "  - "
        if prev_flux is not None and prev_flux > 0:
            ratio = sum_flux / prev_flux
            if ratio > 1.5:
                flux_change_str = f"\033[91m↑{ratio:.1f}x\033[0m"  # Red for jump
            elif ratio < 0.7:
                flux_change_str = f"\033[94m↓{ratio:.1f}x\033[0m"  # Blue for drop
            else:
                flux_change_str = f" {ratio:.1f}x"

        prev_flux = sum_flux

        # 割合
        pct_td = (sum_td / sum_flux * 100) if sum_flux > 0 else 0
        pct_psd = (sum_psd / sum_flux * 100) if sum_flux > 0 else 0

        # Max Info
        max_dens = np.max(density_grid)
        max_temp = np.max(temp_grid)

        # Dominant Grid (最も放出に寄与している場所)
        idx_max = np.unravel_index(np.argmax(flux_grid_total), flux_grid_total.shape)
        dom_lon = np.degrees(lon_c[idx_max[0]])
        dom_lat = np.degrees(lat_c[idx_max[1]])
        dom_rate = rate_total[idx_max]
        dom_dens = density_grid[idx_max]

        # 寿命の計算 (1/rate)
        lifetime_str = f"{1.0 / dom_rate:.1e}s" if dom_rate > 1e-9 else "inf"

        # Print Row
        print(
            f"{taa_val:4d} | {au:.3f} | {sum_flux:.2e} | {flux_change_str:>8} | {pct_td:3.0f}% {pct_psd:3.0f}% | {max_dens:.1e} | {max_temp:5.1f}K | "
            f"Loc({dom_lon:4.0f},{dom_lat:3.0f}) Rate={dom_rate:.1e}(t={lifetime_str}) Dens={dom_dens:.1e}")


if __name__ == "__main__":
    main()