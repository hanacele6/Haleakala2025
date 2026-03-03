# -*- coding: utf-8 -*-
"""
==============================================================================
事後解析ツール: 昼面ナトリウム総量 & プロセス別生成量 (個別プロット版)
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# ==============================================================================
# 1. 設定・定数
# ==============================================================================
# 解析対象ディレクトリ
TARGET_DIR = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_0116_0.4Denabled_2.7_LowestQ"
ORBIT_FILE = 'orbit2025_spice_unwrapped.txt'

# 物理定数
RM = 2.440e6  # 水星半径 [m]
KB = 1.380649e-23
EV_TO_JOULE = 1.602e-19
ROTATION_PERIOD = 58.6462 * 86400

# グリッド設定
N_LON_FIXED = 72
N_LAT = 36

# プロセスパラメータ
# PSD
F_UV_1AU = 1.5e14 * (100 ** 2)
Q_PSD = 2.7e-21 / (100 ** 2)

# TD
TEMP_BASE = 100.0
TEMP_AMP = 600.0
U_EV = 2.7
VIB_FREQ = 1e13


# ==============================================================================
# 2. 計算関数
# ==============================================================================

def calculate_surface_temperature(lon_rad, lat_rad, AU, subsolar_lon_rad):
    # sqrt (ルート) 依存
    scaling = np.sqrt(0.306 / AU)
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)
    if cos_theta <= 0: return TEMP_BASE
    return TEMP_BASE + TEMP_AMP * (cos_theta ** 0.25) * scaling


def calculate_td_rate_factor(temp_k):
    if temp_k < 10.0: return 0.0
    U_JOULE = U_EV * EV_TO_JOULE
    exponent = -U_JOULE / (KB * temp_k)
    if exponent < -700: return 0.0
    return VIB_FREQ * np.exp(exponent)


def calculate_psd_rate_factor(au, cos_theta):
    if cos_theta <= 0: return 0.0
    f_uv = F_UV_1AU / (au ** 2)
    return f_uv * Q_PSD * cos_theta


def get_orbital_info_from_time(rel_hours, orbit_data):
    idx_peri = np.argmin(np.abs(orbit_data[:, 0]))
    t_peri = orbit_data[idx_peri, 2]
    current_time = t_peri + rel_hours * 3600.0
    time_col = orbit_data[:, 2]
    idx = np.searchsorted(time_col, current_time)
    if idx >= len(time_col): idx = len(time_col) - 1
    if idx > 0 and abs(current_time - time_col[idx - 1]) < abs(current_time - time_col[idx]): idx -= 1
    au = orbit_data[idx, 1]
    taa_deg_raw = orbit_data[idx, 0]
    omega_rot = 2 * np.pi / ROTATION_PERIOD
    rotation_angle = omega_rot * (current_time - t_peri)
    taa_rad = np.deg2rad(taa_deg_raw)
    sub_lon = taa_rad - rotation_angle
    sub_lon = (sub_lon + np.pi) % (2 * np.pi) - np.pi
    return taa_deg_raw % 360.0, au, sub_lon


# ==============================================================================
# 3. メイン処理
# ==============================================================================
def main():
    print("--- Individual Plots Analysis ---")

    if not os.path.exists(TARGET_DIR):
        print(f"[ERROR] Directory not found: {TARGET_DIR}")
        return

    pattern = os.path.join(TARGET_DIR, "surface_density_*.npy")
    files = sorted(glob.glob(pattern))

    if not os.path.exists(ORBIT_FILE):
        print(f"[ERROR] Orbit file not found: {ORBIT_FILE}")
        return

    orbit_data = np.loadtxt(ORBIT_FILE)
    orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))

    # グリッド面積計算
    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (RM ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    area_grid = np.tile(cell_areas, (N_LON_FIXED, 1))

    data_list = []
    print(f"Processing {len(files)} files...")

    for i, fpath in enumerate(files):
        if i % 100 == 0: print(f"Processing {i}/{len(files)}...")
        match = re.search(r"t(\d+)", fpath)
        if not match: continue

        try:
            rel_hours = int(match.group(1))
            taa, au, sub_lon = get_orbital_info_from_time(rel_hours, orbit_data)
            surf_dens = np.load(fpath)
        except Exception:
            continue

        total_atoms_dayside = 0.0
        weighted_psd_total = 0.0
        weighted_td_total = 0.0

        for ix in range(N_LON_FIXED):
            lon_f = (lon_edges[ix] + lon_edges[ix + 1]) / 2
            for iy in range(N_LAT):
                lat_f = (lat_edges[iy] + lat_edges[iy + 1]) / 2
                cos_z = np.cos(lat_f) * np.cos(lon_f - sub_lon)

                # 昼面限定
                if cos_z > 0:
                    area = area_grid[ix, iy]
                    dens = surf_dens[ix, iy]

                    # 1. 在庫量
                    total_atoms_dayside += dens * area

                    # 2. PSD Flux
                    psd_rate = calculate_psd_rate_factor(au, cos_z)
                    weighted_psd_total += dens * psd_rate * area

                    # 3. TD Flux
                    temp = calculate_surface_temperature(lon_f, lat_f, au, sub_lon)
                    td_rate = calculate_td_rate_factor(temp)
                    weighted_td_total += dens * td_rate * area

        data_list.append((taa, total_atoms_dayside, weighted_psd_total, weighted_td_total))

    data_arr = np.array(data_list)
    idx_sort = np.argsort(data_arr[:, 0])
    data_arr = data_arr[idx_sort]

    taa_sorted = data_arr[:, 0]
    inventory = data_arr[:, 1]
    flux_psd = data_arr[:, 2]
    flux_td = data_arr[:, 3]

    # ==========================================================================
    # 図1: 昼面総在庫量 (Inventory) - 黒線
    # ==========================================================================
    plt.figure(figsize=(8, 5))
    plt.plot(taa_sorted, inventory, color='black', linewidth=2, label='Dayside Inventory')
    plt.xlabel('True Anomaly Angle [deg]', fontsize=12)
    plt.ylabel('Total Na Atoms [atoms]', fontsize=12)
    plt.title('Total Dayside Na Inventory', fontsize=14)
    plt.xlim(0, 360)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_1_inventory.png', dpi=300)
    print("Saved: plot_1_inventory.png")

    # ==========================================================================
    # 図2: PSD重み付け生成量 (PSD Flux) - 青線
    # ==========================================================================
    plt.figure(figsize=(8, 5))
    plt.plot(taa_sorted, flux_psd, color='blue', linewidth=2, label='PSD Flux')
    plt.xlabel('True Anomaly Angle [deg]', fontsize=12)
    plt.ylabel('PSD Weighted Flux [atoms/s]', fontsize=12)
    plt.title('PSD Generation Potential', fontsize=14)
    plt.xlim(0, 360)
    plt.yscale('log')  # 対数軸
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_2_psd_flux.png', dpi=300)
    print("Saved: plot_2_psd_flux.png")

    # ==========================================================================
    # 図3: TD重み付け生成量 (TD Flux) - 赤線
    # ==========================================================================
    plt.figure(figsize=(8, 5))
    plt.plot(taa_sorted, flux_td, color='red', linewidth=2, label='TD Flux')
    plt.xlabel('True Anomaly Angle [deg]', fontsize=12)
    plt.ylabel('TD Weighted Flux [atoms/s]', fontsize=12)
    plt.title('Thermal Desorption Generation Potential', fontsize=14)
    plt.xlim(0, 360)
    plt.yscale('log')  # 対数軸
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_3_td_flux.png', dpi=300)
    print("Saved: plot_3_td_flux.png")

    plt.show()


if __name__ == "__main__":
    main()