# -*- coding: utf-8 -*-
"""
==============================================================================
事後解析ツール (Leblancスタイル完全版 + 2Dマップ平均化・修正版):
1. TAAごとの時系列推移 (Time-Evolution)
2. 枯渇タイムスケール分布 (Histogram)
3. ★修正: Local Time vs Latitude 2Dマップ (TAA範囲平均 + 全域色付け)
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, NullFormatter
from matplotlib.colors import LogNorm
import os
import glob
import re

# ==============================================================================
# 1. 設定・定数
# ==============================================================================
TARGET_DIR = r"./SimulationResult_202511/DynamicGrid72x36_18.0"
ORBIT_FILE = 'orbit2025_v6.txt'

# --- ★解析設定 ---
TARGET_TAA_CENTER = 30.0  # 解析したいTAAの中心
TAA_WIDTH = 10.0  # 平均化する幅 (+/- 5 deg)
N_BINS = 50  # ヒストグラムの分割数

# グリッド設定
N_LON_FIXED = 72
N_LAT = 36

# 物理定数
PI = np.pi
RM = 2.440e6
KB = 1.380649e-23
EV_TO_JOULE = 1.602e-19
ROTATION_PERIOD = 58.6462 * 86400

# モデルパラメータ
F_UV_1AU = 1.5e14 * (100 ** 2)
Q_PSD = 1.0e-20 / (100 ** 2)
SWS_PARAMS = {
    'FLUX_1AU': 10.0 * 100 ** 3 * 400e3 * 4.0,
    'YIELD': 0.06,
    'REF_DENS': 7.5e14 * 100 ** 2,
    'LON_RANGE': np.deg2rad([-40, 40]),
    'LAT_N_RANGE': np.deg2rad([20, 80]),
    'LAT_S_RANGE': np.deg2rad([-80, -20]),
}


# ==============================================================================
# 2. 計算関数群
# ==============================================================================
def calculate_surface_temperature_leblanc(lon_rad, lat_rad, AU, subsolar_lon_rad):
    T_BASE = 100.0  # 夜面/ベース温度
    T_ANGLE = 600.0  # 日中変動成分

    scaling = np.sqrt(0.306 / AU)
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)

    if cos_theta <= 0:
        return T_BASE
    return T_BASE + T_ANGLE * (cos_theta ** 0.25) * scaling


def calculate_thermal_desorption_rate(temp_k):
    if temp_k < 10.0: return 0.0
    VIB_FREQ = 1e13
    U_MEAN = 1.85
    U_MIN = 1.40
    U_MAX = 2.70
    SIGMA = 0.20
    u_ev_grid = np.linspace(U_MIN, U_MAX, 50)
    u_joule_grid = u_ev_grid * EV_TO_JOULE
    pdf = np.exp(- (u_ev_grid - U_MEAN) ** 2 / (2 * SIGMA ** 2))
    norm_pdf = pdf / np.sum(pdf)
    exponent = -u_joule_grid / (KB * temp_k)
    rates = np.zeros_like(u_ev_grid)
    mask = exponent > -700
    rates[mask] = VIB_FREQ * np.exp(exponent[mask])
    return np.sum(rates * norm_pdf)


def calculate_mmv_total_rate(r_au):
    TOTAL_FLUX_AT_PERI = 5e23
    PERIHELION_AU = 0.307
    AREA = 4 * PI * (RM ** 2)
    avg_flux_peri = TOTAL_FLUX_AT_PERI / AREA
    C = avg_flux_peri * (PERIHELION_AU ** 1.9)
    flux = C * (r_au ** (-1.9))
    return flux * AREA


def get_orbital_info_from_time(rel_hours, orbit_data):
    idx_peri = np.argmin(np.abs(orbit_data[:, 0]))
    t_peri = orbit_data[idx_peri, 2]
    current_time = t_peri + rel_hours * 3600.0
    time_col = orbit_data[:, 2]
    idx = np.searchsorted(time_col, current_time)
    if idx >= len(time_col): idx = len(time_col) - 1
    if idx > 0 and abs(current_time - time_col[idx - 1]) < abs(current_time - time_col[idx]):
        idx -= 1
    au = orbit_data[idx, 1]
    taa_deg = orbit_data[idx, 0]

    omega_rot = 2 * np.pi / ROTATION_PERIOD
    rotation_angle = omega_rot * (current_time - t_peri)
    taa_rad = np.deg2rad(taa_deg)
    sub_lon = taa_rad - rotation_angle
    sub_lon = (sub_lon + np.pi) % (2 * np.pi) - np.pi
    return taa_deg, au, sub_lon


# ==============================================================================
# 3. 解析機能: ヒストグラム (MMV除外版・平均化)
# ==============================================================================
def analyze_lifetime_contribution_averaged(target_taa, width, files, orbit_data):
    print(f"\n--- Starting Averaged Lifetime Analysis (Histogram) ---")
    print(f"Target TAA: {target_taa} +/- {width / 2} deg")

    # --- ファイル収集 (平均化対象) ---
    target_files = []
    taa_min = target_taa - width / 2.0
    taa_max = target_taa + width / 2.0
    cross_zero = False
    if taa_min < 0:
        taa_min += 360;
        cross_zero = True
    elif taa_max > 360:
        taa_max -= 360;
        cross_zero = True

    for fpath in files:
        match = re.search(r"t(\d+)", fpath)
        if not match: continue
        rel_hours = int(match.group(1))
        taa, au, sub_lon = get_orbital_info_from_time(rel_hours, orbit_data)

        hit = False
        if not cross_zero:
            if taa_min <= taa <= taa_max: hit = True
        else:
            if taa >= taa_min or taa <= taa_max: hit = True
        if hit: target_files.append((fpath, taa, au, sub_lon))

    if not target_files:
        print("No files found in range.")
        return

    all_lifetimes = []
    all_prod_td = [];
    all_prod_psd = [];
    all_prod_sws = [];
    all_prod_mmv = []

    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (RM ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    area_grid = np.tile(cell_areas, (N_LON_FIXED, 1))

    # --- 各ファイルをループ処理 ---
    for fpath, taa, au, sub_lon in target_files:
        surf_dens = np.load(fpath)
        f_uv = F_UV_1AU / (au ** 2)
        sw_flux_base = SWS_PARAMS['FLUX_1AU'] / (au ** 2)
        mmv_flux_total = calculate_mmv_total_rate(au)
        mmv_flux_per_m2 = mmv_flux_total / (4 * PI * RM ** 2)

        for i_lon in range(N_LON_FIXED):
            lon_f = (lon_edges[i_lon] + lon_edges[i_lon + 1]) / 2
            lon_sun = (lon_f - sub_lon + np.pi) % (2 * np.pi) - np.pi
            for j_lat in range(N_LAT):
                lat_f = (lat_edges[j_lat] + lat_edges[j_lat + 1]) / 2
                area = area_grid[i_lon, j_lat]
                dens = surf_dens[i_lon, j_lat]
                cos_z = np.cos(lat_f) * np.cos(lon_f - sub_lon)

                # Histogramでは完全に夜や放出ゼロの場所は除外する
                if cos_z <= 0: continue

                flux_psd_val = f_uv * Q_PSD * cos_z * dens
                temp = calculate_surface_temperature_leblanc(lon_f, lat_f, au, sub_lon)
                flux_td_val = dens * calculate_thermal_desorption_rate(temp)

                flux_sws_val = 0.0
                in_lon = SWS_PARAMS['LON_RANGE'][0] <= lon_sun <= SWS_PARAMS['LON_RANGE'][1]
                in_lat = (SWS_PARAMS['LAT_N_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_N_RANGE'][1]) or \
                         (SWS_PARAMS['LAT_S_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_S_RANGE'][1])
                if in_lon and in_lat:
                    flux_sws_val = dens * ((sw_flux_base * SWS_PARAMS['YIELD']) / SWS_PARAMS['REF_DENS'])

                depletion_flux = flux_td_val + flux_psd_val + flux_sws_val
                flux_mmv_val = mmv_flux_per_m2

                if depletion_flux <= 0 or dens <= 0: continue
                lifetime = dens / depletion_flux

                all_lifetimes.append(lifetime)
                all_prod_td.append(flux_td_val * area)
                all_prod_psd.append(flux_psd_val * area)
                all_prod_sws.append(flux_sws_val * area)
                all_prod_mmv.append(flux_mmv_val * area)

    # --- ヒストグラム描画 ---
    lifetimes = np.array(all_lifetimes)
    prod_td = np.array(all_prod_td);
    prod_psd = np.array(all_prod_psd)
    prod_sws = np.array(all_prod_sws);
    prod_mmv = np.array(all_prod_mmv)
    if len(lifetimes) == 0: return

    min_tau = np.min(lifetimes);
    max_tau = np.max(lifetimes)
    bins = np.logspace(np.floor(np.log10(min_tau)), np.ceil(np.log10(max_tau)), N_BINS)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    indices = np.digitize(lifetimes, bins)

    sum_td = np.zeros(len(bin_centers));
    sum_psd = np.zeros(len(bin_centers))
    sum_sws = np.zeros(len(bin_centers));
    sum_mmv = np.zeros(len(bin_centers))

    for i in range(len(lifetimes)):
        idx = indices[i] - 1
        if 0 <= idx < len(bin_centers):
            sum_td[idx] += prod_td[i];
            sum_psd[idx] += prod_psd[i]
            sum_sws[idx] += prod_sws[i];
            sum_mmv[idx] += prod_mmv[i]

    # ファイル数で平均化
    num_files = len(target_files)
    sum_td /= num_files;
    sum_psd /= num_files
    sum_sws /= num_files;
    sum_mmv /= num_files
    total_in_bin = sum_td + sum_psd + sum_sws + sum_mmv

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax1.plot(bin_centers, total_in_bin, 'k-', linewidth=2, label='Total')
    ax1.plot(bin_centers, sum_td, 'r--', label='TD')
    ax1.plot(bin_centers, sum_psd, 'b--', label='PSD')
    ax1.plot(bin_centers, sum_sws, 'g--', label='SWS')
    ax1.plot(bin_centers, sum_mmv, 'orange', linestyle=':', label='MMV')
    ax1.set_xscale('log');
    ax1.set_yscale('log')
    ax1.set_ylabel('Avg. Production Rate [atoms/s]')
    ax1.grid(True, alpha=0.5);
    ax1.legend()

    with np.errstate(divide='ignore', invalid='ignore'):
        frac_td = np.nan_to_num(sum_td / total_in_bin * 100)
        frac_psd = np.nan_to_num(sum_psd / total_in_bin * 100)
        frac_sws = np.nan_to_num(sum_sws / total_in_bin * 100)
        frac_mmv = np.nan_to_num(sum_mmv / total_in_bin * 100)

    ax2.stackplot(bin_centers, frac_td, frac_psd, frac_sws, frac_mmv,
                  labels=['TD', 'PSD', 'SWS', 'MMV'],
                  colors=['red', 'blue', 'green', 'orange'], alpha=0.6)
    ax2.set_xscale('log')
    ax2.set_ylabel('Contribution [%]')
    ax2.set_xlabel('Lifetime [sec]')
    ax2.set_ylim(0, 100);
    ax2.grid(True, alpha=0.5);
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f"lifetime_dist_avg_taa{int(target_taa)}.png", dpi=300)
    print("Histogram saved.")
    plt.show()


# ==============================================================================
# 4. ★修正版: 2Dマップ (平均化 + Regridding + pcolormesh)
# ==============================================================================
def analyze_lifetime_2d_map_averaged(target_taa, width, files, orbit_data):
    """
    指定TAA範囲に含まれる複数のファイルを読み込み、
    それぞれのデータを「Local Time (0-24h)」グリッドに補間(Regridding)してから
    平均化して描画する関数。
    """
    print(f"\n--- Generating Averaged 2D Map for TAA {target_taa} +/- {width / 2} deg ---")

    # --- 1. ファイル特定 (ヒストグラムと同様) ---
    target_files = []
    taa_min = target_taa - width / 2.0
    taa_max = target_taa + width / 2.0
    cross_zero = False
    if taa_min < 0:
        taa_min += 360;
        cross_zero = True
    elif taa_max > 360:
        taa_max -= 360;
        cross_zero = True

    for fpath in files:
        match = re.search(r"t(\d+)", fpath)
        if not match: continue
        rel_hours = int(match.group(1))
        taa, au, sub_lon = get_orbital_info_from_time(rel_hours, orbit_data)

        hit = False
        if not cross_zero:
            if taa_min <= taa <= taa_max: hit = True
        else:
            if taa >= taa_min or taa <= taa_max: hit = True
        if hit: target_files.append((fpath, taa, au, sub_lon))

    if not target_files:
        print("No files found in range for 2D map.")
        return

    print(f"Number of files to average: {len(target_files)}")

    # --- 2. 共通グリッドの定義 (Local Time) ---
    # N_LON_FIXEDと同じ解像度で0h-24hの軸を作る
    common_lt_axis = np.linspace(0, 24, N_LON_FIXED, endpoint=False)

    # 平均化用のアキュムレータ (LT x LAT)
    sum_lifetime_grid = np.zeros((N_LON_FIXED, N_LAT))
    sum_contrib_grid = np.zeros((N_LON_FIXED, N_LAT))

    # Lon/Lat定義
    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lat_deg = np.rad2deg(lat_centers)

    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (RM ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    area_grid = np.tile(cell_areas, (N_LON_FIXED, 1))

    # --- 3. ループ処理 (計算 -> LT変換 -> 蓄積) ---
    for fpath, taa, au, sub_lon in target_files:
        surf_dens = np.load(fpath)

        # フラックス計算用パラメータ
        f_uv = F_UV_1AU / (au ** 2)
        sw_flux_base = SWS_PARAMS['FLUX_1AU'] / (au ** 2)
        mmv_flux_total = calculate_mmv_total_rate(au)
        mmv_flux_per_m2 = mmv_flux_total / (4 * PI * RM ** 2)

        # 一時配列 (Lon x Lat)
        temp_lifetime = np.zeros((N_LON_FIXED, N_LAT))
        temp_prod_rate = np.zeros((N_LON_FIXED, N_LAT))

        # (A) 物理量計算 (Lon grid)
        for i_lon in range(N_LON_FIXED):
            lon_f = lon_centers[i_lon]
            lon_sun = (lon_f - sub_lon + np.pi) % (2 * np.pi) - np.pi

            for j_lat in range(N_LAT):
                lat_f = lat_centers[j_lat]
                dens = surf_dens[i_lon, j_lat]
                area = area_grid[i_lon, j_lat]
                cos_z = np.cos(lat_f) * np.cos(lon_f - sub_lon)

                # Fluxes
                flux_psd = f_uv * Q_PSD * cos_z * dens if cos_z > 0 else 0.0
                temp = calculate_surface_temperature_leblanc(lon_f, lat_f, au, sub_lon)
                flux_td = dens * calculate_thermal_desorption_rate(temp)

                flux_sws = 0.0
                in_lon = SWS_PARAMS['LON_RANGE'][0] <= lon_sun <= SWS_PARAMS['LON_RANGE'][1]
                in_lat = (SWS_PARAMS['LAT_N_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_N_RANGE'][1]) or \
                         (SWS_PARAMS['LAT_S_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_S_RANGE'][1])
                if in_lon and in_lat:
                    flux_sws = dens * ((sw_flux_base * SWS_PARAMS['YIELD']) / SWS_PARAMS['REF_DENS'])

                flux_mmv = mmv_flux_per_m2

                total_flux = flux_td + flux_psd + flux_sws + flux_mmv
                depletion = flux_td + flux_psd + flux_sws

                temp_prod_rate[i_lon, j_lat] = total_flux * area

                if depletion > 0 and dens > 0:
                    temp_lifetime[i_lon, j_lat] = dens / depletion
                else:
                    temp_lifetime[i_lon, j_lat] = 0.0

        # Contribution率 (Global Normalization for this file)
        total_rate_global = np.sum(temp_prod_rate)
        temp_contrib = np.zeros_like(temp_prod_rate)
        if total_rate_global > 0:
            temp_contrib = (temp_prod_rate / total_rate_global) * 100.0

        # (B) LT座標への変換 (Regridding)
        # 現在のファイルの各Lonグリッドに対応するLTを計算 (0-24h)
        # 式: LT = 12 + (Lon - SubLon)/Pi * 12
        current_lt_1d = (12.0 + ((lon_centers - sub_lon + np.pi) % (2 * np.pi) - np.pi) / np.pi * 12.0) % 24.0

        # 補間のためにLT順にソート
        sort_idx = np.argsort(current_lt_1d)
        sorted_lt = current_lt_1d[sort_idx]

        # 周期境界処理 (0hと24hをつなぐためにデータを拡張する)
        # sorted_ltの末尾に「最初の点+24h」を追加し、データも拡張する
        extended_lt = np.concatenate([sorted_lt, [sorted_lt[0] + 24.0]])

        # 各緯度ごとにLT軸へ補間して蓄積
        for j in range(N_LAT):
            # Lifetime
            data_row_lt = temp_lifetime[sort_idx, j]
            extended_data_lt = np.concatenate([data_row_lt, [data_row_lt[0]]])
            # 線形補間 (Common LT Gridへ)
            interp_lt = np.interp(common_lt_axis, extended_lt, extended_data_lt)
            sum_lifetime_grid[:, j] += interp_lt

            # Contribution
            data_row_con = temp_contrib[sort_idx, j]
            extended_data_con = np.concatenate([data_row_con, [data_row_con[0]]])
            interp_con = np.interp(common_lt_axis, extended_lt, extended_data_con)
            sum_contrib_grid[:, j] += interp_con

    # --- 4. 平均化 ---
    avg_lifetime_map = sum_lifetime_grid / len(target_files)
    avg_contrib_map = sum_contrib_grid / len(target_files)

    # --- 5. 描画用データ整形 (Cyclic) ---
    # pcolormesh用に最後のデータを先頭にコピーして閉じる (0h = 24h)
    plot_lt = np.concatenate([common_lt_axis, [24.0]])
    plot_lifetime = np.vstack([avg_lifetime_map, avg_lifetime_map[0:1, :]])
    plot_contrib = np.vstack([avg_contrib_map, avg_contrib_map[0:1, :]])

    LT_GRID, LAT_GRID = np.meshgrid(plot_lt, lat_deg, indexing='ij')

    # --- 6. Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Lifetime Plot (vmin=0.1sec に修正)
    plot_data_lt = plot_lifetime.copy()
    plot_data_lt[plot_data_lt <= 0] = 1e-5  # Log scale safety

    mesh1 = ax1.pcolormesh(LT_GRID, LAT_GRID, plot_data_lt,
                           norm=LogNorm(vmin=1e-1, vmax=1e7),  # <--- 0.1秒まで見えるように変更
                           cmap='jet', shading='auto')
    cb1 = plt.colorbar(mesh1, ax=ax1)
    cb1.set_label('Avg Surface Residence Time [sec]')
    ax1.set_title(f"Averaged Residence Time (TAA {target_taa} +/- {width / 2})")
    ax1.set_ylabel("Latitude [deg]")
    ax1.set_xlim(0, 24);
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.set_ylim(-90, 90)

    # Contribution Plot
    plot_data_con = plot_contrib.copy()
    plot_data_con[plot_data_con <= 0] = 1e-9

    mesh2 = ax2.pcolormesh(LT_GRID, LAT_GRID, plot_data_con,
                           norm=LogNorm(vmin=1e-4, vmax=np.max(plot_data_con) if np.max(plot_data_con) > 0 else 1.0),
                           cmap='magma', shading='auto')
    cb2 = plt.colorbar(mesh2, ax=ax2)
    cb2.set_label('Avg Contribution [%]')
    ax2.set_title("Averaged Contribution Rate")
    ax2.set_ylabel("Latitude [deg]")
    ax2.set_xlabel("Local Time [hour]")
    ax2.set_xlim(0, 24);
    ax2.set_xticks(np.arange(0, 25, 2))
    ax2.set_ylim(-90, 90)

    plt.tight_layout()
    plt.savefig(f"lifetime_map_averaged_taa{int(target_taa)}.png", dpi=200)
    print("Averaged 2D Map saved.")
    plt.show()


# ==============================================================================
# 5. メイン処理
# ==============================================================================
def main():
    pattern = os.path.join(TARGET_DIR, "surface_density_*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No files found.")
        return
    orbit_data = np.loadtxt(ORBIT_FILE)
    orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))

    # Phase 2: Histogram (Average)
    analyze_lifetime_contribution_averaged(TARGET_TAA_CENTER, TAA_WIDTH, files, orbit_data)

    # Phase 3: 2D Map (Average + Regridding)
    analyze_lifetime_2d_map_averaged(TARGET_TAA_CENTER, TAA_WIDTH, files, orbit_data)


if __name__ == "__main__":
    main()