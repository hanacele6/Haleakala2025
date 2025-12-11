# -*- coding: utf-8 -*-
"""
==============================================================================
事後解析ツール (Leblancスタイル版):
表面密度ファイルから時刻を読み、放出プロセスごとの生成率と支配率(%)を計算する
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
import glob
import re

# ==============================================================================
# 1. 設定・定数 (環境に合わせてパスを変更してください)
# ==============================================================================
# 解析対象のディレクトリ
TARGET_DIR = r"./SimulationResult_202511/DynamicGrid72x36_18.0"

# 軌道データファイル
ORBIT_FILE = 'orbit2025_v6.txt'

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
def calculate_surface_temperature_leblanc(lon_rad, lat_rad, r_au, sub_lon):
    T0 = 100.0
    T1 = 600.0
    scaling = np.sqrt(0.306 / r_au)
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - sub_lon)
    if cos_theta <= 0:
        return T0
    return T0 + T1 * (cos_theta ** 0.25) * scaling


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
    # MMV総放出率 [atoms/s]
    TOTAL_FLUX_AT_PERI = 5e23
    PERIHELION_AU = 0.307
    AREA = 4 * PI * (RM ** 2)
    avg_flux_peri = TOTAL_FLUX_AT_PERI / AREA
    C = avg_flux_peri * (PERIHELION_AU ** 1.9)
    flux = C * (r_au ** (-1.9))
    return flux * AREA


def get_orbital_info_from_time(rel_hours, orbit_data):
    """
    ファイル名の tXXXXX (相対時間[hour]) から軌道情報を計算する
    """
    idx_peri = np.argmin(np.abs(orbit_data[:, 0]))
    t_peri = orbit_data[idx_peri, 2]

    current_time = t_peri + rel_hours * 3600.0
    time_col = orbit_data[:, 2]

    # 時刻から検索
    idx = np.searchsorted(time_col, current_time)
    if idx >= len(time_col): idx = len(time_col) - 1
    if idx > 0:
        if abs(current_time - time_col[idx - 1]) < abs(current_time - time_col[idx]):
            idx -= 1

    au = orbit_data[idx, 1]
    taa_deg = orbit_data[idx, 0]

    # Subsolar Longitude
    omega_rot = 2 * np.pi / ROTATION_PERIOD
    rotation_angle = omega_rot * (current_time - t_peri)
    taa_rad = np.deg2rad(taa_deg)
    sub_lon = taa_rad - rotation_angle
    sub_lon = (sub_lon + np.pi) % (2 * np.pi) - np.pi

    return taa_deg, au, sub_lon


# ==============================================================================
# 3. メイン解析処理
# ==============================================================================
def main():
    # ファイルリスト取得
    pattern = os.path.join(TARGET_DIR, "surface_density_*.npy")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"Error: No files found in {TARGET_DIR}")
        return

    # 軌道データ読み込み
    try:
        orbit_data = np.loadtxt(ORBIT_FILE)
        orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))
    except Exception as e:
        print(f"Error loading orbit file: {e}")
        return

    # グリッド設定
    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (RM ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    area_grid = np.tile(cell_areas, (N_LON_FIXED, 1))

    # 結果格納用
    result_taa = []
    rates_td = []
    rates_psd = []
    rates_sws = []
    rates_mmv = []

    print(f"Processing {len(files)} files...")

    for i, fpath in enumerate(files):
        if i % 50 == 0: print(f"Processing... {i}/{len(files)}")

        match = re.search(r"t(\d+)", fpath)
        if not match: continue
        rel_hours = int(match.group(1))

        taa_deg, au, sub_lon = get_orbital_info_from_time(rel_hours, orbit_data)

        surf_dens = np.load(fpath)
        if surf_dens.shape != (N_LON_FIXED, N_LAT):
            continue

        # --- 各プロセスのレート計算 [atoms/s] ---

        # 1. MMV
        total_mmv = calculate_mmv_total_rate(au)

        # 2. PSD, TD, SWS
        total_td = 0.0
        total_psd = 0.0
        total_sws = 0.0

        f_uv = F_UV_1AU / (au ** 2)
        sw_flux = SWS_PARAMS['FLUX_1AU'] / (au ** 2)

        for i_lon in range(N_LON_FIXED):
            lon_f = (lon_edges[i_lon] + lon_edges[i_lon + 1]) / 2
            lon_sun = (lon_f - sub_lon + np.pi) % (2 * np.pi) - np.pi

            for j_lat in range(N_LAT):
                lat_f = (lat_edges[j_lat] + lat_edges[j_lat + 1]) / 2
                area = area_grid[i_lon, j_lat]
                dens = surf_dens[i_lon, j_lat]

                # PSD
                cos_z = np.cos(lat_f) * np.cos(lon_f - sub_lon)
                if cos_z > 0:
                    rate_psd = f_uv * Q_PSD * cos_z
                    total_psd += dens * rate_psd * area

                # TD
                temp = calculate_surface_temperature_leblanc(lon_f, lat_f, au, sub_lon)
                rate_td = calculate_thermal_desorption_rate(temp)
                total_td += dens * rate_td * area

                # SWS
                in_lon = SWS_PARAMS['LON_RANGE'][0] <= lon_sun <= SWS_PARAMS['LON_RANGE'][1]
                in_lat = (SWS_PARAMS['LAT_N_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_N_RANGE'][1]) or \
                         (SWS_PARAMS['LAT_S_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_S_RANGE'][1])

                if in_lon and in_lat:
                    rate_sws = (sw_flux * SWS_PARAMS['YIELD']) / SWS_PARAMS['REF_DENS']
                    total_sws += dens * rate_sws * area

        result_taa.append(taa_deg)
        rates_td.append(total_td)
        rates_psd.append(total_psd)
        rates_sws.append(total_sws)
        rates_mmv.append(total_mmv)

    # numpy配列化とソート
    result_taa = np.array(result_taa)
    idx = np.argsort(result_taa)

    result_taa = result_taa[idx]
    rates_td = np.array(rates_td)[idx]
    rates_psd = np.array(rates_psd)[idx]
    rates_sws = np.array(rates_sws)[idx]
    rates_mmv = np.array(rates_mmv)[idx]

    total_rate = rates_td + rates_psd + rates_sws + rates_mmv

    # --- グラフ描画 (Leblanc et al. 2003 スタイル) ---
    fig = plt.figure(figsize=(10, 12))

    # --- Figure 1: 絶対量 (Source Rate) ---
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(1, 1, 1)

    ax1.plot(result_taa, rates_td, label='TD (Thermal)', color='red', marker='s', markersize=4, linestyle='--')
    ax1.plot(result_taa, rates_psd, label='PSD (Photon)', color='blue', marker='*', markersize=6, linestyle='-')
    ax1.plot(result_taa, rates_sws, label='SWS (Solar Wind)', color='green', marker='x', markersize=6, linestyle='-')
    ax1.plot(result_taa, rates_mmv, label='MMV (Meteoroid)', color='orange', marker='o', markersize=4, linestyle='--')
    ax1.plot(result_taa, total_rate, label='Total', color='black', linestyle='-', linewidth=2)

    ax1.set_yscale('log')
    ax1.set_title("(a) Total Na Production Rate vs TAA", fontsize=14)
    ax1.set_ylabel("Source Rate [atoms/s]", fontsize=12)
    ax1.set_xlabel("True Anomaly Angle (degrees)", fontsize=12)
    ax1.grid(True, which="both", ls=":", alpha=0.7)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(0, 360)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    fig1.savefig("source_rates_absolute.png", dpi=300)
    print("Graph 1 saved as source_rates_absolute.png")

    # --- Figure 2: 寄与率 (% of Total) ---
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(1, 1, 1)

    # パーセンテージ計算
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_td = (rates_td / total_rate) * 100.0
        frac_psd = (rates_psd / total_rate) * 100.0
        frac_sws = (rates_sws / total_rate) * 100.0
        frac_mmv = (rates_mmv / total_rate) * 100.0

    ax2.plot(result_taa, frac_td, label='Thermal desorption', color='red', marker='s', markersize=5, linestyle='--',
             linewidth=1.5)
    ax2.plot(result_taa, frac_psd, label='Photon stimulated desorption', color='blue', marker='*', markersize=8,
             linestyle='-', linewidth=1.5)
    ax2.plot(result_taa, frac_mmv, label='Meteoroid vaporization', color='orange', marker='o', markersize=5,
             linestyle='--', markerfacecolor='none', linewidth=1.5)
    ax2.plot(result_taa, frac_sws, label='Solar wind sputtering', color='green', marker='x', markersize=7,
             linestyle='-', linewidth=1.5)

    ax2.set_yscale('log')
    ax2.set_title("(b) Relative contribution to total Na ejection", fontsize=14)
    ax2.set_xlabel("True Anomaly Angle (degrees)", fontsize=12)
    ax2.set_ylabel("% of total Na ejection", fontsize=12)
    ax2.set_xlim(0, 360)

    # 縦軸設定 (0.001% ~ 150%)
    ax2.set_ylim(0.001, 150)
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    # 目盛り設定 (細かい値も追加)
    ax2.set_yticks([0.001, 0.01, 0.1, 1, 10, 100])

    ax2.grid(True, which="both", ls=":", alpha=0.7)
    ax2.legend(loc='center right', fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    fig2.savefig("source_rates_fraction.png", dpi=300)
    print("Graph 2 saved as source_rates_fraction.png")

    # 最後にまとめて表示
    plt.show()


if __name__ == "__main__":
    main()