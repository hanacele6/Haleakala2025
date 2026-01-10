# -*- coding: utf-8 -*-
"""
==============================================================================
事後解析ツール: 絶対フラックス(Flux) & 寄与率(Contribution) 可視化ツール
(SWS/MMV 文献値比較版)

アップデート内容:
  1. Killen (2004) Table 1 に基づく SWS, MMV の参照値を追加プロット。
     - SWS: 上限/下限の範囲を表示
     - MMV: 近日点/遠日点の値を表示
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator, NullFormatter
import os
import glob
import re

# ==============================================================================
# 1. 設定・定数
# ==============================================================================
# 解析対象ディレクトリ（適宜変更してください）
TARGET_DIR = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_0109_0.4Denabled_2.7_HalfQ"
ORBIT_FILE = 'orbit2025_spice_unwrapped.txt'

# 解析設定
TARGET_TAA_CENTER = 30.0
TAA_WIDTH = 10.0
N_BINS = 50

# グリッド・物理定数
N_LON_FIXED = 72
N_LAT = 36
PI = np.pi
RM = 2.440e6
SURFACE_AREA_M2 = 4 * PI * RM ** 2  # 水星の全表面積 [m^2]
KB = 1.380649e-23
EV_TO_JOULE = 1.602e-19
ROTATION_PERIOD = 58.6462 * 86400

# ソースパラメータ
F_UV_1AU = 1.5e14 * (100 ** 2)
#Q_PSD = 1.0e-20 / (100 ** 2)
Q_PSD = 5.0e-21 / (100 ** 2)
SWS_PARAMS = {
    'FLUX_1AU': 10.0 * 100 ** 3 * 400e3 * 4.0,
    'YIELD': 0.06,
    'REF_DENS': 7.5e14 * 100 ** 2,
    'LON_RANGE': np.deg2rad([-40, 40]),
    'LAT_N_RANGE': np.deg2rad([20, 80]),
    'LAT_S_RANGE': np.deg2rad([-80, -20]),
}
TEMP_BASE = 100.0
TEMP_AMP = 600.0


# ==============================================================================
# 2. 計算関数群
# ==============================================================================
def calculate_surface_temperature_leblanc(lon_rad, lat_rad, AU, subsolar_lon_rad):
    scaling = np.sqrt(0.306 / AU)
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)
    if cos_theta <= 0: return TEMP_BASE
    return TEMP_BASE + TEMP_AMP * (cos_theta ** 0.25) * scaling


def calculate_thermal_desorption_rate(temp_k):
    """
    固定値モデル (U = 2.7 eV)
    """
    if temp_k < 10.0: return 0.0

    VIB_FREQ = 1e13
    U_EV = 2.6  # ユーザー指定の結合エネルギー
    U_JOULE = U_EV * EV_TO_JOULE

    exponent = -U_JOULE / (KB * temp_k)

    # 数値計算上のアンダーフロー防止
    if exponent < -700:
        return 0.0

    return VIB_FREQ * np.exp(exponent)


def calculate_mmv_total_rate(r_au):
    TOTAL_FLUX_AT_PERI = 5e23
    PERIHELION_AU = 0.307
    avg_flux_peri = TOTAL_FLUX_AT_PERI / (4 * PI * RM ** 2)
    C = avg_flux_peri * (PERIHELION_AU ** 1.9)
    flux_per_m2 = C * (r_au ** (-1.9))
    return flux_per_m2 * (4 * PI * RM ** 2)


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
    taa_deg_normalized = taa_deg_raw % 360.0
    return taa_deg_normalized, au, sub_lon


# ==============================================================================
# 3. メイン処理
# ==============================================================================
def main():
    print("--- Flux Analysis Start ---")

    if not os.path.exists(TARGET_DIR):
        print(f"[ERROR] Directory not found: {TARGET_DIR}")
        # return # パスがない場合でも動作確認用にコメントアウト解除してもよい

    pattern = os.path.join(TARGET_DIR, "surface_density_*.npy")
    files = sorted(glob.glob(pattern))
    print(f"[Debug] Found {len(files)} surface density files.")

    if not files:
        print("[ERROR] No files found.")
        return

    if not os.path.exists(ORBIT_FILE):
        print(f"[ERROR] Orbit file not found: {ORBIT_FILE}")
        return

    orbit_data = np.loadtxt(ORBIT_FILE)
    orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))

    # データ格納用
    result_taa = []
    flux_td_cm2 = []
    flux_psd_cm2 = []
    flux_sws_cm2 = []
    flux_mmv_cm2 = []

    rate_total_td = []
    rate_total_psd = []
    rate_total_sws = []
    rate_total_mmv = []

    # グリッド面積計算
    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (RM ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    area_grid = np.tile(cell_areas, (N_LON_FIXED, 1))

    total_files = len(files)
    print("Calculating fluxes...")

    for i, fpath in enumerate(files):
        if i % 50 == 0:
            print(f"Processing {i}/{total_files} ...")

        match = re.search(r"t(\d+)", fpath)
        if not match: continue

        try:
            taa_deg, au, sub_lon = get_orbital_info_from_time(int(match.group(1)), orbit_data)
            surf_dens = np.load(fpath)
        except Exception as e:
            continue

        total_mmv_rate = calculate_mmv_total_rate(au)
        sum_td = 0.0
        sum_psd = 0.0
        sum_sws = 0.0

        f_uv = F_UV_1AU / (au ** 2)
        sw_flux = SWS_PARAMS['FLUX_1AU'] / (au ** 2)

        for ix in range(N_LON_FIXED):
            lon_f = (lon_edges[ix] + lon_edges[ix + 1]) / 2
            lon_sun = (lon_f - sub_lon + np.pi) % (2 * np.pi) - np.pi
            for iy in range(N_LAT):
                lat_f = (lat_edges[iy] + lat_edges[iy + 1]) / 2
                area = area_grid[ix, iy]
                dens = surf_dens[ix, iy]
                cos_z = np.cos(lat_f) * np.cos(lon_f - sub_lon)

                if cos_z > 0:
                    sum_psd += dens * f_uv * Q_PSD * cos_z * area

                temp = calculate_surface_temperature_leblanc(lon_f, lat_f, au, sub_lon)
                sum_td += dens * calculate_thermal_desorption_rate(temp) * area

                in_lon = SWS_PARAMS['LON_RANGE'][0] <= lon_sun <= SWS_PARAMS['LON_RANGE'][1]
                in_lat = (SWS_PARAMS['LAT_N_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_N_RANGE'][1]) or \
                         (SWS_PARAMS['LAT_S_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_S_RANGE'][1])
                if in_lon and in_lat:
                    sum_sws += dens * ((sw_flux * SWS_PARAMS['YIELD']) / SWS_PARAMS['REF_DENS']) * area

        result_taa.append(taa_deg)

        rate_total_td.append(sum_td)
        rate_total_psd.append(sum_psd)
        rate_total_sws.append(sum_sws)
        rate_total_mmv.append(total_mmv_rate)

        surface_area_cm2 = SURFACE_AREA_M2 * 1e4
        flux_td_cm2.append(sum_td / surface_area_cm2)
        flux_psd_cm2.append(sum_psd / surface_area_cm2)
        flux_sws_cm2.append(sum_sws / surface_area_cm2)
        flux_mmv_cm2.append(total_mmv_rate / surface_area_cm2)

    # ソート
    idx = np.argsort(result_taa)
    result_taa = np.array(result_taa)[idx]

    rate_td = np.array(rate_total_td)[idx]
    rate_psd = np.array(rate_total_psd)[idx]
    rate_sws = np.array(rate_total_sws)[idx]
    rate_mmv = np.array(rate_total_mmv)[idx]

    flux_td = np.array(flux_td_cm2)[idx]
    flux_psd = np.array(flux_psd_cm2)[idx]
    flux_sws = np.array(flux_sws_cm2)[idx]
    flux_mmv = np.array(flux_mmv_cm2)[idx]

    # ==========================================================================
    # グラフ1: 全球平均フラックスの対数推移 (Literature Comparison)
    # ==========================================================================
    print("Generating Log-Flux Plot...")
    plt.figure(figsize=(10, 6))

    # Simulation Results
    plt.plot(result_taa, flux_psd, label='Sim: PSD', color='blue', linewidth=2)
    plt.plot(result_taa, flux_td, label='Sim: TD (U=2.7eV)', color='red', linewidth=2)
    plt.plot(result_taa, flux_mmv, label='Sim: MMV', color='orange', linestyle='--')
    plt.plot(result_taa, flux_sws, label='Sim: SWS', color='green', linestyle=':')

    # --- Literature References (Killen 2004 Table 1) ---

    # 1. PSD Reference (Perihelion Max)
    # Table 1: 6.0e7
    plt.scatter([0, 360], [6.0e7, 6.0e7], color='blue', marker='x', s=100, zorder=5,
                label='Ref: PSD Max (6e7)')

    # 2. TD/Impact Comparison Reference (U=2.7eV)
    # Text mentions ~3.0e5 for U=2.7eV sites
    plt.scatter([0, 360], [3.0e5, 3.0e5], color='red', marker='x', s=100, zorder=5,
                label='Ref: TD U=2.7 (3e5)')

    # 3. SWS References (Upper/Lower Limits)
    # Table 1: Upper 3.5e7, Lower 3.5e5
    # 帯として表示
    #plt.axhspan(3.5e5, 3.5e7, color='green', alpha=0.1, label='Ref: SWS Range (Killen 04)')
    #plt.hlines(3.5e7, 0, 360, colors='green', linestyles='-.', alpha=0.5)
    #plt.hlines(3.5e5, 0, 360, colors='green', linestyles='-.', alpha=0.5)

    # 4. MMV References (Perihelion/Aphelion)
    # Table 1: Perihelion 6.0e5, Aphelion 2.4e5
    plt.scatter([0, 360], [6.0e5, 6.0e5], color='orange', marker='D', s=80, zorder=5,
                label='Ref: MMV Peri (6e5)')
    plt.scatter([180], [2.4e5], color='orange', marker='s', s=80, zorder=5,
                label='Ref: MMV Aph (2.4e5)')

    plt.yscale('log')
    plt.ylim(1e4, 1e9)
    plt.xlim(0, 360)

    plt.xlabel("True Anomaly Angle (TAA) [deg]", fontsize=12)
    plt.ylabel("Global Avg Flux [atoms cm$^{-2}$ s$^{-1}$]", fontsize=12)
    plt.title(f"Source Flux: Simulation vs Killen (2004) References", fontsize=14)
    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.grid(True, which="minor", ls=":", alpha=0.3)

    # 凡例の位置調整
    plt.legend(loc='upper right', ncol=2, fontsize=9)

    plt.tight_layout()
    plt.savefig("source_flux_logscale_with_ref.png", dpi=300)
    print("Saved: source_flux_logscale_with_ref.png")

    # ==========================================================================
    # グラフ2: 寄与率(%) の推移 (既存機能)
    # ==========================================================================
    print("Generating Contribution Rate Plot...")
    plt.figure(figsize=(10, 6))

    total_rate = rate_td + rate_psd + rate_sws + rate_mmv
    with np.errstate(divide='ignore', invalid='ignore'):
        p_td = np.nan_to_num(rate_td / total_rate * 100)
        p_psd = np.nan_to_num(rate_psd / total_rate * 100)
        p_sws = np.nan_to_num(rate_sws / total_rate * 100)
        p_mmv = np.nan_to_num(rate_mmv / total_rate * 100)

    plt.stackplot(result_taa, p_td, p_psd, p_sws, p_mmv,
                  labels=['TD', 'PSD', 'SWS', 'MMV'],
                  colors=['red', 'blue', 'green', 'orange'], alpha=0.8)

    plt.xlim(0, 360)
    plt.ylim(0, 100)
    plt.xlabel("True Anomaly Angle (TAA) [deg]", fontsize=12)
    plt.ylabel("Contribution Ratio [%]", fontsize=12)
    plt.title("Source Contribution Ratio (Linear Scale)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.savefig("source_contribution_evolution.png", dpi=300)
    print("Saved: source_contribution_evolution.png")

    print("All tasks finished.")
    plt.show() # 必要に応じてコメントアウト解除


if __name__ == "__main__":
    main()