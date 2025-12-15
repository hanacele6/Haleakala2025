# -*- coding: utf-8 -*-
"""
==============================================================================
事後解析ツール (Leblancスタイル完全版):
修正点: MMVを「表面滞留寿命」の計算から除外しました。
       MMVは表面密度に依存しないバルクプロセスであるため、
       表面原子の枯渇タイムスケールには寄与しない物理モデルとしました。
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, NullFormatter
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
TAA_WIDTH = 10.0  # 平均化する幅
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

    # === 夜面 ===
    if cos_theta <= 0:
        return T_BASE

    # === 昼面 ===
    # T = 100 + 600 * scaling * (cos^0.25)
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
    """
    MMVは表面密度に依存しないソースとして計算
    """
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
# 3. 解析機能: 寿命分布 (MMV除外修正版)
# ==============================================================================
def analyze_lifetime_contribution_averaged(target_taa, width, files, orbit_data):
    print(f"\n--- Starting Averaged Lifetime Analysis ---")
    print(f"Target TAA: {target_taa} +/- {width / 2} deg")

    target_files = []
    taa_min = target_taa - width / 2.0
    taa_max = target_taa + width / 2.0
    cross_zero = False
    if taa_min < 0:
        taa_min += 360; cross_zero = True
    elif taa_max > 360:
        taa_max -= 360; cross_zero = True

    print("Scanning files for the TAA range...")
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
        print("Error: No files found in range.")
        return

    print(f"Found {len(target_files)} files in range. Calculating distribution...")

    all_lifetimes = []
    all_prod_td = []
    all_prod_psd = []
    all_prod_sws = []
    all_prod_mmv = []

    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (RM ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    area_grid = np.tile(cell_areas, (N_LON_FIXED, 1))

    for fpath, taa, au, sub_lon in target_files:
        surf_dens = np.load(fpath)
        f_uv = F_UV_1AU / (au ** 2)
        sw_flux_base = SWS_PARAMS['FLUX_1AU'] / (au ** 2)
        mmv_flux_total = calculate_mmv_total_rate(au)
        # MMV flux (atoms/m2/s) - 表面密度非依存
        mmv_flux_per_m2 = mmv_flux_total / (4 * PI * RM ** 2)

        for i_lon in range(N_LON_FIXED):
            lon_f = (lon_edges[i_lon] + lon_edges[i_lon + 1]) / 2
            lon_sun = (lon_f - sub_lon + np.pi) % (2 * np.pi) - np.pi

            for j_lat in range(N_LAT):
                lat_f = (lat_edges[j_lat] + lat_edges[j_lat + 1]) / 2
                area = area_grid[i_lon, j_lat]
                dens = surf_dens[i_lon, j_lat]

                # 昼間側のみ (Cos > 0)
                cos_z = np.cos(lat_f) * np.cos(lon_f - sub_lon)
                if cos_z <= 0: continue

                # --- 表面密度に依存するFlux (Depletion Flux) ---
                # 1. PSD
                flux_psd_val = f_uv * Q_PSD * cos_z * dens  # Ratecoeff * N

                # 2. TD
                temp = calculate_surface_temperature_leblanc(lon_f, lat_f, au, sub_lon)
                rate_td_coeff = calculate_thermal_desorption_rate(temp)
                flux_td_val = dens * rate_td_coeff  # Ratecoeff * N

                # 3. SWS
                flux_sws_val = 0.0
                in_lon = SWS_PARAMS['LON_RANGE'][0] <= lon_sun <= SWS_PARAMS['LON_RANGE'][1]
                in_lat = (SWS_PARAMS['LAT_N_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_N_RANGE'][1]) or \
                         (SWS_PARAMS['LAT_S_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_S_RANGE'][1])
                if in_lon and in_lat:
                    flux_sws_val = dens * ((sw_flux_base * SWS_PARAMS['YIELD']) / SWS_PARAMS['REF_DENS'])

                # ★修正点: 寿命計算に用いるFlux (Loss Flux) に MMV を含めない
                depletion_flux = flux_td_val + flux_psd_val + flux_sws_val

                # --- 表面密度に依存しないFlux (Source Flux) ---
                # 4. MMV (グラフ比較用には必要だが、寿命計算には使わない)
                flux_mmv_val = mmv_flux_per_m2

                # 寿命計算 (Loss Fluxが0なら寿命無限大だが、昼間なのでPSDがあるため0にはならない)
                if depletion_flux <= 0: continue
                if dens <= 0: continue  # 密度0なら計算不可

                lifetime = dens / depletion_flux

                all_lifetimes.append(lifetime)
                all_prod_td.append(flux_td_val * area)
                all_prod_psd.append(flux_psd_val * area)
                all_prod_sws.append(flux_sws_val * area)
                all_prod_mmv.append(flux_mmv_val * area)  # プロット用に保存

    # numpy配列化
    lifetimes = np.array(all_lifetimes)
    prod_td = np.array(all_prod_td)
    prod_psd = np.array(all_prod_psd)
    prod_sws = np.array(all_prod_sws)
    prod_mmv = np.array(all_prod_mmv)

    if len(lifetimes) == 0:
        print("No valid data points found.")
        return

    # ヒストグラム
    min_tau = np.min(lifetimes)
    max_tau = np.max(lifetimes)
    bins = np.logspace(np.floor(np.log10(min_tau)), np.ceil(np.log10(max_tau)), N_BINS)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    indices = np.digitize(lifetimes, bins)

    sum_td = np.zeros(len(bin_centers))
    sum_psd = np.zeros(len(bin_centers))
    sum_sws = np.zeros(len(bin_centers))
    sum_mmv = np.zeros(len(bin_centers))

    for i in range(len(lifetimes)):
        idx = indices[i] - 1
        if 0 <= idx < len(bin_centers):
            sum_td[idx] += prod_td[i]
            sum_psd[idx] += prod_psd[i]
            sum_sws[idx] += prod_sws[i]
            sum_mmv[idx] += prod_mmv[i]

    # 平均化
    num_files = len(target_files)
    sum_td /= num_files
    sum_psd /= num_files
    sum_sws /= num_files
    sum_mmv /= num_files
    total_in_bin = sum_td + sum_psd + sum_sws + sum_mmv

    # プロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # 上段
    ax1.plot(bin_centers, total_in_bin, 'k-', linewidth=2, label='Total Rate')
    ax1.plot(bin_centers, sum_td, 'r--', label='TD')
    ax1.plot(bin_centers, sum_psd, 'b--', label='PSD')
    ax1.plot(bin_centers, sum_sws, 'g--', label='SWS')
    ax1.plot(bin_centers, sum_mmv, 'orange', linestyle=':', label='MMV')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Avg. Production Rate [atoms/s]')
    ax1.set_title(f'Production Distribution vs Surface Residence Time\n(Avg TAA = {target_taa} $\pm$ {width / 2} deg)')
    ax1.grid(True, which="major", ls="-", alpha=0.5)

    # 目盛り
    loc_major = LogLocator(base=10.0, numticks=10)
    loc_minor = LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1), numticks=10)
    ax1.xaxis.set_major_locator(loc_major)
    ax1.xaxis.set_minor_locator(loc_minor)
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.legend()

    # 下段
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
    ax2.set_xlabel('Surface Residence Time (Lifetime) [sec]')
    ax2.set_xlim(bins[0], bins[-1])
    ax2.set_ylim(0, 100)
    ax2.xaxis.set_major_locator(loc_major)
    ax2.xaxis.set_minor_locator(loc_minor)
    ax2.grid(True, which="major", ls="-", alpha=0.5)
    ax2.legend(loc='upper left')

    # 時間ガイド
    for sec, label in [(60, '1m'), (3600, '1h'), (86400, '1d'), (86400 * 30, '1M')]:
        if bins[0] < sec < bins[-1]:
            ax2.axvline(sec, color='k', linestyle='--', alpha=0.3)
            ax2.text(sec, 102, label, ha='center', fontsize=8)

    plt.tight_layout()
    save_name = f"lifetime_dist_avg_taa{int(target_taa)}.png"
    plt.savefig(save_name, dpi=300)
    print(f"Detail Graph saved as {save_name}")
    plt.show()


# ==============================================================================
# 4. メイン処理
# ==============================================================================
def main():
    pattern = os.path.join(TARGET_DIR, "surface_density_*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Error: No files found in {TARGET_DIR}")
        return

    orbit_data = np.loadtxt(ORBIT_FILE)
    orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))

    # Phase 1: Global Time-Evolution (変更なし)
    print("--- Phase 1: Generating Global Time-Evolution Plot ---")

    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (RM ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    area_grid = np.tile(cell_areas, (N_LON_FIXED, 1))

    result_taa = []
    rates_td = [];
    rates_psd = [];
    rates_sws = [];
    rates_mmv = []

    for i, fpath in enumerate(files):
        if i % 100 == 0: print(f"Processing... {i}/{len(files)}")
        match = re.search(r"t(\d+)", fpath)
        if not match: continue
        rel_hours = int(match.group(1))
        taa_deg, au, sub_lon = get_orbital_info_from_time(rel_hours, orbit_data)
        surf_dens = np.load(fpath)

        total_mmv = calculate_mmv_total_rate(au)
        total_td = 0.0;
        total_psd = 0.0;
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
                    total_psd += dens * f_uv * Q_PSD * cos_z * area

                # TD
                temp = calculate_surface_temperature_leblanc(lon_f, lat_f, au, sub_lon)
                total_td += dens * calculate_thermal_desorption_rate(temp) * area

                # SWS
                in_lon = SWS_PARAMS['LON_RANGE'][0] <= lon_sun <= SWS_PARAMS['LON_RANGE'][1]
                in_lat = (SWS_PARAMS['LAT_N_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_N_RANGE'][1]) or \
                         (SWS_PARAMS['LAT_S_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_S_RANGE'][1])
                if in_lon and in_lat:
                    total_sws += dens * ((sw_flux * SWS_PARAMS['YIELD']) / SWS_PARAMS['REF_DENS']) * area

        result_taa.append(taa_deg)
        rates_td.append(total_td);
        rates_psd.append(total_psd)
        rates_sws.append(total_sws);
        rates_mmv.append(total_mmv)

    # Sort & Plot
    result_taa = np.array(result_taa)
    idx = np.argsort(result_taa)
    result_taa = result_taa[idx]
    rates_td = np.array(rates_td)[idx];
    rates_psd = np.array(rates_psd)[idx]
    rates_sws = np.array(rates_sws)[idx];
    rates_mmv = np.array(rates_mmv)[idx]
    total_rate = rates_td + rates_psd + rates_sws + rates_mmv

    # Graph 1
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(result_taa, rates_td, 'r--s', label='TD')
    ax1.plot(result_taa, rates_psd, 'b-*', label='PSD')
    ax1.plot(result_taa, rates_sws, 'g-x', label='SWS')
    ax1.plot(result_taa, rates_mmv, 'o--', color='orange', label='MMV')
    ax1.plot(result_taa, total_rate, 'k-', linewidth=2, label='Total')
    ax1.set_yscale('log')
    ax1.set_title("Total Na Production Rate vs TAA")
    ax1.set_ylabel("Source Rate [atoms/s]")
    ax1.set_xlabel("TAA [deg]")
    ax1.grid(True)
    ax1.legend()
    fig1.savefig("source_rates_absolute.png", dpi=300)

    # Graph 2
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(1, 1, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_td = (rates_td / total_rate) * 100.0
        frac_psd = (rates_psd / total_rate) * 100.0
        frac_sws = (rates_sws / total_rate) * 100.0
        frac_mmv = (rates_mmv / total_rate) * 100.0

    ax2.plot(result_taa, frac_td, 'r--s', label='TD')
    ax2.plot(result_taa, frac_psd, 'b-*', label='PSD')
    ax2.plot(result_taa, frac_sws, 'g-x', label='SWS')
    ax2.plot(result_taa, frac_mmv, 'o--', color='orange', label='MMV')

    ax2.set_yscale('log')
    ax2.set_ylim(0.01, 150)
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.set_title("Relative contribution to total Na ejection")
    ax2.set_ylabel("% of total")
    ax2.set_xlabel("TAA [deg]")
    ax2.grid(True)
    ax2.legend()
    fig2.savefig("source_rates_fraction.png", dpi=300)

    # Phase 2: Detail
    analyze_lifetime_contribution_averaged(TARGET_TAA_CENTER, TAA_WIDTH, files, orbit_data)


if __name__ == "__main__":
    main()