# -*- coding: utf-8 -*-
"""
==============================================================================
事後解析ツール (Leblancスタイル完全版 + 2Dマップ平均化 + SZA累積グラフ版):
1. TAAごとの時系列推移 (Time-Evolution) - ※このスクリプトには含まれません
2. 枯渇タイムスケール分布 (Histogram)
   ★追加: 天頂角解析 (上段: 寿命 vs SZA, 下段: 累積寄与率 vs SZA)
3. Local Time vs Latitude 2Dマップ (TAA範囲平均 + 全域色付け)

【修正点】
・Unwrappedな軌道データ（360度を超えるTAA）に対応するため、
  get_orbital_info_from_time の返り値でTAAを % 360.0 しています。
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
TARGET_DIR = r"./SimulationResult_202512/DynamicGrid72x36_EqMode_Hard_DT500_T0100_4.0"
ORBIT_FILE = 'orbit2025_spice_unwrapped.txt'

# --- ★解析設定 ---
TARGET_TAA_CENTER = 30.0  # 解析したいTAAの中心
TAA_WIDTH = 10.0  # 平均化する幅 (+/- 5 deg)
N_BINS = 50  # ヒストグラムの分割数
N_BINS_ANGLE = 45  # 天頂角プロットの分割数 (90度を45分割=2度刻み)

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
    """
    時刻(rel_hours)から軌道情報を取得する。
    Unwrappedなデータに対応するため、TAAの返り値を360で割った余りに正規化する。
    """
    idx_peri = np.argmin(np.abs(orbit_data[:, 0]))
    t_peri = orbit_data[idx_peri, 2]
    current_time = t_peri + rel_hours * 3600.0
    time_col = orbit_data[:, 2]

    # 時間で検索 (Unwrappedかどうかに依存しない)
    idx = np.searchsorted(time_col, current_time)
    if idx >= len(time_col): idx = len(time_col) - 1
    if idx > 0 and abs(current_time - time_col[idx - 1]) < abs(current_time - time_col[idx]):
        idx -= 1

    au = orbit_data[idx, 1]
    taa_deg_raw = orbit_data[idx, 0]  # これは360を超える可能性がある

    # Sub-solar longitude計算 (ここでは連続性が重要なのでrawを使うか、回転角との差分をとる)
    omega_rot = 2 * np.pi / ROTATION_PERIOD
    rotation_angle = omega_rot * (current_time - t_peri)
    taa_rad = np.deg2rad(taa_deg_raw)
    sub_lon = taa_rad - rotation_angle
    # -pi ~ pi に正規化
    sub_lon = (sub_lon + np.pi) % (2 * np.pi) - np.pi

    # ★重要修正: 呼び出し元のフィルタリング用には 0-360 の値を返す
    taa_deg_normalized = taa_deg_raw % 360.0

    return taa_deg_normalized, au, sub_lon


# ==============================================================================
# 3. 解析機能: ヒストグラム & SZA依存性 (2ウィンドウ)
# ==============================================================================
def analyze_lifetime_contribution_averaged(target_taa, width, files, orbit_data):
    print(f"\n--- Starting Averaged Lifetime Analysis (Histogram & SZA) ---")
    print(f"Target TAA: {target_taa} +/- {width / 2} deg")

    # --- ファイル収集 ---
    target_files = []
    # ターゲット範囲の設定 (0-360空間)
    taa_min = target_taa - width / 2.0
    taa_max = target_taa + width / 2.0
    cross_zero = False
    if taa_min < 0:
        taa_min += 360
        cross_zero = True
    elif taa_max > 360:
        taa_max -= 360
        cross_zero = True

    for fpath in files:
        match = re.search(r"t(\d+)", fpath)
        if not match: continue
        rel_hours = int(match.group(1))

        # ここで taa は 0-360 に正規化されて返ってくる
        taa, au, sub_lon = get_orbital_info_from_time(rel_hours, orbit_data)

        hit = False
        if not cross_zero:
            if taa_min <= taa <= taa_max: hit = True
        else:
            if taa >= taa_min or taa <= taa_max: hit = True

        if hit:
            target_files.append((fpath, taa, au, sub_lon))

    if not target_files:
        print("No files found in range.")
        return

    print(f"Files found: {len(target_files)}")

    # データ格納用リスト
    all_lifetimes = []
    all_sza = []
    all_prod_td = []
    all_prod_psd = []
    all_prod_sws = []
    all_prod_mmv = []

    # SZAビン定義
    bins_sza = np.linspace(0, 90, N_BINS_ANGLE + 1)

    # 損失率の平均計算用
    k_td_sum_sza = np.zeros(N_BINS_ANGLE)
    k_psd_sum_sza = np.zeros(N_BINS_ANGLE)
    k_sws_sum_sza = np.zeros(N_BINS_ANGLE)
    count_sza = np.zeros(N_BINS_ANGLE)

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

                if cos_z <= 0: continue  # Daysideのみ

                # Loss Rates
                k_psd_val = f_uv * Q_PSD * cos_z
                flux_psd_val = k_psd_val * dens

                temp = calculate_surface_temperature_leblanc(lon_f, lat_f, au, sub_lon)
                k_td_val = calculate_thermal_desorption_rate(temp)
                flux_td_val = k_td_val * dens

                k_sws_val = 0.0
                in_lon = SWS_PARAMS['LON_RANGE'][0] <= lon_sun <= SWS_PARAMS['LON_RANGE'][1]
                in_lat = (SWS_PARAMS['LAT_N_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_N_RANGE'][1]) or \
                         (SWS_PARAMS['LAT_S_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_S_RANGE'][1])
                if in_lon and in_lat:
                    k_sws_val = (sw_flux_base * SWS_PARAMS['YIELD']) / SWS_PARAMS['REF_DENS']
                flux_sws_val = k_sws_val * dens

                flux_mmv_val = mmv_flux_per_m2

                k_total = k_td_val + k_psd_val + k_sws_val
                depletion_flux = flux_td_val + flux_psd_val + flux_sws_val

                if depletion_flux <= 0 or dens <= 0: continue
                lifetime = 1.0 / k_total

                sza_val = np.degrees(np.arccos(cos_z))

                # Store for Histograms
                all_lifetimes.append(lifetime)
                all_sza.append(sza_val)
                all_prod_td.append(flux_td_val * area)
                all_prod_psd.append(flux_psd_val * area)
                all_prod_sws.append(flux_sws_val * area)
                all_prod_mmv.append(flux_mmv_val * area)

                # Store for Lifetime vs SZA Plot
                bin_idx = np.digitize(sza_val, bins_sza) - 1
                if 0 <= bin_idx < N_BINS_ANGLE:
                    k_td_sum_sza[bin_idx] += k_td_val
                    k_psd_sum_sza[bin_idx] += k_psd_val
                    k_sws_sum_sza[bin_idx] += k_sws_val
                    count_sza[bin_idx] += 1

    # --- 集計開始 ---
    lifetimes = np.array(all_lifetimes)
    sza_array = np.array(all_sza)
    prod_td = np.array(all_prod_td)
    prod_psd = np.array(all_prod_psd)
    prod_sws = np.array(all_prod_sws)
    prod_mmv = np.array(all_prod_mmv)

    if len(lifetimes) == 0: return
    num_files = len(target_files)

    # 1. Lifetime Histogram (平均生成率 & 寄与率)
    min_tau = np.min(lifetimes)
    max_tau = np.max(lifetimes)
    bins_tau = np.logspace(np.floor(np.log10(min_tau)), np.ceil(np.log10(max_tau)), N_BINS)
    bin_centers_tau = np.sqrt(bins_tau[:-1] * bins_tau[1:])
    indices_tau = np.digitize(lifetimes, bins_tau)

    sum_td_tau = np.zeros(len(bin_centers_tau))
    sum_psd_tau = np.zeros(len(bin_centers_tau))
    sum_sws_tau = np.zeros(len(bin_centers_tau))
    sum_mmv_tau = np.zeros(len(bin_centers_tau))

    for i in range(len(lifetimes)):
        idx = indices_tau[i] - 1
        if 0 <= idx < len(bin_centers_tau):
            sum_td_tau[idx] += prod_td[i]
            sum_psd_tau[idx] += prod_psd[i]
            sum_sws_tau[idx] += prod_sws[i]
            sum_mmv_tau[idx] += prod_mmv[i]

    sum_td_tau /= num_files
    sum_psd_tau /= num_files
    sum_sws_tau /= num_files
    sum_mmv_tau /= num_files
    total_tau = sum_td_tau + sum_psd_tau + sum_sws_tau + sum_mmv_tau

    with np.errstate(divide='ignore', invalid='ignore'):
        frac_td_tau = np.nan_to_num(sum_td_tau / total_tau * 100)
        frac_psd_tau = np.nan_to_num(sum_psd_tau / total_tau * 100)
        frac_sws_tau = np.nan_to_num(sum_sws_tau / total_tau * 100)
        frac_mmv_tau = np.nan_to_num(sum_mmv_tau / total_tau * 100)

    # 2. SZA Analysis
    indices_sza = np.digitize(sza_array, bins_sza)

    # (A) Cumulative Contribution Calculation (★累積グラフ用)
    sum_td_sza = np.zeros(N_BINS_ANGLE)
    sum_psd_sza = np.zeros(N_BINS_ANGLE)
    sum_sws_sza = np.zeros(N_BINS_ANGLE)
    sum_mmv_sza = np.zeros(N_BINS_ANGLE)

    for i in range(len(lifetimes)):
        idx = indices_sza[i] - 1
        if 0 <= idx < N_BINS_ANGLE:
            sum_td_sza[idx] += prod_td[i]
            sum_psd_sza[idx] += prod_psd[i]
            sum_sws_sza[idx] += prod_sws[i]
            sum_mmv_sza[idx] += prod_mmv[i]

    # 時間平均
    sum_td_sza /= num_files
    sum_psd_sza /= num_files
    sum_sws_sza /= num_files
    sum_mmv_sza /= num_files

    # 累積和 (Cumulative Sum)
    cum_td = np.cumsum(sum_td_sza)
    cum_psd = np.cumsum(sum_psd_sza)
    cum_sws = np.cumsum(sum_sws_sza)
    cum_mmv = np.cumsum(sum_mmv_sza)

    # 総計 (全てのビンの合計 = 最後の累積値の合計)
    grand_total_rate = cum_td[-1] + cum_psd[-1] + cum_sws[-1] + cum_mmv[-1]

    # %変換 (全体に対する累積寄与率)
    if grand_total_rate > 0:
        y_cum_td = cum_td / grand_total_rate * 100
        y_cum_psd = cum_psd / grand_total_rate * 100
        y_cum_sws = cum_sws / grand_total_rate * 100
        y_cum_mmv = cum_mmv / grand_total_rate * 100
    else:
        y_cum_td = np.zeros_like(cum_td)
        y_cum_psd = np.zeros_like(cum_psd)
        y_cum_sws = np.zeros_like(cum_sws)
        y_cum_mmv = np.zeros_like(cum_mmv)

    # (B) Lifetime vs SZA Calculation (平均)
    bin_centers_sza = (bins_sza[:-1] + bins_sza[1:]) / 2  # 上段のグラフ用

    avg_k_td = np.zeros_like(count_sza)
    avg_k_psd = np.zeros_like(count_sza)
    avg_k_sws = np.zeros_like(count_sza)

    mask = count_sza > 0
    avg_k_td[mask] = k_td_sum_sza[mask] / count_sza[mask]
    avg_k_psd[mask] = k_psd_sum_sza[mask] / count_sza[mask]
    avg_k_sws[mask] = k_sws_sum_sza[mask] / count_sza[mask]

    avg_k_total = avg_k_td + avg_k_psd + avg_k_sws

    def safe_inv(val_arr):
        res = np.full_like(val_arr, np.nan)
        m = val_arr > 1e-30
        res[m] = 1.0 / val_arr[m]
        return res

    tau_sza_total = safe_inv(avg_k_total)
    tau_sza_td = safe_inv(avg_k_td)
    tau_sza_psd = safe_inv(avg_k_psd)
    tau_sza_sws = safe_inv(avg_k_sws)

    # ==========================================================================
    # 描画処理
    # ==========================================================================

    # --- Window 1: Lifetime Histogram ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax1.plot(bin_centers_tau, total_tau, 'k-', linewidth=2, label='Total')
    ax1.plot(bin_centers_tau, sum_td_tau, 'r--', label='TD')
    ax1.plot(bin_centers_tau, sum_psd_tau, 'b--', label='PSD')
    ax1.plot(bin_centers_tau, sum_sws_tau, 'g--', label='SWS')
    ax1.plot(bin_centers_tau, sum_mmv_tau, 'orange', linestyle=':', label='MMV')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Avg. Production Rate [atoms/s]')
    ax1.set_title(f"Production Rate & Contribution (TAA {target_taa} +/- {width / 2})")
    ax1.grid(True, alpha=0.5)
    ax1.legend()

    ax2.stackplot(bin_centers_tau, frac_td_tau, frac_psd_tau, frac_sws_tau, frac_mmv_tau,
                  labels=['TD', 'PSD', 'SWS', 'MMV'],
                  colors=['red', 'blue', 'green', 'orange'], alpha=0.6)
    ax2.set_xscale('log')
    ax2.set_ylabel('Contribution [%]')
    ax2.set_xlabel('Lifetime [sec]')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.5)
    ax2.legend(loc='upper left')

    plt.figure(fig1.number)
    plt.tight_layout()
    plt.savefig(f"lifetime_dist_avg_taa{int(target_taa)}.png", dpi=300)
    print("Histogram (Window 1) saved.")

    # --- Window 2: SZA Analysis (2段) ---
    fig2, (ax_sza_1, ax_sza_2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # 上段: Lifetime vs SZA (平均)
    ax_sza_1.plot(bin_centers_sza, tau_sza_total, 'k-', linewidth=2, label='Total Lifetime')
    ax_sza_1.plot(bin_centers_sza, tau_sza_td, 'r--', label='TD Time Scale')
    ax_sza_1.plot(bin_centers_sza, tau_sza_psd, 'b--', label='PSD Time Scale')
    ax_sza_1.plot(bin_centers_sza, tau_sza_sws, 'g--', label='SWS Time Scale')

    ax_sza_1.set_yscale('log')
    ax_sza_1.set_ylabel('Time Scale [sec]')
    ax_sza_1.set_title(f"Time Scale vs SZA (TAA {target_taa})")
    ax_sza_1.grid(True, alpha=0.5, which="both")
    ax_sza_1.legend(loc='upper left')

    # 下段: Cumulative Contribution vs SZA (★累積グラフ)
    # X軸はビンの右端 (0度からX度までの累積)
    x_cum = bins_sza[1:]

    ax_sza_2.stackplot(x_cum, y_cum_td, y_cum_psd, y_cum_sws, y_cum_mmv,
                       labels=['TD', 'PSD', 'SWS', 'MMV'],
                       colors=['red', 'blue', 'green', 'orange'], alpha=0.6)

    ax_sza_2.set_ylabel('Cumulative Contribution [%]')
    ax_sza_2.set_xlabel('Solar Zenith Angle (accumulated from 0) [deg]')
    ax_sza_2.set_xlim(0, 90)
    ax_sza_2.set_xticks(np.arange(0, 91, 10))
    ax_sza_2.set_ylim(0, 100)  # 累積なので必ず100%に達する

    ax_sza_2.grid(True, alpha=0.5)
    ax_sza_2.legend(loc='upper left')

    plt.figure(fig2.number)
    plt.tight_layout()
    plt.savefig(f"sza_analysis_taa{int(target_taa)}.png", dpi=300)
    print("SZA Analysis (Window 2) saved.")

    plt.show()


# ==============================================================================
# 4. 2Dマップ (変更なし)
# ==============================================================================
def analyze_lifetime_2d_map_averaged(target_taa, width, files, orbit_data):
    print(f"\n--- Generating Averaged 2D Map for TAA {target_taa} +/- {width / 2} deg ---")

    target_files = []
    taa_min = target_taa - width / 2.0
    taa_max = target_taa + width / 2.0
    cross_zero = False
    if taa_min < 0:
        taa_min += 360
        cross_zero = True
    elif taa_max > 360:
        taa_max -= 360
        cross_zero = True

    for fpath in files:
        match = re.search(r"t(\d+)", fpath)
        if not match: continue
        rel_hours = int(match.group(1))

        # 修正: 正規化されたTAAを使用
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

    common_lt_axis = np.linspace(0, 24, N_LON_FIXED, endpoint=False)
    sum_lifetime_grid = np.zeros((N_LON_FIXED, N_LAT))
    sum_contrib_grid = np.zeros((N_LON_FIXED, N_LAT))

    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lat_deg = np.rad2deg(lat_centers)

    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (RM ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    area_grid = np.tile(cell_areas, (N_LON_FIXED, 1))

    for fpath, taa, au, sub_lon in target_files:
        surf_dens = np.load(fpath)
        f_uv = F_UV_1AU / (au ** 2)
        sw_flux_base = SWS_PARAMS['FLUX_1AU'] / (au ** 2)
        mmv_flux_total = calculate_mmv_total_rate(au)
        mmv_flux_per_m2 = mmv_flux_total / (4 * PI * RM ** 2)

        temp_lifetime = np.zeros((N_LON_FIXED, N_LAT))
        temp_prod_rate = np.zeros((N_LON_FIXED, N_LAT))

        for i_lon in range(N_LON_FIXED):
            lon_f = lon_centers[i_lon]
            lon_sun = (lon_f - sub_lon + np.pi) % (2 * np.pi) - np.pi
            for j_lat in range(N_LAT):
                lat_f = lat_centers[j_lat]
                dens = surf_dens[i_lon, j_lat]
                area = area_grid[i_lon, j_lat]
                cos_z = np.cos(lat_f) * np.cos(lon_f - sub_lon)

                flux_psd = f_uv * Q_PSD * cos_z * dens if cos_z > 0 else 0.0
                temp = calculate_surface_temperature_leblanc(lon_f, lat_f, au, sub_lon)
                flux_td = dens * calculate_thermal_desorption_rate(temp)

                flux_sws_val = 0.0
                in_lon = SWS_PARAMS['LON_RANGE'][0] <= lon_sun <= SWS_PARAMS['LON_RANGE'][1]
                in_lat = (SWS_PARAMS['LAT_N_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_N_RANGE'][1]) or \
                         (SWS_PARAMS['LAT_S_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_S_RANGE'][1])
                if in_lon and in_lat:
                    flux_sws_val = dens * ((sw_flux_base * SWS_PARAMS['YIELD']) / SWS_PARAMS['REF_DENS'])

                flux_mmv = mmv_flux_per_m2
                total_flux = flux_td + flux_psd + flux_sws_val + flux_mmv
                depletion = flux_td + flux_psd + flux_sws_val

                temp_prod_rate[i_lon, j_lat] = total_flux * area
                if depletion > 0 and dens > 0:
                    temp_lifetime[i_lon, j_lat] = dens / depletion
                else:
                    temp_lifetime[i_lon, j_lat] = 0.0

        total_rate_global = np.sum(temp_prod_rate)
        temp_contrib = np.zeros_like(temp_prod_rate)
        if total_rate_global > 0:
            temp_contrib = (temp_prod_rate / total_rate_global) * 100.0

        current_lt_1d = (12.0 + ((lon_centers - sub_lon + np.pi) % (2 * np.pi) - np.pi) / np.pi * 12.0) % 24.0
        sort_idx = np.argsort(current_lt_1d)
        sorted_lt = current_lt_1d[sort_idx]
        extended_lt = np.concatenate([sorted_lt, [sorted_lt[0] + 24.0]])

        for j in range(N_LAT):
            data_row_lt = temp_lifetime[sort_idx, j]
            extended_data_lt = np.concatenate([data_row_lt, [data_row_lt[0]]])
            interp_lt = np.interp(common_lt_axis, extended_lt, extended_data_lt)
            sum_lifetime_grid[:, j] += interp_lt

            data_row_con = temp_contrib[sort_idx, j]
            extended_data_con = np.concatenate([data_row_con, [data_row_con[0]]])
            interp_con = np.interp(common_lt_axis, extended_lt, extended_data_con)
            sum_contrib_grid[:, j] += interp_con

    avg_lifetime_map = sum_lifetime_grid / len(target_files)
    avg_contrib_map = sum_contrib_grid / len(target_files)

    plot_lt = np.concatenate([common_lt_axis, [24.0]])
    plot_lifetime = np.vstack([avg_lifetime_map, avg_lifetime_map[0:1, :]])
    plot_contrib = np.vstack([avg_contrib_map, avg_contrib_map[0:1, :]])
    LT_GRID, LAT_GRID = np.meshgrid(plot_lt, lat_deg, indexing='ij')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    plot_data_lt = plot_lifetime.copy()
    plot_data_lt[plot_data_lt <= 0] = 1e-5
    mesh1 = ax1.pcolormesh(LT_GRID, LAT_GRID, plot_data_lt,
                           norm=LogNorm(vmin=1e-1, vmax=1e7),
                           cmap='jet', shading='auto')
    cb1 = plt.colorbar(mesh1, ax=ax1)
    cb1.set_label('Avg Surface Residence Time [sec]')
    ax1.set_title(f"Averaged Residence Time (TAA {target_taa} +/- {width / 2})")
    ax1.set_ylabel("Latitude [deg]")
    ax1.set_xlim(0, 24)
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.set_ylim(-90, 90)

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
    ax2.set_xlim(0, 24)
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

    # ここでUnwrap処理が入っていますが、入力ファイルが既にUnwrappedならそのまま動作します。
    # Wrapされた（0-360リセットあり）データの場合は連続化されます。
    orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))

    analyze_lifetime_contribution_averaged(TARGET_TAA_CENTER, TAA_WIDTH, files, orbit_data)
    analyze_lifetime_2d_map_averaged(TARGET_TAA_CENTER, TAA_WIDTH, files, orbit_data)


if __name__ == "__main__":
    main()