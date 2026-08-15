# -*- coding: utf-8 -*-
"""
放出位置の解析(実測のみ・再構成なし)

band_statistics_per_taa.csv の実測 Gen_PSD / Gen_TD (eff_cosバンド × TAA) を使って:

  [1] 放出位置の重心 (PSD/TD, Q比較)
  [2] TAAごとの天頂角に対する放出量プロファイル (PSD・TD別々) ※線形スケール
  [3] TAAごとの天頂角に対する平均表面密度プロファイル (生の.npyから)
  [4] PSD vs TD 重ね比較。TAAごとにサブプロットを分け、PSDとTDを線形スケールで比較。
      背景に「TDが主力の領域」を陰影で示し、下段にTD寄与率も表示。
  ★追加: [2]と[4]に「近日点でDawnターミネーターだった地点」の現在のeff_cos位置を追跡プロット
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
from matplotlib.ticker import MultipleLocator
import os
import glob
import re

# ==========================================
# 設定
# ==========================================
BASE_DIR = r"./SimulationResult_202607"

MODELS = {
    "Q2.0 (Standard)": "ParabolicHop_72x36_NoEq_DT100_0716_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr_test2",
    "Q0.3 (Weak PSD)": "ParabolicHop_72x36_NoEq_DT100_0716_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr_test2",
}
COLORS = {"Q2.0 (Standard)": "crimson", "Q0.3 (Weak PSD)": "steelblue"}

SIDE = "Dawn"          # "Dawn" / "Dusk"
SMOOTH_WINDOW_DEG = 3  # TAA方向の移動平均窓[deg]。0でなし。

# 天頂角プロファイルで重ねるTAA
PROFILE_TAAS = [120, 140, 160, 180]

ORBIT_FILE_PATH = 'orbit2025_spice_unwrapped.txt'
N_LON, N_LAT = 72, 36
R_BODY_KM = 2439.7
ZENITH_N_BINS = 20
ORBIT_E = 0.20563  # 軌道離心率


# ==========================================
# 軌道力学: 近日点Dawn地点の追跡
# ==========================================
def get_perihelion_dawn_effcos(taa_deg, e=ORBIT_E):
    """
    指定したTAAにおいて、「近日点(TAA=0)でDawnターミネーター(eff_cos=0)だった経度」が
    現在の太陽天頂角余弦(eff_cos)でどこにいるかを計算する。
    """
    taa_rad = np.radians(taa_deg)
    
    # 離心近点角 (E) を安全に計算 (0 ~ 2pi)
    E_rad = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(taa_rad / 2), 
                           np.sqrt(1 + e) * np.cos(taa_rad / 2))
    
    # 平均近点角 (M)
    M_rad = E_rad - e * np.sin(E_rad)
    
    # 太陽直下点(SSP)の経度 (3:2共鳴)
    ssp_lon_rad = taa_rad - 1.5 * M_rad
    
    # 近日点でのDawnターミネーター地点の経度は -90度 (-pi/2)
    lon_dawn_rad = np.radians(-90)
    
    # 現在のSZAとeff_cos
    sza_dawn_rad = lon_dawn_rad - ssp_lon_rad
    return np.cos(sza_dawn_rad)


# ==========================================
# 読み込み (CSV)
# ==========================================
def load_gen(model_dir, side):
    path = os.path.join(model_dir, "band_statistics_per_taa.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"見つかりません: {path}")
    df = pd.read_csv(path)

    gen_psd_col = f"Gen_PSD_{side}"
    gen_td_col = f"Gen_TD_{side}"

    bands = df[['Band_Index', 'EffCos_Lo', 'EffCos_Hi']].drop_duplicates().sort_values('Band_Index')
    band_centers = ((bands['EffCos_Lo'] + bands['EffCos_Hi']) / 2.0).values
    band_labels = [f"{lo:.2f}–{hi:.2f}" for lo, hi in zip(bands['EffCos_Lo'], bands['EffCos_Hi'])]

    def grid(col):
        return df.pivot(index='TAA_Bin', columns='Band_Index', values=col).sort_index().values

    return {
        'gen_psd': grid(gen_psd_col),
        'gen_td': grid(gen_td_col),
        'band_centers': band_centers,
        'band_labels': band_labels,
        'taa': np.sort(df['TAA_Bin'].unique()),
        'n_bands': len(band_centers),
    }

def circular_smooth(y, w):
    if w <= 0: return y
    w = int(w)
    k = np.ones(w) / w
    ext = np.concatenate([y[-w:], y, y[:w]])
    return np.convolve(ext, k, mode='same')[w:-w]

def smooth_2d(arr, w):
    if w <= 0: return arr
    out = np.empty_like(arr)
    for k in range(arr.shape[1]):
        out[:, k] = circular_smooth(arr[:, k], w)
    return out


# ==========================================
# 読み込み (グリッド・軌道)
# ==========================================
def load_orbit_data(orbit_file_path):
    try:
        orbit_data = np.loadtxt(orbit_file_path)
        orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))
        t_file_start = orbit_data[0, 2]
        return orbit_data, t_file_start
    except Exception as e:
        print(f"軌道ファイル読み込みエラー: {e}")
        return None, None

def get_subsolar_longitude_linear(time_h, t_file_start, orbit_data):
    time_col = orbit_data[:, 2]
    current_t = t_file_start + (float(time_h) * 3600.0)
    t_lookup = np.clip(current_t, time_col[0], time_col[-1])
    return np.interp(t_lookup, time_col, orbit_data[:, 5])

def calculate_cell_areas(n_lon, n_lat, r_body_km):
    r_body_cm = r_body_km * 1e5
    dlon_rad = 2.0 * np.pi / n_lon
    lat_edges = np.linspace(-np.pi/2, np.pi/2, n_lat + 1)
    areas_m2 = np.zeros((n_lat, n_lon))
    areas_cm2 = np.zeros((n_lat, n_lon))
    for i in range(n_lat):
        factor = np.sin(lat_edges[i+1]) - np.sin(lat_edges[i])
        areas_cm2[i, :] = (r_body_cm ** 2) * dlon_rad * factor
        areas_m2[i, :] = (r_body_km * 1e3) ** 2 * dlon_rad * factor
    return areas_m2, areas_cm2

def compute_effcos_grid(n_lon, n_lat, subsolar_lon_deg):
    lon_centers = np.linspace(-180, 180, n_lon + 1)
    lon_centers = (lon_centers[:-1] + lon_centers[1:]) / 2.0
    lat_centers = np.linspace(-90, 90, n_lat + 1)
    lat_centers = (lat_centers[:-1] + lat_centers[1:]) / 2.0
    lon_sun = (np.deg2rad(lon_centers)[None, :] - np.deg2rad(subsolar_lon_deg) + np.pi) % (2 * np.pi) - np.pi
    lat2d = np.deg2rad(lat_centers)[:, None]
    eff_cos = np.cos(lat2d) * np.cos(lon_sun)
    is_dawn = np.broadcast_to((lon_sun < 0.0), eff_cos.shape)
    return eff_cos, is_dawn

def get_closest_surface_density_file(target_dir, target_taa):
    grid_files = glob.glob(os.path.join(target_dir, "density_grid_*.npy"))
    if not grid_files: return None, None, None
    min_diff = float('inf')
    best = (None, None, None)
    for f in grid_files:
        m = re.search(r'_t(\d+)_taa(\d+)\.npy$', os.path.basename(f))
        if m:
            time_id, taa = int(m.group(1)), int(m.group(2))
            diff = abs(taa - target_taa)
            if diff < min_diff:
                surf = os.path.join(target_dir, f"surface_density_t{time_id:05d}.npy")
                if os.path.exists(surf):
                    min_diff = diff
                    best = (surf, time_id, taa)
    return best

def load_and_align_density(filepath, time_h, orbit_data, t_start):
    data = np.load(filepath)
    if data.ndim == 3: data = np.sum(data, axis=2)
    subsolar_lon_deg = get_subsolar_longitude_linear(time_h, t_start, orbit_data)
    sun_index = int(np.round(((subsolar_lon_deg + 180.0) % 360.0) / (360.0 / N_LON))) % N_LON
    data = np.roll(data, shift=(N_LON // 2) - sun_index, axis=0)
    data_T = np.nan_to_num(data.T, nan=0.0) / 10000.0
    eff_cos, is_dawn = compute_effcos_grid(N_LON, N_LAT, 0.0)
    return data_T, eff_cos, is_dawn


# ==========================================
# [1] 放出位置の重心
# ==========================================
def plot_emission_centroid(models, side, smooth_deg=3):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    peak_info = {"PSD": {}, "TD": {}}

    for label, subdir in models.items():
        try:
            d = load_gen(os.path.join(BASE_DIR, subdir), side)
        except Exception as e:
            continue
        bc = d['band_centers']
        taa = d['taa']
        c = COLORS.get(label)
        for ax, key, name in [(axes[0], 'gen_psd', 'PSD'), (axes[1], 'gen_td', 'TD')]:
            gen = smooth_2d(d[key], smooth_deg)
            gsum = gen.sum(axis=1)
            centroid = np.where(gsum > 0, (gen * bc[None, :]).sum(axis=1) / np.where(gsum > 0, gsum, 1), np.nan)
            ax.plot(taa, centroid, '-', color=c, lw=2.2, label=label)
            peak_info[name][label] = taa[np.nanargmax(centroid)]

    for ax, name, extra in [(axes[0], 'PSD', '低い=ターミネーター寄り'), (axes[1], 'TD', '高い=正午(SSP)寄り')]:
        ax.axvline(180, color='gray', ls=':', alpha=0.6, label='遠日点(180)')
        ax.set_ylabel(f'{name}放出の eff_cos 重心')
        ax.set_title(f'{name}放出が起きる局所時刻(eff_cos)の重心 — {side}側  ({extra})')
        ax.grid(True, ls='--', alpha=0.5)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel('TAA [deg]')
    axes[-1].set_xlim(0, 360)
    axes[-1].xaxis.set_major_locator(MultipleLocator(60))
    plt.tight_layout()
    plt.show()


# ==========================================
# [2] TAAごとの放出量プロファイル (PSD・TD別々) - 線形スケール
# ==========================================
def plot_zenith_emission_profile(models, side, taa_list):
    for label, subdir in models.items():
        try:
            d = load_gen(os.path.join(BASE_DIR, subdir), side)
        except Exception as e:
            continue
        bc = d['band_centers']
        taa_axis = d['taa']
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        cmap = plt.get_cmap('viridis')
        colors = [cmap(x) for x in np.linspace(0, 0.85, len(taa_list))]
        
        for ax, key, name in [(axes[0], 'gen_psd', 'PSD'), (axes[1], 'gen_td', 'TD')]:
            for ci, target in enumerate(taa_list):
                ti = np.argmin(np.abs(taa_axis - target))
                actual_taa = taa_axis[ti]
                
                # 放出量のプロット
                ax.plot(bc, d[key][ti, :], '-o', color=colors[ci], ms=5, lw=1.8, label=f'TAA={actual_taa}°')
                
                # 近日点Dawn地点の現在地を同色の点線でプロット
                ec_dawn = get_perihelion_dawn_effcos(actual_taa)
                if ec_dawn >= 0:
                    ax.axvline(ec_dawn, color=colors[ci], ls=':', lw=2, alpha=0.7)
            
            # 線形スケール＋指数表記
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
            ax.set_xlabel('eff_cos [0=ターミネーター → 1=SSP]')
            ax.set_ylabel(f'{name}放出量 [atoms]')
            ax.set_title(f'{name}放出')
            ax.set_xlim(0, 1)
            ax.grid(True, which='both', ls='--', alpha=0.4)
            ax.legend(title='True Anomaly\n(点線は近日点Dawn地点)', fontsize=9)
            
        fig.suptitle(f'{label} — 天頂角に対する放出量プロファイル ({side}側) [線形スケール]', fontsize=13)
        plt.tight_layout()
        plt.show()


# ==========================================
# [4] PSD vs TD 重ね比較 (TAAごとのサブプロット) - 線形スケール
# ==========================================
def plot_psd_td_overlay(models, side, taa_list):
    for label, subdir in models.items():
        try:
            d = load_gen(os.path.join(BASE_DIR, subdir), side)
        except Exception as e:
            continue
        bc = d['band_centers']
        taa_axis = d['taa']

        n = len(taa_list)
        ncol = 2
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(8 * ncol, 4.5 * nrow), squeeze=False)
        axes_flat = axes.flatten()

        for pi, target in enumerate(taa_list):
            ax = axes_flat[pi]
            ti = np.argmin(np.abs(taa_axis - target))
            actual = taa_axis[ti]
            psd = d['gen_psd'][ti, :]
            td = d['gen_td'][ti, :]

            lp, = ax.plot(bc, psd, '-o', color='royalblue', ms=5, lw=2, label='PSD放出')
            lt, = ax.plot(bc, td, '-o', color='crimson', ms=5, lw=2, label='TD放出')
            
            # 線形スケール＋指数表記
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

            # TD主力域の背景塗色処理 (交点を線形補間)
            eps = 1e-300
            dlog = np.log10(np.maximum(td, eps)) - np.log10(np.maximum(psd, eps))
            spans = []  
            cur_start = None
            for b in range(len(bc)):
                if dlog[b] > 0 and cur_start is None:
                    if b > 0 and dlog[b-1] <= 0:
                        t = (0 - dlog[b-1]) / (dlog[b] - dlog[b-1])
                        cur_start = bc[b-1] + t * (bc[b] - bc[b-1])
                    else:
                        cur_start = bc[b]  
                elif dlog[b] <= 0 and cur_start is not None:
                    t = (0 - dlog[b-1]) / (dlog[b] - dlog[b-1])
                    x_end = bc[b-1] + t * (bc[b] - bc[b-1])
                    spans.append((cur_start, x_end))
                    cur_start = None
            if cur_start is not None: spans.append((cur_start, 1.0))

            for x0, x1 in spans:
                ax.axvspan(max(x0, 0.0), min(x1, 1.0), color='crimson', alpha=0.10)

            # 近日点でDawnターミネーターだった地点(-90度)の追跡
            ec_dawn = get_perihelion_dawn_effcos(actual)
            h_ld = None
            if ec_dawn >= 0:
                h_ld = ax.axvline(ec_dawn, color='darkorange', ls='--', lw=2.5, 
                                  label='近日点Dawn地点 (温存在庫の本隊)')

            # TD寄与率(右軸)
            ax2 = ax.twinx()
            tot = psd + td
            frac = np.where(tot > 0, td / tot, np.nan)
            lf, = ax2.plot(bc, frac, ':', color='black', lw=1.5, label='TD寄与率')
            ax2.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.6)
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('TD寄与率', fontsize=9, color='black')
            ax2.tick_params(axis='y', labelsize=8)

            ax.set_title(f'TAA = {actual}°', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.grid(True, which='both', ls='--', alpha=0.3)
            
            # 凡例の統合
            if pi == 0:
                handles = [lp, lt, lf]
                if h_ld is not None: handles.append(h_ld)
                ax.legend(handles=handles, loc='upper left', fontsize=9, ncol=2)

        for pi in range(n, len(axes_flat)):
            axes_flat[pi].set_visible(False)

        for pi in range(n):
            r, cc = divmod(pi, ncol)
            if r == nrow - 1 or pi + ncol >= n:
                axes_flat[pi].set_xlabel('eff_cos [0=ターミネーター → 1=SSP]', fontsize=11)
            if cc == 0:
                axes_flat[pi].set_ylabel('放出量 [atoms]', fontsize=11)

        fig.suptitle(f'{label} — PSD vs TD 放出量の重ね比較 ({side}側) [線形スケール]\n'
                     f'薄赤帯=TDがPSDを上回る領域 / オレンジ破線=巨大在庫が運ばれてきた位置', fontsize=14)
        plt.tight_layout()
        plt.show()


# ==========================================
# [4b] TD寄与率ヒートマップ
# ==========================================
def plot_td_fraction_heatmap(models, side, smooth_deg=3):
    for label, subdir in models.items():
        try:
            d = load_gen(os.path.join(BASE_DIR, subdir), side)
        except Exception as e:
            continue
        psd = smooth_2d(d['gen_psd'], smooth_deg)
        td = smooth_2d(d['gen_td'], smooth_deg)
        tot = psd + td
        frac = np.where(tot > 0, td / tot, np.nan) 

        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(frac.T, aspect='auto', origin='lower',
                       extent=[0, 360, 0, d['n_bands']], cmap='coolwarm',
                       vmin=0, vmax=1)
        ax.set_yticks(np.arange(d['n_bands']) + 0.5)
        ax.set_yticklabels(d['band_labels'], fontsize=8)
        ax.set_xlabel('TAA [deg]')
        ax.set_ylabel('eff_cos バンド (ターミネーター←→SSP)')
        ax.set_title(f'{label} — TD寄与率 TD/(PSD+TD) ({side}側)\n青=PSD主力, 赤=TD主力')
        ax.axvline(180, color='k', ls=':', alpha=0.5)
        
        # ヒートマップ上にも近日点Dawnの軌跡を重ねる(おまけ)
        taas = np.linspace(0, 360, 200)
        ec_dawns = [get_perihelion_dawn_effcos(t) for t in taas]
        # eff_cos を band_index(0~n_bands) のスケールに変換してプロット
        ec_y = [x * d['n_bands'] if x >= 0 else np.nan for x in ec_dawns]
        ax.plot(taas, ec_y, color='lime', lw=2, ls='--', label='近日点Dawn地点の軌跡')
        
        ax.xaxis.set_major_locator(MultipleLocator(60))
        cbar = plt.colorbar(im, ax=ax, label='TD寄与率')
        cbar.ax.axhline(0.5, color='k', lw=1)
        ax.legend(loc='upper left', fontsize=9)
        plt.tight_layout()
        plt.show()


# ==========================================
# [3] 天頂角 vs 平均表面密度 (生の.npy)
# ==========================================
def plot_zenith_density_profile(models, side, taa_list, orbit_data, t_start, n_bins=20):
    if orbit_data is None: return
    side_upper = side.upper()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    _, areas_cm2 = calculate_cell_areas(N_LON, N_LAT, R_BODY_KM)

    for label, subdir in models.items():
        target_dir = os.path.join(BASE_DIR, subdir)
        fig, ax = plt.subplots(figsize=(9, 6))
        cmap = plt.get_cmap('viridis')
        colors = [cmap(x) for x in np.linspace(0, 0.85, len(taa_list))]
        for ci, target_taa in enumerate(taa_list):
            filepath, time_h, actual_taa = get_closest_surface_density_file(target_dir, target_taa)
            if not filepath: continue
            try:
                data_T, eff_cos, is_dawn = load_and_align_density(filepath, time_h, orbit_data, t_start)
            except Exception:
                continue
            day_mask = eff_cos > 0.0
            if side_upper == "DAWN": side_mask = day_mask & is_dawn
            elif side_upper == "DUSK": side_mask = day_mask & (~is_dawn)
            else: side_mask = day_mask
            
            ec_flat = eff_cos[side_mask]
            dens_flat = data_T[side_mask]
            area_flat = areas_cm2[side_mask]
            prof = np.full(n_bins, np.nan)
            for b in range(n_bins):
                m = (ec_flat >= bin_edges[b]) & (ec_flat < bin_edges[b+1])
                if np.any(m):
                    prof[b] = np.sum(dens_flat[m] * area_flat[m]) / np.sum(area_flat[m])
            ax.plot(bin_centers, prof, '-o', color=colors[ci], markersize=5, lw=1.8, label=f'TAA={actual_taa}°')
        
        ax.set_xlabel('eff_cos [0=ターミネーター → 1=SSP]')
        ax.set_ylabel('平均表面密度 [atoms/cm²]')
        side_jp = {"DAWN": "明け方側", "DUSK": "夕方側"}.get(side_upper, side)
        ax.set_title(f'{label} — 天頂角に対する表面密度プロファイル ({side_jp})')
        ax.set_yscale('log')
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.legend(title='True Anomaly')
        plt.tight_layout()
        plt.show()

# ==========================================
# (おまけ) 放出量ヒートマップ
# ==========================================
def plot_emission_heatmap(models, side, smooth_deg=3):
    for label, subdir in models.items():
        try:
            d = load_gen(os.path.join(BASE_DIR, subdir), side)
        except Exception:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for ax, key, name in [(axes[0], 'gen_psd', 'PSD'), (axes[1], 'gen_td', 'TD')]:
            grid = smooth_2d(d[key], smooth_deg).T
            with np.errstate(divide='ignore'):
                loggrid = np.log10(np.where(grid > 0, grid, np.nan))
            im = ax.imshow(loggrid, aspect='auto', origin='lower',
                           extent=[0, 360, 0, d['n_bands']], cmap='inferno')
            ax.set_yticks(np.arange(d['n_bands']) + 0.5)
            ax.set_yticklabels(d['band_labels'], fontsize=8)
            ax.set_xlabel('TAA [deg]')
            ax.set_ylabel('eff_cos バンド')
            ax.set_title(f'{name}放出量')
            ax.axvline(180, color='white', ls=':', alpha=0.6)
            ax.xaxis.set_major_locator(MultipleLocator(60))
            plt.colorbar(im, ax=ax, label='log10(atoms)')
        fig.suptitle(f'{label} — 放出量ヒートマップ ({side}側)', fontsize=13)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # [1] 放出位置の重心
    plot_emission_centroid(MODELS, SIDE, smooth_deg=SMOOTH_WINDOW_DEG)

    # [2] TAAごとの放出量プロファイル (PSD・TD別々) ※線形スケール
    plot_zenith_emission_profile(MODELS, SIDE, PROFILE_TAAS)

    # [4] PSD vs TD 重ね比較 (TAAごとにサブプロット) ※線形スケール
    plot_psd_td_overlay(MODELS, SIDE, PROFILE_TAAS)

    # [4b] TD寄与率ヒートマップ (おまけで軌跡追加)
    plot_td_fraction_heatmap(MODELS, SIDE, smooth_deg=SMOOTH_WINDOW_DEG)

    # [3] 天頂角 vs 平均表面密度 (生の.npy)
    if os.path.exists(ORBIT_FILE_PATH):
        orbit_data, t_start = load_orbit_data(ORBIT_FILE_PATH)
        plot_zenith_density_profile(MODELS, SIDE, PROFILE_TAAS, orbit_data, t_start, n_bins=ZENITH_N_BINS)
    else:
        print(f"\n[警告] 軌道ファイル ({ORBIT_FILE_PATH}) なし。表面密度プロットをスキップ。")

    # (おまけ) 放出量ヒートマップ
    plot_emission_heatmap(MODELS, SIDE, smooth_deg=SMOOTH_WINDOW_DEG)