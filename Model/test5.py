# -*- coding: utf-8 -*-
"""
放出位置の解析(実測のみ・再構成なし)

band_statistics_per_taa.csv の実測 Gen_PSD / Gen_TD (eff_cosバンド × TAA) を使って:

  [1] 放出位置の重心 (PSD/TD, Q比較)
  [2] TAAごとの天頂角に対する放出量プロファイル (PSD・TD別々)
  [3] TAAごとの天頂角に対する平均表面密度プロファイル (生の.npyから)
  [4] ★新規: PSD vs TD 重ね比較。TAAごとにサブプロットを分け、
      各パネルで PSD と TD の2本だけを重ねる(線が増えない)。
      背景に「TDが主力の領域」を陰影で示し、下段にTD寄与率も表示。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
from matplotlib.ticker import MultipleLocator
import os
import glob
import re
import sys

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
    if w <= 0:
        return y
    w = int(w)
    k = np.ones(w) / w
    ext = np.concatenate([y[-w:], y, y[:w]])
    return np.convolve(ext, k, mode='same')[w:-w]


def smooth_2d(arr, w):
    if w <= 0:
        return arr
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
        print(f"軌道ファイル読み込み完了: {orbit_file_path}")
        return orbit_data, t_file_start
    except Exception as e:
        print(f"軌道ファイル読み込みエラー: {e}")
        return None, None


def get_subsolar_longitude_linear(time_h, t_file_start, orbit_data):
    time_col_original = orbit_data[:, 2]
    current_t_sec = t_file_start + (float(time_h) * 3600.0)
    t_lookup = np.clip(current_t_sec, time_col_original[0], time_col_original[-1])
    return np.interp(t_lookup, time_col_original, orbit_data[:, 5])


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
    lon_rad = np.deg2rad(lon_centers)
    lat_rad = np.deg2rad(lat_centers)
    sub_rad = np.deg2rad(subsolar_lon_deg)
    lon_sun = (lon_rad[None, :] - sub_rad + np.pi) % (2 * np.pi) - np.pi
    lat2d = lat_rad[:, None]
    eff_cos = np.cos(lat2d) * np.cos(lon_sun)
    is_dawn = np.broadcast_to((lon_sun < 0.0), eff_cos.shape)
    return eff_cos, is_dawn


def get_closest_surface_density_file(target_dir, target_taa):
    grid_files = glob.glob(os.path.join(target_dir, "density_grid_*.npy"))
    if not grid_files:
        return None, None, None
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
    if data.ndim == 3:
        data = np.sum(data, axis=2)
    subsolar_lon_deg = get_subsolar_longitude_linear(time_h, t_start, orbit_data)
    dlon = 360.0 / N_LON
    sun_pos_norm = (subsolar_lon_deg + 180.0) % 360.0
    sun_index = int(np.round(sun_pos_norm / dlon)) % N_LON
    shift = (N_LON // 2) - sun_index
    data = np.roll(data, shift=shift, axis=0)
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
            print(f"[エラー] {label}: {e}")
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

    print("\n" + "=" * 70)
    print(f"=== 放出重心が最大になる TAA ({side}側) ===")
    print("=" * 70)
    for name, dd in peak_info.items():
        line = f"  [{name}]  "
        for label, pk in dd.items():
            line += f"{label}: {pk:.0f}°   "
        labels = list(dd.keys())
        if len(labels) == 2:
            line += f"→ Δ={dd[labels[1]] - dd[labels[0]]:+.0f}°"
        print(line)
    print("=" * 70)


# ==========================================
# [2] TAAごとの放出量プロファイル (PSD・TD別々)
# ==========================================
def plot_zenith_emission_profile(models, side, taa_list):
    for label, subdir in models.items():
        try:
            d = load_gen(os.path.join(BASE_DIR, subdir), side)
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            continue
        bc = d['band_centers']
        taa_axis = d['taa']
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        cmap = plt.get_cmap('viridis')
        colors = [cmap(x) for x in np.linspace(0, 0.85, len(taa_list))]
        for ax, key, name in [(axes[0], 'gen_psd', 'PSD'), (axes[1], 'gen_td', 'TD')]:
            for ci, target in enumerate(taa_list):
                ti = np.argmin(np.abs(taa_axis - target))
                ax.plot(bc, d[key][ti, :], '-o', color=colors[ci], ms=5, lw=1.8, label=f'TAA={taa_axis[ti]}°')
            ax.set_yscale('log')
            ax.set_xlabel('eff_cos [0=ターミネーター → 1=SSP]')
            ax.set_ylabel(f'{name}放出量 [atoms]')
            ax.set_title(f'{name}放出')
            ax.set_xlim(0, 1)
            ax.grid(True, which='both', ls='--', alpha=0.4)
            ax.legend(title='True Anomaly', fontsize=9)
        fig.suptitle(f'{label} — 天頂角に対する放出量プロファイル ({side}側)', fontsize=13)
        plt.tight_layout()
        plt.show()


# ==========================================
# [4] ★新規: PSD vs TD 重ね比較 (TAAごとにサブプロット)
# ==========================================
def plot_psd_td_overlay(models, side, taa_list):
    """各TAAを別パネルにして、そのパネル内で PSD と TD の2本だけを重ねる。
    線が増えないので「どっちが主力か」がパネルごとに一目で分かる。
    - PSD=青実線, TD=赤実線
    - 背景陰影: TDがPSDを上回る eff_cos 帯を薄赤で塗る(TD主力域)
    - 各パネル下に TD寄与率 (TD/(PSD+TD)) を細い黒線で重ねる(右軸)
    モデル(Q)ごとに1枚。"""
    for label, subdir in models.items():
        try:
            d = load_gen(os.path.join(BASE_DIR, subdir), side)
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            continue
        bc = d['band_centers']
        taa_axis = d['taa']

        n = len(taa_list)
        ncol = 2
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 4 * nrow), squeeze=False)
        axes_flat = axes.flatten()

        for pi, target in enumerate(taa_list):
            ax = axes_flat[pi]
            ti = np.argmin(np.abs(taa_axis - target))
            actual = taa_axis[ti]
            psd = d['gen_psd'][ti, :]
            td = d['gen_td'][ti, :]

            # PSD と TD の2本(これだけ)
            lp, = ax.plot(bc, psd, '-o', color='royalblue', ms=5, lw=2, label='PSD')
            lt, = ax.plot(bc, td, '-o', color='crimson', ms=5, lw=2, label='TD')
            ax.set_yscale('log')

            # --- TD主力域を「実際にTD>PSDとなる区間」で塗る(交点を線形補間で算出) ---
            # log空間で d = log(td) - log(psd) を作り、d>0 の区間を塗る。
            # d の符号がバンド間で変わる箇所は線形補間で交点 eff_cos を求める。
            eps = 1e-300
            dlog = np.log10(np.maximum(td, eps)) - np.log10(np.maximum(psd, eps))
            spans = []  # TD主力(td>psd)の連続区間 [x_start, x_end]
            cur_start = None
            for b in range(len(bc)):
                if dlog[b] > 0 and cur_start is None:
                    # 区間開始。前のバンドとの間に交点があれば、そこを開始点に
                    if b > 0 and dlog[b-1] <= 0:
                        # dlog[b-1]<=0, dlog[b]>0 の間で線形補間
                        t = (0 - dlog[b-1]) / (dlog[b] - dlog[b-1])
                        cur_start = bc[b-1] + t * (bc[b] - bc[b-1])
                    else:
                        cur_start = bc[b]  # 端(最初のバンドから既にTD>PSD)
                elif dlog[b] <= 0 and cur_start is not None:
                    # 区間終了。交点を終了点に
                    t = (0 - dlog[b-1]) / (dlog[b] - dlog[b-1])
                    x_end = bc[b-1] + t * (bc[b] - bc[b-1])
                    spans.append((cur_start, x_end))
                    cur_start = None
            if cur_start is not None:
                # 最後までTD主力が続いた場合、右端(eff_cos=1)まで
                spans.append((cur_start, 1.0))

            for x0, x1 in spans:
                ax.axvspan(max(x0, 0.0), min(x1, 1.0), color='crimson', alpha=0.10)

            # 交点(PSD↔TD逆転点)に縦線マーカー
            crossover_xs = []
            for b in range(1, len(bc)):
                if (dlog[b-1] <= 0) != (dlog[b] <= 0):  # 符号が変わった
                    t = (0 - dlog[b-1]) / (dlog[b] - dlog[b-1])
                    crossover_xs.append(bc[b-1] + t * (bc[b] - bc[b-1]))
            for xc in crossover_xs:
                ax.axvline(xc, color='gray', ls='-', lw=0.8, alpha=0.7)

            # TD寄与率(右軸)
            ax2 = ax.twinx()
            tot = psd + td
            frac = np.where(tot > 0, td / tot, np.nan)
            lf, = ax2.plot(bc, frac, ':', color='black', lw=1.3, label='TD寄与率')
            ax2.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.6)
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('TD寄与率', fontsize=9, color='black')
            ax2.tick_params(axis='y', labelsize=8)

            ax.set_title(f'TAA = {actual}°')
            ax.set_xlim(0, 1)
            ax.grid(True, which='both', ls='--', alpha=0.3)
            if pi == 0:
                ax.legend(handles=[lp, lt, lf], loc='lower center', fontsize=9, ncol=3)

        # 余ったパネルを消す
        for pi in range(n, len(axes_flat)):
            axes_flat[pi].set_visible(False)

        # 共通ラベル
        for pi in range(n):
            r, cc = divmod(pi, ncol)
            if r == nrow - 1 or pi + ncol >= n:
                axes_flat[pi].set_xlabel('eff_cos [0=ターミネーター → 1=SSP]')
            if cc == 0:
                axes_flat[pi].set_ylabel('放出量 [atoms]')

        fig.suptitle(f'{label} — PSD vs TD 放出量の重ね比較 ({side}側)\n'
                     f'薄赤帯=TD主力域, 点線=TD寄与率(右軸), 0.5超でTD主力', fontsize=12)
        plt.tight_layout()
        plt.show()


# ==========================================
# [4b] ★新規: TD寄与率ヒートマップ (TAA × eff_cos)
# ==========================================
def plot_td_fraction_heatmap(models, side, smooth_deg=3):
    """TD/(PSD+TD) を TAA×eff_cos のヒートマップに。
    0.5を境に発散配色。どこでPSD主力/TD主力かが一目で分かり、Q比較もできる。"""
    for label, subdir in models.items():
        try:
            d = load_gen(os.path.join(BASE_DIR, subdir), side)
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            continue
        psd = smooth_2d(d['gen_psd'], smooth_deg)
        td = smooth_2d(d['gen_td'], smooth_deg)
        tot = psd + td
        frac = np.where(tot > 0, td / tot, np.nan)  # (360, n_bands)

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
        ax.xaxis.set_major_locator(MultipleLocator(60))
        cbar = plt.colorbar(im, ax=ax, label='TD寄与率')
        cbar.ax.axhline(0.5, color='k', lw=1)
        plt.tight_layout()
        plt.show()


# ==========================================
# [3] 天頂角 vs 平均表面密度 (生の.npy)
# ==========================================
def plot_zenith_density_profile(models, side, taa_list, orbit_data, t_start, n_bins=20):
    if orbit_data is None:
        print("[エラー] 軌道データがないため表面密度プロファイルをスキップ。")
        return
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
            if not filepath:
                print(f"  [スキップ] {label}: TAA={target_taa} 付近の.npyなし")
                continue
            try:
                data_T, eff_cos, is_dawn = load_and_align_density(filepath, time_h, orbit_data, t_start)
            except Exception as e:
                print(f"  [エラー] {label} taa={target_taa}: {e}")
                continue
            day_mask = eff_cos > 0.0
            if side_upper == "DAWN":
                side_mask = day_mask & is_dawn
            elif side_upper == "DUSK":
                side_mask = day_mask & (~is_dawn)
            else:
                side_mask = day_mask
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
        except Exception as e:
            print(f"[エラー] {label}: {e}")
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

    # [2] TAAごとの放出量プロファイル (PSD・TD別々)
    plot_zenith_emission_profile(MODELS, SIDE, PROFILE_TAAS)

    # [4] ★新規: PSD vs TD 重ね比較 (TAAごとにサブプロット)
    plot_psd_td_overlay(MODELS, SIDE, PROFILE_TAAS)

    # [4b] ★新規: TD寄与率ヒートマップ
    plot_td_fraction_heatmap(MODELS, SIDE, smooth_deg=SMOOTH_WINDOW_DEG)

    # [3] 天頂角 vs 平均表面密度 (生の.npy)
    if os.path.exists(ORBIT_FILE_PATH):
        orbit_data, t_start = load_orbit_data(ORBIT_FILE_PATH)
        plot_zenith_density_profile(MODELS, SIDE, PROFILE_TAAS, orbit_data, t_start, n_bins=ZENITH_N_BINS)
    else:
        print(f"\n[警告] 軌道ファイル ({ORBIT_FILE_PATH}) なし。表面密度プロットをスキップ。")

    # (おまけ) 放出量ヒートマップ
    plot_emission_heatmap(MODELS, SIDE, smooth_deg=SMOOTH_WINDOW_DEG)