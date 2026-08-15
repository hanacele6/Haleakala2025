# -*- coding: utf-8 -*-
"""
在庫解析(窓を1箇所で統一) — .npyスナップショットから計算

--- なぜこれが必要か ---
これまで窓(eff_cos範囲)が解析ごとにバラバラだった:
  ・band_statistics の最終バンド : 0.55-1.01 (天頂角 0-57°)
  ・凍結テスト(SUB_BANDS合計)    : 0.40-1.01 (天頂角 0-66°)
  ・経度解析(WINDOW_EFFCOS)      : 0.70-1.01 → 0.70-0.85 (天頂角 32-46°)
CSVは本体の EFF_COS_BAND_EDGES で丸め込まれているため任意の窓を取れない。
.npy から直接計算すれば窓を自由に選べる。

--- 出力 ---
[1] 在庫のTAA依存: 昼面全体 / Dawn側 / Dusk側 (Q比較)
[2] 凍結テスト(受動輸送): 同じ窓で、消費ゼロなら在庫がどう推移したか
    → [1]と[2]が同じ窓なので、そのまま並べて議論できる
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
from matplotlib.ticker import MultipleLocator
import os
import glob
import re

# ==========================================
# ★ 窓の設定はここ1箇所だけ ★
# ==========================================
WINDOW_EFFCOS = (0.55, 1.01)  
# 参考: (0.55, 1.01) = 天頂角 0-57°,  (0.40, 1.01) = 天頂角 0-66°

# ==========================================
# その他の設定
# ==========================================
BASE_DIR = r"./SimulationResult_202607"

MODELS = {
    "Q2.0 (Standard)": "ParabolicHop_72x36_NoEq_DT100_0716_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr_test2",
    "Q0.3 (Weak PSD)": "ParabolicHop_72x36_NoEq_DT100_0716_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr_test2",
}
COLORS = {"Q2.0 (Standard)": "crimson", "Q0.3 (Weak PSD)": "steelblue"}

ORBIT_FILE_PATH = 'orbit2025_spice_unwrapped.txt'
N_LON, N_LAT = 72, 36
R_BODY_KM = 2439.7
MERCURY_YEAR_SEC = 87.969 * 86400

FREEZE_TAAS = [0, 100,]     # 凍結テストで在庫パターンを固定するTAA
PEAK_SEARCH_RANGE = (100, 250)  # ピーク探索範囲(2回目の通過を拾わないため)


# ==========================================
# 幾何・軌道
# ==========================================
def load_orbit_data(path):
    orbit = np.loadtxt(path)
    orbit[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit[:, 0])))
    return orbit, orbit[0, 2]


def get_subsolar_lon(time_h, t_start, orbit):
    tcol = orbit[:, 2]
    t = np.clip(t_start + float(time_h) * 3600.0, tcol[0], tcol[-1])
    return np.interp(t, tcol, orbit[:, 5])


def cell_areas_cm2(n_lon, n_lat, r_km):
    r_cm = r_km * 1e5
    dlon = 2.0 * np.pi / n_lon
    lat_edges = np.linspace(-np.pi/2, np.pi/2, n_lat + 1)
    areas = np.zeros((n_lat, n_lon))
    for i in range(n_lat):
        areas[i, :] = (r_cm ** 2) * dlon * (np.sin(lat_edges[i+1]) - np.sin(lat_edges[i]))
    return areas


def effcos_grid(n_lon, n_lat, sub_lon_deg):
    lon_e = np.linspace(-180, 180, n_lon + 1)
    lon_c = np.deg2rad((lon_e[:-1] + lon_e[1:]) / 2.0)
    lat_e = np.linspace(-90, 90, n_lat + 1)
    lat_c = np.deg2rad((lat_e[:-1] + lat_e[1:]) / 2.0)
    lon_sun = (lon_c[None, :] - np.deg2rad(sub_lon_deg) + np.pi) % (2*np.pi) - np.pi
    eff_cos = np.cos(lat_c[:, None]) * np.cos(lon_sun)
    is_dawn = np.broadcast_to(lon_sun < 0.0, eff_cos.shape)
    return eff_cos, is_dawn


# ==========================================
# スナップショット
# ==========================================
def final_year_snapshots(target_dir):
    files = glob.glob(os.path.join(target_dir, "density_grid_*.npy"))
    recs = []
    for f in files:
        m = re.search(r'_t(\d+)_taa(\d+)\.npy$', os.path.basename(f))
        if m:
            th, taa = int(m.group(1)), int(m.group(2))
            surf = os.path.join(target_dir, f"surface_density_t{th:05d}.npy")
            if os.path.exists(surf):
                recs.append({'time_h': th, 'taa': taa, 'path': surf})
    if not recs:
        raise FileNotFoundError(f"スナップショットなし: {target_dir}")
    max_year = max(int(r['time_h'] * 3600.0 // MERCURY_YEAR_SEC) + 1 for r in recs)
    recs = [r for r in recs if int(r['time_h'] * 3600.0 // MERCURY_YEAR_SEC) + 1 == max_year]
    recs.sort(key=lambda r: r['taa'])
    return recs, max_year


def load_surface_bodyfixed(path):
    d = np.load(path)
    if d.ndim == 3:
        d = np.sum(d, axis=2)
    return np.nan_to_num(d.T, nan=0.0) / 10000.0


def stock_in_window(dens_T, eff_cos, is_dawn, areas, window, side):
    """指定した窓(eff_cos範囲)に入る在庫総量 [atoms]
    side: 'DAWN' / 'DUSK' / 'FULL'"""
    lo, hi = window
    inwin = (eff_cos >= lo) & (eff_cos < hi)
    if side == "DAWN":
        m = inwin & is_dawn
    elif side == "DUSK":
        m = inwin & (~is_dawn)
    else:
        m = inwin
    return np.sum(dens_T[m] * areas[m]) if np.any(m) else 0.0


def peak_taa_in_range(taas, values, rng):
    lo, hi = rng
    m = (taas >= lo) & (taas <= hi)
    if not np.any(m):
        return np.nan
    return taas[m][int(np.nanargmax(np.asarray(values)[m]))]


# ==========================================
# [1] 在庫のTAA依存 (全体 / Dawn / Dusk)
# ==========================================
def plot_stock_variations(models, orbit, t_start, window):
    areas = cell_areas_cm2(N_LON, N_LAT, R_BODY_KM)
    lo, hi = window
    z_hi = np.rad2deg(np.arccos(min(hi, 1.0)))
    z_lo = np.rad2deg(np.arccos(lo))

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    sides = ["FULL", "DAWN", "DUSK"]
    titles = ['昼面全体 (Dawn + Dusk)', 'Dawn側のみ', 'Dusk側のみ']

    peaks = {s: {} for s in sides}

    for label, subdir in models.items():
        try:
            recs, yr = final_year_snapshots(os.path.join(BASE_DIR, subdir))
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            continue

        taas = np.array([r['taa'] for r in recs])
        vals = {s: [] for s in sides}
        for r in recs:
            dens = load_surface_bodyfixed(r['path'])
            sl = get_subsolar_lon(r['time_h'], t_start, orbit)
            ec, isd = effcos_grid(N_LON, N_LAT, sl)
            for s in sides:
                vals[s].append(stock_in_window(dens, ec, isd, areas, window, s))

        for ax, s in zip(axes, sides):
            y = np.array(vals[s])
            pk = peak_taa_in_range(taas, y, PEAK_SEARCH_RANGE)
            peaks[s][label] = pk
            ax.plot(taas, y, '-', color=COLORS.get(label), lw=2,
                    label=f'{label} (peak={pk:.0f}°)')
        print(f"[読込] {label}: Year {yr}, {len(recs)}スナップショット")

    for ax, t in zip(axes, titles):
        ax.axvline(180, color='gray', ls=':', alpha=0.7, label='遠日点(180)')
        ax.axvline(165, color='green', ls='--', alpha=0.5, label='Dawn柱密度ピーク(165)')
        ax.set_title(t, fontsize=12)
        ax.set_ylabel('在庫 [atoms]')
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel('TAA [deg]')
    axes[-1].set_xlim(0, 360)
    axes[-1].xaxis.set_major_locator(MultipleLocator(60))
    fig.suptitle(f'在庫のTAA依存  —  窓: eff_cos {lo:.2f}-{hi:.2f} (天頂角 {z_hi:.0f}°-{z_lo:.0f}°)',
                 fontsize=13)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 74)
    print(f"=== [1] 在庫ピークTAA  (窓 eff_cos {lo:.2f}-{hi:.2f}) ===")
    print("=" * 74)
    labels = list(models.keys())
    print(f"{'領域':<10}" + "".join(f"{l:<22}" for l in labels) + "差")
    print("-" * 74)
    for s, t in zip(sides, titles):
        line = f"{t[:8]:<10}"
        vs = []
        for l in labels:
            v = peaks[s].get(l, np.nan)
            vs.append(v)
            line += f"{v:<22.0f}"
        if len(vs) == 2 and all(np.isfinite(vs)):
            line += f"{vs[1]-vs[0]:+.0f}°"
        print(line)
    print("=" * 74)


# ==========================================
# [2] 凍結テスト(受動輸送) — 同じ窓で
# ==========================================
def freeze_test(models, orbit, t_start, window, freeze_taas, side="DAWN"):
    areas = cell_areas_cm2(N_LON, N_LAT, R_BODY_KM)
    lo, hi = window

    results = {}
    fig, axes = plt.subplots(len(models), 1, figsize=(11, 5 * len(models)), sharex=True)
    if len(models) == 1:
        axes = [axes]

    for ax, (label, subdir) in zip(axes, models.items()):
        try:
            recs, yr = final_year_snapshots(os.path.join(BASE_DIR, subdir))
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            continue
        taas = np.array([r['taa'] for r in recs])

        actual = []
        for r in recs:
            dens = load_surface_bodyfixed(r['path'])
            sl = get_subsolar_lon(r['time_h'], t_start, orbit)
            ec, isd = effcos_grid(N_LON, N_LAT, sl)
            actual.append(stock_in_window(dens, ec, isd, areas, window, side))
        actual = np.array(actual)
        a_pk = peak_taa_in_range(taas, actual, PEAK_SEARCH_RANGE)

        ax.plot(taas, actual, '-', color='black', lw=2.4, label=f'実測 (peak={a_pk:.0f}°)')

        res = {'actual_peak': a_pk, 'passive': {}}
        cmap = plt.get_cmap('autumn')
        for ci, ft in enumerate(freeze_taas):
            i0 = int(np.argmin(np.abs(taas - ft)))
            frozen = load_surface_bodyfixed(recs[i0]['path'])
            passive = []
            for r in recs:
                sl = get_subsolar_lon(r['time_h'], t_start, orbit)
                ec, isd = effcos_grid(N_LON, N_LAT, sl)
                passive.append(stock_in_window(frozen, ec, isd, areas, window, side))
            passive = np.array(passive)
            p_pk = peak_taa_in_range(taas, passive, PEAK_SEARCH_RANGE)
            res['passive'][taas[i0]] = p_pk
            ax.plot(taas, passive, '--', lw=1.8,
                    color=cmap(ci / max(len(freeze_taas)-1, 1) * 0.7),
                    label=f'受動輸送のみ TAA={taas[i0]}°凍結 (peak={p_pk:.0f}°)')

        results[label] = res
        ax.axvline(180, color='gray', ls=':', alpha=0.6)
        ax.axvline(165, color='green', ls='--', alpha=0.5)
        ax.set_title(f'{label}  ({side}側)', fontsize=12)
        ax.set_ylabel('在庫 [atoms]')
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel('TAA [deg]')
    axes[-1].set_xlim(0, 360)
    axes[-1].xaxis.set_major_locator(MultipleLocator(60))
    z_hi = np.rad2deg(np.arccos(min(hi, 1.0)))
    z_lo = np.rad2deg(np.arccos(lo))
    fig.suptitle(f'凍結テスト  —  窓: eff_cos {lo:.2f}-{hi:.2f} (天頂角 {z_hi:.0f}°-{z_lo:.0f}°)\n'
                 '破線=消費ゼロで運んだだけの場合', fontsize=13)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 74)
    print(f"=== [2] 凍結テスト ピークTAA  (窓 eff_cos {lo:.2f}-{hi:.2f}, {side}側) ===")
    print("=" * 74)
    labels = list(results.keys())
    fts = sorted({t for r in results.values() for t in r['passive']})
    print(f"{'凍結TAA':<12}" + "".join(f"{l:<22}" for l in labels) + "差")
    print("-" * 74)
    for ft in fts:
        line = f"{ft:<12.0f}"
        vs = [results[l]['passive'].get(ft, np.nan) for l in labels]
        for v in vs:
            line += f"{v:<22.0f}"
        if len(vs) == 2 and all(np.isfinite(vs)):
            line += f"{vs[1]-vs[0]:+.0f}°"
        print(line)
    line = f"{'実測':<12}"
    vs = [results[l]['actual_peak'] for l in labels]
    for v in vs:
        line += f"{v:<22.0f}"
    if len(vs) == 2 and all(np.isfinite(vs)):
        line += f"{vs[1]-vs[0]:+.0f}°"
    print(line)
    print("=" * 74)
    print("\n[読み方] 受動(消費ゼロ)の差が実測の差にどれだけ迫るか")
    print("         → 迫っていれば『Q差はパターン起源』")


if __name__ == "__main__":
    if not os.path.exists(ORBIT_FILE_PATH):
        print(f"エラー: 軌道ファイルなし {ORBIT_FILE_PATH}")
    else:
        orbit, t_start = load_orbit_data(ORBIT_FILE_PATH)
        lo, hi = WINDOW_EFFCOS
        print(f"軌道ファイル読み込み完了")
        print(f"窓: eff_cos {lo:.2f}-{hi:.2f} "
              f"(天頂角 {np.rad2deg(np.arccos(min(hi,1.0))):.0f}°-{np.rad2deg(np.arccos(lo)):.0f}°)\n")

        plot_stock_variations(MODELS, orbit, t_start, WINDOW_EFFCOS)
        freeze_test(MODELS, orbit, t_start, WINDOW_EFFCOS, FREEZE_TAAS, side="DAWN")