# -*- coding: utf-8 -*-
"""
放出オフrun の評価 (表面在庫ピークで判定・標準runと比較)

--- 背景 ---
最終年だけ放出(PSD/TD/SWS)をゼロにしたrun。
表面在庫パターンは最終年初めの状態で凍結され、その後は新規放出による
変化がなくなる(電離による大気減衰のみ)。
→ 在庫ピークが放出オフでも保たれるか = 「近日点までに形成されたパターンが
  ピーク位置を決めているか」を本体シミュレーションで直接判定する。

--- 注意 ---
放出を止めると系は定常でなくなる(消費が減り在庫は溜まる方向、
あるいは供給とのバランスで単調ドリフト)。絶対量比較は不可。
そこで:
  1. TAA=0 と 360 の不一致でドリフト量を定量化
  2. 両端を結ぶ直線でデトレンドしてからピークを再評価
  3. ドリフトとピークシフトのオーダーを比較して有意性を判断

--- 出力 ---
[1] Dawn/Dusk/全体の在庫TAA分布: 放出オフ vs 標準 (生 + デトレンド)
[2] ピークTAA表 (生・デトレンド後) と ドリフト量
"""

import numpy as np
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

# 同じQで、放出オフ と 標準 の2run
MODELS = {
    "標準 (放出あり)": "ParabolicHop_72x36_NoEq_DT100_0713_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr",
    "放出オフ (最終年)": "ParabolicHop_72x36_NoEq_DT100_0724_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_ModB1.0T0.0P0.0_15yr_nonGentest",
}
COLORS = {"標準 (放出あり)": "gray", "放出オフ (最終年)": "crimson"}

ORBIT_FILE_PATH = 'orbit2025_spice_unwrapped.txt'
N_LON, N_LAT = 72, 36
R_BODY_KM = 2439.7
MERCURY_YEAR_SEC = 87.969 * 86400

WINDOW_EFFCOS = (0.55, 1.01)     # 在庫を見る窓 (天頂角 0-57°)
PEAK_SEARCH_RANGE = (60, 300)    # ピーク探索範囲
FRONT_HALF_ONLY = True           # Trueなら前半(TAA<=180)を主に評価(放出オフで後半が崩れやすいため)


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
    lo, hi = window
    inwin = (eff_cos >= lo) & (eff_cos < hi)
    if side == "DAWN":
        m = inwin & is_dawn
    elif side == "DUSK":
        m = inwin & (~is_dawn)
    else:
        m = inwin
    return np.sum(dens_T[m] * areas[m]) if np.any(m) else 0.0


def detrend_endpoints(taas, y):
    """TAA=0と360(=先頭と末尾)を結ぶ直線を差し引く。周期境界のドリフト除去。"""
    if len(y) < 2:
        return y.copy()
    # 先頭と末尾を直線で結ぶ
    x0, x1 = taas[0], taas[-1]
    y0, y1 = y[0], y[-1]
    if x1 == x0:
        return y.copy()
    trend = y0 + (y1 - y0) * (taas - x0) / (x1 - x0)
    return y - trend + np.mean(y)   # 平均を戻して正の値に


def peak_in_range(taas, y, rng, front_half=False):
    lo, hi = rng
    m = (taas >= lo) & (taas <= hi)
    if front_half:
        m = m & (taas <= 180)
    if not np.any(m):
        return np.nan
    return taas[m][int(np.nanargmax(np.asarray(y)[m]))]


# ==========================================
# 評価
# ==========================================
def evaluate(models, orbit, t_start, window, side="DAWN"):
    areas = cell_areas_cm2(N_LON, N_LAT, R_BODY_KM)
    lo, hi = window

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    results = {}

    for label, subdir in models.items():
        full = os.path.join(BASE_DIR, subdir)
        try:
            recs, yr = final_year_snapshots(full)
        except Exception as e:
            print(f"[エラー] {label}: {e}  (フォルダ名を確認してください: {subdir})")
            continue

        taas = np.array([r['taa'] for r in recs])
        vals = []
        for r in recs:
            dens = load_surface_bodyfixed(r['path'])
            sl = get_subsolar_lon(r['time_h'], t_start, orbit)
            ec, isd = effcos_grid(N_LON, N_LAT, sl)
            vals.append(stock_in_window(dens, ec, isd, areas, window, side))
        vals = np.array(vals)

        # 生ピーク
        raw_peak = peak_in_range(taas, vals, PEAK_SEARCH_RANGE, FRONT_HALF_ONLY)
        # デトレンド後
        detr = detrend_endpoints(taas, vals)
        detr_peak = peak_in_range(taas, detr, PEAK_SEARCH_RANGE, FRONT_HALF_ONLY)
        # ドリフト量(両端の相対差)
        drift = (vals[-1] - vals[0]) / np.mean(vals) * 100 if np.mean(vals) > 0 else np.nan

        results[label] = dict(raw_peak=raw_peak, detr_peak=detr_peak, drift=drift,
                              taas=taas, vals=vals, detr=detr)

        c = COLORS.get(label)
        axes[0].plot(taas, vals, '-', color=c, lw=2, label=f'{label} (peak={raw_peak:.0f}°)')
        axes[1].plot(taas, detr, '-', color=c, lw=2, label=f'{label} (peak={detr_peak:.0f}°)')
        print(f"[読込] {label}: Year {yr}, {len(recs)}スナップショット, ドリフト {drift:+.1f}%")

    z_hi = np.rad2deg(np.arccos(min(hi, 1.0)))
    z_lo = np.rad2deg(np.arccos(lo))
    axes[0].set_title(f'{side}側 在庫のTAA分布 (生データ)  窓: eff_cos {lo:.2f}-{hi:.2f} '
                      f'(天頂角 {z_hi:.0f}°-{z_lo:.0f}°)', fontsize=11)
    axes[1].set_title(f'{side}側 在庫のTAA分布 (両端デトレンド後)', fontsize=11)
    for ax in axes:
        ax.axvline(180, color='gray', ls=':', alpha=0.6, label='遠日点(180)')
        ax.axvline(165, color='green', ls='--', alpha=0.5, label='Dawn柱密度ピーク(165)')
        if FRONT_HALF_ONLY:
            ax.axvspan(180, 360, color='gray', alpha=0.06)
        ax.set_ylabel('在庫 [atoms]')
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel('TAA [deg]')
    axes[-1].set_xlim(0, 360)
    axes[-1].xaxis.set_major_locator(MultipleLocator(60))
    fig.suptitle(f'放出オフ vs 標準 — {side}側在庫ピークの比較\n'
                 '灰色帯=放出オフで崩れやすい後半(評価から除外可)', fontsize=13)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # サマリー
    # ==========================================
    print("\n" + "=" * 78)
    print(f"=== 在庫ピークTAA サマリー ({side}側) ===")
    print("=" * 78)
    print(f"{'run':<20}{'生ピーク':<12}{'デトレンド後':<14}{'ドリフト量':<12}")
    print("-" * 78)
    for label, r in results.items():
        print(f"{label:<20}{r['raw_peak']:<12.0f}{r['detr_peak']:<14.0f}{r['drift']:<+12.1f}%")
    print("=" * 78)

    labels = list(results.keys())
    if len(labels) == 2:
        a, b = results[labels[0]], results[labels[1]]
        print("\n[判定]")
        print(f"  生ピークの差       : {b['raw_peak'] - a['raw_peak']:+.0f}°")
        print(f"  デトレンド後の差   : {b['detr_peak'] - a['detr_peak']:+.0f}°")
        max_drift = max(abs(a['drift']), abs(b['drift']))
        shift = abs(b['detr_peak'] - a['detr_peak'])
        print(f"  最大ドリフト量     : {max_drift:.1f}%")
        print()
        if abs(b['detr_peak'] - a['detr_peak']) <= 8:
            print("  → 放出オフでもピークがほぼ動かない")
            print("     = ピーク位置は放出でなく在庫パターンで決まっている(パターン起源)")
        else:
            print("  → 放出オフでピークが動く = 年内の放出がピーク位置に寄与している")
        if max_drift > 20:
            print(f"  ※ ドリフトが大きい({max_drift:.0f}%)ので生データの後半は信頼性低。")
            print("    デトレンド後の値、または前半(TAA<180)を主に見ること。")


if __name__ == "__main__":
    if not os.path.exists(ORBIT_FILE_PATH):
        print(f"エラー: 軌道ファイルなし {ORBIT_FILE_PATH}")
    else:
        orbit, t_start = load_orbit_data(ORBIT_FILE_PATH)
        print("軌道ファイル読み込み完了\n")
        for side in ["DAWN", "DUSK", "FULL"]:
            evaluate(MODELS, orbit, t_start, WINDOW_EFFCOS, side=side)