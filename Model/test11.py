# -*- coding: utf-8 -*-
"""
渋滞域の「通過」と「消費」を分離する解析 (再run不要・.npyスナップショットのみ使用)

--- 背景 ---
5枚目の図の最上位バンド(eff_cos 0.55-1.01)は広すぎる。
このバンド内でTD率は118倍変化するため、「TDが効く範囲の在庫」とは言えない。
また、在庫が2.5倍違うのにTD放出が1.3倍しか違わないのは、
バンド内での在庫の位置(eff_cos分布)がQで違うためと考えられる。

--- このスクリプトの2つの検証 ---
[A] 狭帯域での在庫ピーク位置
    eff_cosを細かく切り、各サブバンドで在庫のTAA分布のピークを求める。
    ・渋滞の塊が通過しているだけなら → 高eff_cosのバンドほどピークTAAが遅い
      (塊が順に通り過ぎる signature)
    ・どのバンドでも同じTAAにピーク → 通過以外の要因(消費/供給)が支配

[B] 受動輸送テスト (消費ゼロの仮想実験)
    表面在庫は「本体固定座標」で記録されているので、
    ある時刻の在庫パターンを凍結し、太陽直下点だけを進めれば
    「その後まったく消費・供給がなかった場合」の在庫分布が再現できる。
    ・実測と一致 → その区間の在庫変化は純粋な輸送(幾何)で説明できる
    ・実測が下回る → その差が消費(TD/PSD)の寄与
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

ORBIT_FILE_PATH = 'orbit2025_spice_unwrapped.txt'
N_LON, N_LAT = 72, 36
R_BODY_KM = 2439.7
MERCURY_YEAR_SEC = 87.969 * 86400

SIDE = "DAWN"   # "DAWN" / "DUSK"

# [A] 狭帯域の定義 (ターミネーター寄り → SSP寄り)
SUB_BANDS = [
    (0.40, 0.55),
    (0.55, 0.70),
    (0.70, 0.85),
    (0.85, 1.01),
]

# [B] 受動輸送テストで在庫パターンを凍結するTAA
FREEZE_TAAS = [0,]


# ==========================================
# 軌道・幾何
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
    """本体固定座標のまま eff_cos と Dawn判定を返す。data.T と同じ (lat, lon) 並び。"""
    lon_e = np.linspace(-180, 180, n_lon + 1)
    lon_c = np.deg2rad((lon_e[:-1] + lon_e[1:]) / 2.0)
    lat_e = np.linspace(-90, 90, n_lat + 1)
    lat_c = np.deg2rad((lat_e[:-1] + lat_e[1:]) / 2.0)
    lon_sun = (lon_c[None, :] - np.deg2rad(sub_lon_deg) + np.pi) % (2*np.pi) - np.pi
    eff_cos = np.cos(lat_c[:, None]) * np.cos(lon_sun)
    is_dawn = np.broadcast_to(lon_sun < 0.0, eff_cos.shape)
    return eff_cos, is_dawn


# ==========================================
# スナップショット読み込み
# ==========================================
def final_year_snapshots(target_dir):
    """最終年の (time_h, taa, surface_densityパス) をTAA順で返す"""
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
        raise FileNotFoundError(f"スナップショットが見つかりません: {target_dir}")

    max_year = max(int(r['time_h'] * 3600.0 // MERCURY_YEAR_SEC) + 1 for r in recs)
    recs = [r for r in recs if int(r['time_h'] * 3600.0 // MERCURY_YEAR_SEC) + 1 == max_year]
    recs.sort(key=lambda r: r['taa'])
    return recs, max_year


def load_surface_bodyfixed(path):
    """本体固定のまま (lat, lon) の面密度 [atoms/cm^2] を返す(rollしない)"""
    d = np.load(path)
    if d.ndim == 3:
        d = np.sum(d, axis=2)
    return np.nan_to_num(d.T, nan=0.0) / 10000.0


def stock_in_bands(dens_T, eff_cos, is_dawn, areas, sub_bands, side):
    """サブバンドごとの在庫総量 [atoms]"""
    day = eff_cos > 0.0
    mask_side = (day & is_dawn) if side == "DAWN" else (day & ~is_dawn)
    out = []
    for lo, hi in sub_bands:
        m = mask_side & (eff_cos >= lo) & (eff_cos < hi)
        out.append(np.sum(dens_T[m] * areas[m]) if np.any(m) else 0.0)
    return np.array(out)


# ==========================================
# [A] 狭帯域での在庫ピーク位置
# ==========================================
def analyze_subbands(models, orbit, t_start, sub_bands, side):
    areas = cell_areas_cm2(N_LON, N_LAT, R_BODY_KM)
    nb = len(sub_bands)

    fig, axes = plt.subplots(nb, 1, figsize=(11, 3.0 * nb), sharex=True)
    if nb == 1:
        axes = [axes]

    peak_table = {}

    for label, subdir in models.items():
        try:
            recs, yr = final_year_snapshots(os.path.join(BASE_DIR, subdir))
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            continue

        taas, stocks = [], []
        for r in recs:
            dens = load_surface_bodyfixed(r['path'])
            sl = get_subsolar_lon(r['time_h'], t_start, orbit)
            ec, isd = effcos_grid(N_LON, N_LAT, sl)
            stocks.append(stock_in_bands(dens, ec, isd, areas, sub_bands, side))
            taas.append(r['taa'])
        taas = np.array(taas)
        stocks = np.array(stocks)   # (n_snap, n_band)

        peaks = []
        for b in range(nb):
            y = stocks[:, b]
            pk = taas[int(np.nanargmax(y))] if np.any(y > 0) else np.nan
            peaks.append(pk)
            axes[b].plot(taas, y, '-o', ms=3, color=COLORS.get(label), lw=1.8,
                         label=f'{label} (peak={pk:.0f}°)')
        peak_table[label] = peaks
        print(f"[読込] {label}: Year {yr}, {len(recs)}スナップショット")

    for b, (lo, hi) in enumerate(sub_bands):
        z_hi = np.rad2deg(np.arccos(min(hi, 1.0)))
        z_lo = np.rad2deg(np.arccos(lo))
        axes[b].set_title(f'eff_cos {lo:.2f}–{hi:.2f}  (天頂角 {z_hi:.0f}°–{z_lo:.0f}°)', fontsize=11)
        axes[b].set_ylabel('在庫 [atoms]')
        axes[b].axvline(180, color='gray', ls=':', alpha=0.6)
        axes[b].axvline(165, color='green', ls='--', alpha=0.5)
        axes[b].grid(True, ls='--', alpha=0.4)
        axes[b].legend(fontsize=8)
    axes[-1].set_xlabel('TAA [deg]')
    axes[-1].set_xlim(0, 360)
    axes[-1].xaxis.set_major_locator(MultipleLocator(60))
    fig.suptitle(f'[A] 狭帯域ごとの在庫のTAA分布 ({side}側)\n'
                 '塊の通過なら → 高eff_cosのバンドほどピークが遅いはず', fontsize=13)
    plt.tight_layout()
    plt.show()

    # サマリー
    print("\n" + "=" * 72)
    print(f"=== [A] サブバンド別 在庫ピークTAA ({side}側) ===")
    print("=" * 72)
    hdr = f"{'eff_cosバンド':<18}"
    for label in peak_table:
        hdr += f"{label:<22}"
    print(hdr)
    print("-" * 72)
    for b, (lo, hi) in enumerate(sub_bands):
        line = f"{f'{lo:.2f}-{hi:.2f}':<18}"
        for label in peak_table:
            line += f"{peak_table[label][b]:<22.0f}"
        print(line)
    print("-" * 72)
    for label, pk in peak_table.items():
        valid = [p for p in pk if np.isfinite(p)]
        if len(valid) >= 2:
            trend = "遅い側へ単調移動(=塊の通過)" if valid[-1] > valid[0] + 5 else \
                    ("ほぼ同じTAA(=通過では説明できない)" if abs(valid[-1]-valid[0]) <= 5
                     else "早い側へ移動(想定外)")
            print(f"  {label}: {valid[0]:.0f}° → {valid[-1]:.0f}°  … {trend}")
    print("=" * 72)


# ==========================================
# [B] 受動輸送テスト (消費ゼロの仮想実験)
# ==========================================
def passive_advection_test(models, orbit, t_start, sub_bands, side, freeze_taas):
    """在庫パターンを凍結し太陽直下点だけ進めた場合(消費ゼロ)と実測を比較。"""
    areas = cell_areas_cm2(N_LON, N_LAT, R_BODY_KM)

    for label, subdir in models.items():
        try:
            recs, yr = final_year_snapshots(os.path.join(BASE_DIR, subdir))
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            continue

        taas = np.array([r['taa'] for r in recs])

        # 実測(全バンド合計 = 昼側全体)
        actual = []
        for r in recs:
            dens = load_surface_bodyfixed(r['path'])
            sl = get_subsolar_lon(r['time_h'], t_start, orbit)
            ec, isd = effcos_grid(N_LON, N_LAT, sl)
            actual.append(stock_in_bands(dens, ec, isd, areas, sub_bands, side).sum())
        actual = np.array(actual)

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(taas, actual, '-o', ms=3, color='black', lw=2.2, label='実測(消費あり)')

        cmap = plt.get_cmap('autumn')
        for ci, ft in enumerate(freeze_taas):
            i0 = int(np.argmin(np.abs(taas - ft)))
            frozen = load_surface_bodyfixed(recs[i0]['path'])   # 本体固定パターンを凍結

            passive = []
            for r in recs:
                sl = get_subsolar_lon(r['time_h'], t_start, orbit)
                ec, isd = effcos_grid(N_LON, N_LAT, sl)
                passive.append(stock_in_bands(frozen, ec, isd, areas, sub_bands, side).sum())
            passive = np.array(passive)

            # 凍結点以降だけ描画(それ以前は意味を持たない)
            m = taas >= taas[i0]
            ax.plot(taas[m], passive[m], '--', lw=1.8,
                    color=cmap(ci / max(len(freeze_taas)-1, 1) * 0.7),
                    label=f'受動輸送のみ (TAA={taas[i0]}°で凍結)')

        ax.axvline(180, color='gray', ls=':', alpha=0.6, label='遠日点(180)')
        ax.axvline(165, color='green', ls='--', alpha=0.5, label='移動後ピーク(165)')
        ax.set_xlabel('TAA [deg]')
        ax.set_ylabel('昼側在庫 [atoms]')
        ax.set_title(f'{label} — [B] 受動輸送テスト ({side}側)\n'
                     '破線=消費ゼロなら在庫がどう推移したか / 実線との差が消費の寄与')
        ax.set_xlim(0, 360)
        ax.xaxis.set_major_locator(MultipleLocator(60))
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.show()

        # 数値サマリー
        print(f"\n  [{label}] 受動輸送 vs 実測 (昼側{side}全体)")
        print(f"  {'凍結TAA':<10}{'受動ピーク':<12}{'実測ピーク':<12}{'TAA180での実測/受動':<20}")
        for ft in freeze_taas:
            i0 = int(np.argmin(np.abs(taas - ft)))
            frozen = load_surface_bodyfixed(recs[i0]['path'])
            passive = []
            for r in recs:
                sl = get_subsolar_lon(r['time_h'], t_start, orbit)
                ec, isd = effcos_grid(N_LON, N_LAT, sl)
                passive.append(stock_in_bands(frozen, ec, isd, areas, sub_bands, side).sum())
            passive = np.array(passive)
            m = taas >= taas[i0]
            p_pk = taas[m][int(np.nanargmax(passive[m]))]
            a_pk = taas[m][int(np.nanargmax(actual[m]))]
            i180 = int(np.argmin(np.abs(taas - 180)))
            ratio = actual[i180] / passive[i180] if passive[i180] > 0 else np.nan
            print(f"  {taas[i0]:<10.0f}{p_pk:<12.0f}{a_pk:<12.0f}{ratio:<20.3f}")


if __name__ == "__main__":
    if not os.path.exists(ORBIT_FILE_PATH):
        print(f"エラー: 軌道ファイルなし {ORBIT_FILE_PATH}")
    else:
        orbit, t_start = load_orbit_data(ORBIT_FILE_PATH)
        print(f"軌道ファイル読み込み完了\n")

        analyze_subbands(MODELS, orbit, t_start, SUB_BANDS, SIDE)
        passive_advection_test(MODELS, orbit, t_start, SUB_BANDS, SIDE, FREEZE_TAAS)