# -*- coding: utf-8 -*-
"""
本体固定座標での在庫の経度分布 (再run不要・.npyのみ)

--- 背景 ---
TAA=0で在庫パターンを凍結し、その後まったく消費させずに運んだだけでも
在庫ピークのTAAがQで変わる。
→ 165° と 180° の差は「消費」ではなく、
  その時点で既に本体固定パターンとして刻まれている在庫の形と経度位置で決まる。

水星は3:2共鳴なので、
  経度 0°, 180°  … 近日点で太陽直下 (hot poles)
  経度 90°, 270° … 遠日点で太陽直下 (warm poles)
という関係が永久に固定され、本体固定座標に消えない縦縞が焼き付く。
PSDの強さでこの縞の削られ方が変わり、在庫の山の経度が動くはず。

--- このスクリプトの出力 ---
[A] 本体固定の経度に対する在庫分布 (Q比較・複数TAAで重ね描き)
    hot poles / warm poles に補助線。
    ・パターンがTAAによらずほぼ不変なら「焼き付いた縞」であることの確認
    ・山の経度がQでずれていれば、それがピークシフトの起源

[B] 経度 → 放出窓の通過TAA の対応づけ
    各本体固定経度が eff_cos 窓 (既定 0.70-0.85, Dawn側) を通過するTAAを求め、
    在庫分布と重ねて「山がいつ窓を通るか」を数値で出す。
    → 在庫の山の経度から予測される放出ピークTAAが、
      実測の柱密度ピーク(165° / 180°)と合うかを確認する。
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

MODELS = {
    "Q2.0 (Standard)": "Def2_ParabolicHop_72x36_NoEq_DT100_0729_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr",
    "Q0.3 (Weak PSD)": "Def2_ParabolicHop_72x36_NoEq_DT100_0729_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr",
}
COLORS = {"Q2.0 (Standard)": "crimson", "Q0.3 (Weak PSD)": "steelblue"}

ORBIT_FILE_PATH = 'orbit2025_spice_unwrapped.txt'
N_LON, N_LAT = 72, 36
R_BODY_KM = 2439.7
MERCURY_YEAR_SEC = 87.969 * 86400

# [A] 経度分布を見るTAA (複数指定して重ねる)
#PROFILE_TAAS = [0, 60, 120, 180, 240, 300]
PROFILE_TAAS = [0]

# [B] 放出窓 (在庫ピークが柱密度ピークと一致した帯)
WINDOW_EFFCOS = (0.70, 0.85)

# 緯度の重み: True=全緯度を面積重みで合計, False=赤道付近のみ
USE_ALL_LAT = True
EQUATOR_BAND_DEG = 30.0   # USE_ALL_LAT=False のとき使う緯度範囲 ±


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


def lon_centers(n_lon):
    e = np.linspace(-180, 180, n_lon + 1)
    return (e[:-1] + e[1:]) / 2.0


def lat_centers(n_lat):
    e = np.linspace(-90, 90, n_lat + 1)
    return (e[:-1] + e[1:]) / 2.0


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
    """本体固定のまま (lat, lon) [atoms/cm^2]"""
    d = np.load(path)
    if d.ndim == 3:
        d = np.sum(d, axis=2)
    return np.nan_to_num(d.T, nan=0.0) / 10000.0


def longitude_profile(dens_T, areas):
    """緯度方向に面積重みで合計 → 経度ごとの在庫 [atoms]"""
    if USE_ALL_LAT:
        mask = np.ones_like(dens_T, dtype=bool)
    else:
        latc = lat_centers(N_LAT)
        mask = np.abs(latc)[:, None] <= EQUATOR_BAND_DEG
        mask = np.broadcast_to(mask, dens_T.shape)
    w = np.where(mask, dens_T * areas, 0.0)
    return w.sum(axis=0)   # (n_lon,)


# ==========================================
# [A] 本体固定の経度分布
# ==========================================
def plot_longitude_profiles(models, orbit, t_start, profile_taas):
    areas = cell_areas_cm2(N_LON, N_LAT, R_BODY_KM)
    lonc = lon_centers(N_LON)

    n = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    summary = {}

    for ax, (label, subdir) in zip(axes, models.items()):
        try:
            recs, yr = final_year_snapshots(os.path.join(BASE_DIR, subdir))
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            continue

        taas = np.array([r['taa'] for r in recs])
        cmap = plt.get_cmap('viridis')
        cols = [cmap(x) for x in np.linspace(0, 0.85, len(profile_taas))]

        prof_stack = []
        for ci, tgt in enumerate(profile_taas):
            i = int(np.argmin(np.abs(taas - tgt)))
            dens = load_surface_bodyfixed(recs[i]['path'])
            prof = longitude_profile(dens, areas)
            prof_stack.append(prof)
            ax.plot(lonc, prof, '-', color=cols[ci], lw=1.8, label=f'TAA={taas[i]}°')

        prof_stack = np.array(prof_stack)
        mean_prof = prof_stack.mean(axis=0)
        peak_lon = lonc[int(np.argmax(mean_prof))]
        # パターンの時間変動(TAAによらず不変か)
        var_ratio = prof_stack.std(axis=0).mean() / mean_prof.mean() if mean_prof.mean() > 0 else np.nan
        summary[label] = dict(peak_lon=peak_lon, mean_prof=mean_prof, var_ratio=var_ratio)

        #ax.plot(lonc, mean_prof, '--', color='black', lw=2.0, alpha=0.7, label='TAA平均')
        for hp in (0, 180, -180):
            ax.axvline(hp, color='crimson', ls=':', alpha=0.6)
        for wp in (90, -90):
            ax.axvline(wp, color='royalblue', ls=':', alpha=0.6)
        ax.axvline(peak_lon, color='green', ls='-', alpha=0.5, lw=2)

        ax.set_title(f'{label}  —  在庫の山: 経度 {peak_lon:.0f}°  '
                     f'(パターンの時間変動 {var_ratio*100:.1f}%)', fontsize=11)
        ax.set_ylabel('在庫 [atoms]')
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=8, ncol=2)

    axes[-1].set_xlabel('本体固定 経度 [deg]   (赤点線=hot poles 0/180°, 青点線=warm poles ±90°)')
    axes[-1].set_xlim(-180, 180)
    axes[-1].xaxis.set_major_locator(MultipleLocator(45))
    fig.suptitle('本体固定座標での在庫の経度分布\n'
                 'TAAによらずほぼ不変なら「焼き付いた縞」。山の経度がQでずれるかが焦点', fontsize=13)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 74)
    print("=== [A] 在庫の山の経度 ===")
    print("=" * 74)
    for label, s in summary.items():
        print(f"  {label:<20} 山の経度 = {s['peak_lon']:>7.1f}°   "
              f"時間変動 = {s['var_ratio']*100:.1f}%")
    labels = list(summary.keys())
    if len(labels) == 2:
        d = summary[labels[1]]['peak_lon'] - summary[labels[0]]['peak_lon']
        print(f"\n  → 山の経度差: {d:+.1f}°")
    print("=" * 74)
    return summary


# ==========================================
# [B] 経度 → 放出窓の通過TAA
# ==========================================
def map_longitude_to_window_taa(models, orbit, t_start, window, summary):
    """各本体固定経度が eff_cos 窓(Dawn側)を通過するTAAを求める。
    赤道(cos lat = 1)基準: eff_cos = cos(lon - sub_lon), Dawn は lon_sun < 0。"""
    lo, hi = window
    # 窓に対応する lon_sun の範囲(Dawn側なので負)
    ls_hi = -np.rad2deg(np.arccos(min(hi, 1.0)))   # eff_cos大 → lon_sunは0に近い(負)
    ls_lo = -np.rad2deg(np.arccos(lo))             # eff_cos小 → より負
    print("\n" + "=" * 74)
    print(f"=== [B] 放出窓 eff_cos {lo:.2f}-{hi:.2f} (Dawn側) の通過TAA ===")
    print("=" * 74)
    print(f"  窓に対応する lon_sun: {ls_lo:.1f}° 〜 {ls_hi:.1f}°  "
          f"(天頂角 {np.rad2deg(np.arccos(hi)):.0f}°〜{np.rad2deg(np.arccos(lo)):.0f}°)")

    lonc = lon_centers(N_LON)
    fig, ax = plt.subplots(figsize=(11, 6))

    for label, subdir in models.items():
        try:
            recs, yr = final_year_snapshots(os.path.join(BASE_DIR, subdir))
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            continue
        taas = np.array([r['taa'] for r in recs])
        subl = np.array([get_subsolar_lon(r['time_h'], t_start, orbit) for r in recs])

        # 各経度について、窓に入るTAAの中央値を求める
        transit_taa = np.full(len(lonc), np.nan)
        for j, L in enumerate(lonc):
            ls = (L - subl + 180.0) % 360.0 - 180.0
            inwin = (ls >= ls_lo) & (ls <= ls_hi)
            if np.any(inwin):
                cand = taas[inwin]
                transit_taa[j] = np.median(cand)

        if label in summary:
            mp = summary[label]['mean_prof']
            pk_lon = summary[label]['peak_lon']
            j = int(np.argmin(np.abs(lonc - pk_lon)))
            pred = transit_taa[j]
            # 在庫で重み付けした通過TAA(山の広がりを考慮)
            valid = np.isfinite(transit_taa) & (mp > 0)
            wavg = (np.sum(transit_taa[valid] * mp[valid]) / np.sum(mp[valid])
                    if np.any(valid) else np.nan)
            print(f"\n  [{label}]")
            print(f"    在庫の山 経度 {pk_lon:.0f}° → 窓の通過TAA = {pred:.0f}°")
            print(f"    在庫重み付き平均の通過TAA        = {wavg:.0f}°")

            ax.plot(lonc, transit_taa, '-', color=COLORS.get(label), lw=2, label=f'{label}: 通過TAA')
            ax.axvline(pk_lon, color=COLORS.get(label), ls='--', alpha=0.6)
            if np.isfinite(pred):
                ax.plot([pk_lon], [pred], 'o', ms=10, color=COLORS.get(label),
                        markeredgecolor='black', zorder=5)

    ax.axhline(165, color='green', ls='--', alpha=0.6, label='柱密度ピーク実測 165°')
    ax.axhline(180, color='gray', ls=':', alpha=0.6, label='遠日点 180°')
    for hp in (0, 180, -180):
        ax.axvline(hp, color='crimson', ls=':', alpha=0.4)
    for wp in (90, -90):
        ax.axvline(wp, color='royalblue', ls=':', alpha=0.4)

    ax.set_xlabel('本体固定 経度 [deg]')
    ax.set_ylabel('放出窓を通過するTAA [deg]')
    ax.set_title(f'経度 → 放出窓(eff_cos {lo:.2f}-{hi:.2f})の通過TAA\n'
                 '丸印=在庫の山の経度に対応する通過TAA。165°/180°と比べる')
    ax.set_xlim(-180, 180)
    ax.xaxis.set_major_locator(MultipleLocator(45))
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    print("=" * 74)
    print("\n[判定]")
    print("  在庫の山の経度から予測される通過TAAが、")
    print("  Q2.0で180°付近・Q0.3で165°付近になっていれば、")
    print("  『本体固定パターンの山の経度差がピークシフトの起源』が確定する。")


if __name__ == "__main__":
    if not os.path.exists(ORBIT_FILE_PATH):
        print(f"エラー: 軌道ファイルなし {ORBIT_FILE_PATH}")
    else:
        orbit, t_start = load_orbit_data(ORBIT_FILE_PATH)
        print("軌道ファイル読み込み完了\n")
        summary = plot_longitude_profiles(MODELS, orbit, t_start, PROFILE_TAAS)
        map_longitude_to_window_taa(MODELS, orbit, t_start, WINDOW_EFFCOS, summary)