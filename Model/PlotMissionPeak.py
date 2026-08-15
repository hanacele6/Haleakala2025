# -*- coding: utf-8 -*-
"""
在庫と放出量のピーク TAA の Q 依存

4つの指標を同じ土俵で比較する。

  A. 重みなしの在庫          … 領域制限あり (既定: Dawn 側の昼面)
  B. 放出量                  … budget_statistics_per_taa.csv の Gen_* (記録値)
  C. 低天頂角の在庫          … cos(天頂角) がしきい値以上の領域だけ
  W. 放出率で重み付けした在庫 … 1ステップで実際に放出される割合を重みにする

W の重みは (1 - exp(-rate*dt)) で、0〜1 に収まる「そのセルの在庫のうち
1ステップで放出される割合」。放出量そのものではないので、Gen_* の代わりには
使えないが、天頂角による領域制限のような恣意性がなく、低天頂角が自然に
重視されるという利点がある。
"""

import os
import re
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
from matplotlib.ticker import MultipleLocator

# ==========================================
# 設定
# ==========================================
BASE_DIR = r"./SimulationResult_202607"
ORBIT_FILE_PATH = 'orbit2025_spice_unwrapped.txt'

MODELS = {
    0.1:  "New_ParabolicHop_72x36_NoEq_DT100_0804_BD0.8_UG1.85_Q0.1_A2.0e+07_LT190k_15yr",
    0.27: "Def2_ParabolicHop_72x36_NoEq_DT100_0729_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr",
    1.0:  "ParabolicHop_72x36_NoEq_DT100_0710_BD0.8_UG1.85_Q1.0_A2.0e+07_LT190k_15yr",
    2.0:  "Def2_ParabolicHop_72x36_NoEq_DT100_0729_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr",
    3.0:  "ParabolicHop_72x36_NoEq_DT100_0710_BD0.8_UG1.85_Q3.0_A2.0e+07_LT190k_15yr",
    5.0:  "New_ParabolicHop_72x36_NoEq_DT100_0804_BD0.8_UG1.85_Q5.0_A2.0e+07_LT190k_15yr",
    10.0: "New_ParabolicHop_72x36_NoEq_DT100_0804_BD0.8_UG1.85_Q10.0_A2.0e+07_LT190k_15yr",
}

TARGET_YEAR = 15
FILE_STRIDE = 1              # 重ければ 2〜5 に

# Dawn 側だけを見るか ('dawn' / 'dusk' / 'both')
SIDE = 'dawn'

# A: 重みなし在庫の領域。太陽直下点からの角度 [deg]、昼面のみ
#    (0 = 太陽直下点, 90 = ターミネーター)
STOCK_REGION_DEG = (0.0, 90.0)

# C: 低天頂角の在庫。cos(天頂角) のしきい値
#    cos >= 0.5446 が天頂角 0〜57°。0.5 なら 0〜60°
COS_THRESHOLD = 0.5446

# 領域制限の任意性を確認するために、追加で試す設定
EXTRA_COS_THRESHOLDS = [0.3, 0.5446, 0.8]

# ピーク検出 (PeakTAA_vs_Q.py と揃える)
PEAK_WINDOW = (90.0, 270.0)
FIT_HALFWIDTH_DEG = 50.0
TAA_STEP = 1.0

SHOW_PLOT = True
SAVE_PNG = True

# ==========================================
# 物理定数 (Visualize_SurfaceDensity6_1.py と一致させる)
# ==========================================
EV_TO_J = 1.602176634e-19
K_B = 1.380649e-23
NU_0 = 1.0e13

DT_SIM = 100.0
TEMP_BASE, TEMP_AMP, TEMP_NIGHT = 100.0, 600.0, 100.0
USE_AREA_WEIGHTED_FLUX = False
U_MIN_EV, U_MAX_EV = 1.4, 2.7
F_UV_1AU_M2 = 1.5e14 * (100 ** 2)
Q_PSD_UNIT_CONV = 1.0 / (100.0 ** 2)

INCLUDE_SWS = True
SWS_FLUX_1AU = 10.0 * 100 ** 3 * 400e3
SWS_YIELD = 0.06
SWS_REF_DENS = 7.5e14 * 100 ** 2
SWS_LON_RANGE = (-40.0, 40.0)
SWS_LAT_N, SWS_LAT_S = (20.0, 80.0), (-80.0, -20.0)

N_LON, N_LAT = 72, 36
R_BODY_KM = 2440.0
MERCURY_YEAR_SEC = 87.969 * 86400


# ==========================================
# 補助
# ==========================================
def load_orbit(path):
    d = np.loadtxt(path)
    d[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(d[:, 0])))
    d[:, 5] = np.rad2deg(np.unwrap(np.deg2rad(d[:, 5])))
    return d, d[0, 2]


def orbit_at(time_h, orbit, t0):
    tc = orbit[:, 2]
    t = np.clip(t0 + float(time_h) * 3600.0, tc[0], tc[-1])
    return np.interp(t, tc, orbit[:, 5]), np.interp(t, tc, orbit[:, 1])


def dawn_sign(orbit):
    t, sl = orbit[:, 2], orbit[:, 5]
    if t[-1] == t[0]:
        return 1
    return -1 if (sl[-1] - sl[0]) / (t[-1] - t[0]) > 0 else 1


def areas_m2():
    r = R_BODY_KM * 1e3
    dlon = 2 * np.pi / N_LON
    ed = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    a = np.zeros((N_LAT, N_LON))
    for i in range(N_LAT):
        a[i, :] = r ** 2 * dlon * (np.sin(ed[i + 1]) - np.sin(ed[i]))
    return a


def parse_q(name):
    m = re.search(r'Q(\d+\.\d+)', name)
    return float(m.group(1)) if m else 2.0


def parse_u(name):
    m = re.search(r'U[GF](\d+\.\d+)', name)
    return ('gaussian_random' if 'UG' in name else 'fixed',
            float(m.group(1)) if m else 1.85)


def get_bins(n, u_model, u_mu, q_coeff):
    u = np.full(n, u_mu) if u_model == 'fixed' else np.linspace(U_MIN_EV, U_MAX_EV, n)
    return u, np.full(n, q_coeff * 1.0e-20 * Q_PSD_UNIT_CONV)


def list_surface_files(d, year):
    out = []
    for f in glob.glob(os.path.join(d, "density_grid_*.npy")):
        m = re.search(r'_t(\d+)_taa(\d+)\.npy$', os.path.basename(f))
        if not m:
            continue
        th, taa = int(m.group(1)), int(m.group(2))
        if int(th * 3600.0 // MERCURY_YEAR_SEC) + 1 != year:
            continue
        sp = os.path.join(d, f"surface_density_t{th:05d}.npy")
        if os.path.exists(sp):
            out.append({'taa': taa, 'time_h': th, 'path': sp})
    out.sort(key=lambda x: x['time_h'])
    return out


# ==========================================
# ピーク検出
# ==========================================
def _angdiff(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def window_mask(taa, w):
    if w is None:
        return np.ones(len(taa), dtype=bool)
    lo, hi = w
    return (taa >= lo) & (taa <= hi) if lo <= hi else (taa >= lo) | (taa <= hi)


def peak_taa(taa, y, window=PEAK_WINDOW, half=FIT_HALFWIDTH_DEG):
    taa = np.asarray(taa, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    grid = np.arange(0.0, 360.0, TAA_STEP)
    yy = np.interp(grid, taa[ok], y[ok], period=360.0)

    m = window_mask(grid, window)
    idx = np.where(m)[0]
    p0 = float(grid[idx[np.argmax(yy[idx])]])
    d = _angdiff(grid, p0)
    sel = np.abs(d) <= half
    if sel.sum() < 4:
        return p0
    c = np.polyfit(d[sel], yy[sel], 2)
    if c[0] >= 0:
        return p0
    return float((p0 - c[1] / (2 * c[0])) % 360.0)


# ==========================================
# 在庫の集計 (A / C / W)
# ==========================================
def analyze_stock(subdir, orbit, t0, ds, areas):
    path = os.path.join(BASE_DIR, subdir)
    files = list_surface_files(path, TARGET_YEAR)[::FILE_STRIDE]
    if not files:
        raise RuntimeError(f"Year {TARGET_YEAR} の surface_density がありません: {path}")

    u_model, u_mu = parse_u(subdir)
    q_coeff = parse_q(subdir)

    lon_c = np.linspace(-180, 180, N_LON + 1)[:-1] + 360.0 / N_LON / 2.0
    lat_c = np.linspace(-90, 90, N_LAT + 1)[:-1] + 180.0 / N_LAT / 2.0

    taas = []
    recA, recW = [], []
    recC = {th: [] for th in EXTRA_COS_THRESHOLDS}

    for f in files:
        try:
            surf = np.load(f['path'], allow_pickle=False)
        except Exception:
            continue
        if surf.ndim != 3 or surf.shape[0] != N_LON:
            continue
        surf = np.clip(np.nan_to_num(surf, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

        sub_lon, AU = orbit_at(f['time_h'], orbit, t0)
        lon_sun = (lon_c - sub_lon + 180.0) % 360.0 - 180.0
        lon_d = lon_sun * (1.0 if ds > 0 else -1.0)   # 負 = dawn

        cos2d = (np.cos(np.deg2rad(lat_c))[:, None] *
                 np.cos(np.deg2rad(lon_sun))[None, :])
        cos_safe = np.maximum(cos2d, 0.0)

        # --- 半球マスク ---
        if SIDE == 'dawn':
            side_m = lon_d < 0.0
        elif SIDE == 'dusk':
            side_m = lon_d > 0.0
        else:
            side_m = np.ones(N_LON, dtype=bool)
        side2d = np.broadcast_to(side_m[None, :], (N_LAT, N_LON))

        dens = surf.sum(axis=2).T                     # (LAT, LON)

        # --- A: 領域制限つき在庫 ---
        ang = np.rad2deg(np.arccos(np.clip(cos2d, -1, 1)))
        mA = (ang >= STOCK_REGION_DEG[0]) & (ang <= STOCK_REGION_DEG[1]) & side2d
        recA.append(float(np.sum(dens * areas * mA)))

        # --- C: 低天頂角の在庫 (cos しきい値ごと) ---
        for th in EXTRA_COS_THRESHOLDS:
            mC = (cos2d >= th) & side2d
            recC[th].append(float(np.sum(dens * areas * mC)))

        # --- W: 放出率で重み付けした在庫 ---
        temp = TEMP_BASE + TEMP_AMP * (cos_safe ** 0.25) * np.sqrt(0.306 / AU)
        illum = np.where(cos2d > 0.0, 1.0, 0.0)
        if INCLUDE_SWS:
            ml = (lon_sun >= SWS_LON_RANGE[0]) & (lon_sun <= SWS_LON_RANGE[1])
            mt = (((lat_c >= SWS_LAT_N[0]) & (lat_c <= SWS_LAT_N[1])) |
                  ((lat_c >= SWS_LAT_S[0]) & (lat_c <= SWS_LAT_S[1])))
            rate_sws = np.where(mt[:, None] & ml[None, :],
                                (SWS_FLUX_1AU / AU ** 2) * SWS_YIELD / SWS_REF_DENS, 0.0)
        else:
            rate_sws = 0.0
        psd_geom = (F_UV_1AU_M2 / AU ** 2) * cos_safe * illum
        u_bins, q_bins = get_bins(surf.shape[2], u_model, u_mu, q_coeff)

        wtot = 0.0
        for b in range(surf.shape[2]):
            u_j = u_bins[b] * EV_TO_J
            r_psd = psd_geom * q_bins[b]
            r_day = np.where(temp >= 10.0,
                             NU_0 * np.exp(np.maximum(-u_j / (K_B * temp), -700.0)), 0.0)
            r_night = NU_0 * np.exp(max(-u_j / (K_B * TEMP_NIGHT), -700.0))
            r_td = np.where(illum > 0.5, r_day, r_night)
            r = r_psd + r_td + rate_sws
            # 1ステップで放出される割合 (0〜1 に収まるので発散しない)
            frac = 1.0 - np.exp(-np.minimum(r * DT_SIM, 700.0))
            wtot += float(np.sum(surf[:, :, b].T * frac * areas * side2d))
        recW.append(wtot)

        taas.append(float(f['taa']))

    idx = np.argsort(taas)
    taas = np.array(taas)[idx]
    out = {'A': np.array(recA)[idx], 'W': np.array(recW)[idx]}
    for th in EXTRA_COS_THRESHOLDS:
        out[f'C{th}'] = np.array(recC[th])[idx]
    return taas, out


# ==========================================
# 放出量 (B)
# ==========================================
def load_emission(subdir, side=SIDE):
    path = os.path.join(BASE_DIR, subdir, "budget_statistics_per_taa.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path).sort_values('TAA_Bin')
    taa = df['TAA_Bin'].to_numpy(float)
    suf = {'dawn': '_Dawn', 'dusk': '_Dusk', 'both': ''}[side]
    tot = np.zeros_like(taa)
    for p in ('PSD', 'TD', 'SWS', 'MMV'):
        c = f'Gen_{p}{suf}'
        if c not in df.columns:
            c = f'Gen_{p}'
        if c in df.columns:
            tot = tot + df[c].to_numpy(float)
    return taa, tot


# ==========================================
# メイン
# ==========================================
def main():
    orbit, t0 = load_orbit(ORBIT_FILE_PATH)
    ds = dawn_sign(orbit)
    areas = areas_m2()
    print(f"dawn 判定: dawn = lon_sun {'<' if ds > 0 else '>'} 0   / 対象: {SIDE}")

    res = {}
    for q, sub in sorted(MODELS.items()):
        if not sub:
            continue
        print(f"\n[Q={q}] {sub}")
        try:
            taa_s, st = analyze_stock(sub, orbit, t0, ds, areas)
        except Exception as e:
            print(f"  在庫でエラー: {e}")
            continue
        try:
            taa_e, em = load_emission(sub)
        except Exception as e:
            print(f"  放出量でエラー: {e}")
            taa_e, em = None, None
        res[q] = dict(taa_s=taa_s, st=st, taa_e=taa_e, em=em)
        print("  OK")

    if not res:
        print("有効なデータがありませんでした。")
        return

    qs = sorted(res.keys())
    cmain = f'C{COS_THRESHOLD}'
    rows = []
    for q in qs:
        r = res[q]
        pA = peak_taa(r['taa_s'], r['st']['A'])
        pW = peak_taa(r['taa_s'], r['st']['W'])
        pC = peak_taa(r['taa_s'], r['st'].get(cmain, r['st']['A']))
        pB = peak_taa(r['taa_e'], r['em']) if r['em'] is not None else np.nan
        rows.append((q, pA, pC, pW, pB))

    print("\n" + "=" * 84)
    print(f"{'Q':>6} {'A 在庫':>10} {'C 低天頂角':>11} {'W 率重み':>10} "
          f"{'B 放出量':>10} {'B-A':>8} {'B-W':>8}")
    print("-" * 84)
    for q, pA, pC, pW, pB in rows:
        print(f"{q:>6} {pA:>10.1f} {pC:>11.1f} {pW:>10.1f} {pB:>10.1f} "
              f"{_angdiff(pB, pA):>8.1f} {_angdiff(pB, pW):>8.1f}")
    print("=" * 84)
    print("B-A が Q によらず一定 → 在庫の時間分布がピーク位置を決めている")
    print("B-A が Q とともに変わる → 放出率の側も独立に効いている")

    print("\n--- 領域制限の任意性チェック (C のしきい値を変える) ---")
    hdr = "  ".join(f'cos≥{th}' for th in EXTRA_COS_THRESHOLDS)
    print(f"{'Q':>6}  {hdr}")
    for q in qs:
        vals = [peak_taa(res[q]['taa_s'], res[q]['st'][f'C{th}'])
                for th in EXTRA_COS_THRESHOLDS]
        print(f"{q:>6}  " + "  ".join(f"{v:7.1f}" for v in vals))

    with open('stock_emission_peaks.csv', 'w', encoding='utf-8') as f:
        f.write('Q,peak_A_stock,peak_C_lowzenith,peak_W_weighted,peak_B_emission\n')
        for q, pA, pC, pW, pB in rows:
            f.write(f"{q},{pA:.2f},{pC:.2f},{pW:.2f},{pB:.2f}\n")
    print("\n  -> stock_emission_peaks.csv")

    # ---------- 図 ----------
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    a1, a2, a3, a4, a5, a6 = axes.ravel()
    arr = np.array(rows, dtype=float)

    a1.plot(arr[:, 0], arr[:, 1], 'o-', color='royalblue', lw=2, ms=7, label='A 在庫 (領域制限)')
    a1.plot(arr[:, 0], arr[:, 2], 's-', color='seagreen', lw=2, ms=6,
            label=f'C 低天頂角 (cos≥{COS_THRESHOLD})')
    a1.plot(arr[:, 0], arr[:, 3], '^-', color='darkorange', lw=2, ms=6, label='W 率重み在庫')
    a1.plot(arr[:, 0], arr[:, 4], 'D-', color='crimson', lw=2.4, ms=7, label='B 放出量')
    a1.axhline(180, color='gray', ls=':', lw=1.2, label='遠日点')
    a1.set_xscale('log')
    a1.set_xlabel('Q  [×10⁻²⁰ cm²]')
    a1.set_ylabel('ピークの TAA [deg]')
    a1.set_title(f'ピーク TAA の Q 依存 ({SIDE})', fontsize=12)
    a1.grid(True, which='both', ls='--', alpha=0.4)
    a1.legend(fontsize=8)

    a2.plot(arr[:, 0], _angdiff(arr[:, 4], arr[:, 1]), 'o-', color='royalblue',
            lw=2, ms=7, label='B − A')
    a2.plot(arr[:, 0], _angdiff(arr[:, 4], arr[:, 2]), 's-', color='seagreen',
            lw=2, ms=6, label='B − C')
    a2.plot(arr[:, 0], _angdiff(arr[:, 4], arr[:, 3]), '^-', color='darkorange',
            lw=2, ms=6, label='B − W')
    a2.axhline(0, color='gray', ls='--', alpha=0.7)
    a2.set_xscale('log')
    a2.set_xlabel('Q  [×10⁻²⁰ cm²]')
    a2.set_ylabel('放出量ピーク − 在庫ピーク [deg]')
    a2.set_title('ピークのずれ — 一定なら在庫が決めている', fontsize=12)
    a2.grid(True, which='both', ls='--', alpha=0.4)
    a2.legend(fontsize=9)

    cmap = plt.get_cmap('viridis')
    lo, hi = np.log10(min(qs)), np.log10(max(qs))

    def _prof(ax, getter, title, peak_col):
        for q in qs:
            c = cmap((np.log10(q) - lo) / (hi - lo) if hi > lo else 0.5)
            x, y = getter(res[q])
            if y is None or x is None or not np.isfinite(y).any():
                continue
            ax.plot(x, y / np.nanmax(y) if np.nanmax(y) else y, '-',
                    color=c, lw=2, label=f'Q={q}')
        for r_ in rows:
            c = cmap((np.log10(r_[0]) - lo) / (hi - lo) if hi > lo else 0.5)
            if np.isfinite(r_[peak_col]):
                ax.axvline(r_[peak_col], color=c, ls=':', lw=1.1)
        ax.axvline(180, color='gray', ls=':', alpha=0.6)
        ax.set_xlim(0, 360)
        ax.xaxis.set_major_locator(MultipleLocator(60))
        ax.set_xlabel('TAA [deg]')
        ax.set_ylabel('規格化された量')
        ax.set_title(title, fontsize=12)
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=8, ncol=2)

    _prof(a3, lambda r: (r['taa_s'], r['st']['A']),
          'A 在庫 (領域制限)', 1)
    _prof(a4, lambda r: (r['taa_s'], r['st'].get(cmain)),
          f'C 低天頂角 (cos≥{COS_THRESHOLD})', 2)
    _prof(a5, lambda r: (r['taa_s'], r['st']['W']),
          'W 放出率で重み付けした在庫', 3)
    _prof(a6, lambda r: (r['taa_e'], r['em']),
          'B 放出量 (記録値)', 4)

    plt.tight_layout()
    if SAVE_PNG:
        fig.savefig(f'stock_emission_peaks_{SIDE}.png', dpi=140, bbox_inches='tight')
        print(f"  -> stock_emission_peaks_{SIDE}.png")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()