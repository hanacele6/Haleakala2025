# -*- coding: utf-8 -*-
"""
放出率で重み付けした在庫 W の解析

  W = Σ_bins Σ_cells  在庫[atoms/m²] × (1 - exp(-rate × dt)) × セル面積

(1 - exp(-rate·dt)) は「そのセル・そのビンの在庫のうち 1 ステップで放出される割合」で、
0〜1 に収まるので発散しない。天頂角による領域制限のような恣意的な境界を置かずに、
放出が実際に起きている場所を自動的に重視できる。

分けて出すもの
  - Dawn 半球 / Dusk 半球 / 全球
  - 昼面のみ / 夜側も含む      (INCLUDE_NIGHT)
  - 全過程の重み / PSD だけの重み / TD だけの重み

注意: W は 在庫 × 放出率 なので、放出量の定義そのものに近い。
      「W で放出量を再現できた」ことに意味はない。W の使いどころは、
      天頂角で領域を切った場合と同じ答えが出るかを確かめて、
      境界の取り方の任意性を消すことにある。
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
    #0.1:  "New_ParabolicHop_72x36_NoEq_DT100_0804_BD0.8_UG1.85_Q0.1_A2.0e+07_LT190k_15yr",
    0.27: "Def2_ParabolicHop_72x36_NoEq_DT100_0729_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr",
    #1.0:  "ParabolicHop_72x36_NoEq_DT100_0710_BD0.8_UG1.85_Q1.0_A2.0e+07_LT190k_15yr",
    2.0:  "Def2_ParabolicHop_72x36_NoEq_DT100_0729_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr",
    #3.0:  "ParabolicHop_72x36_NoEq_DT100_0710_BD0.8_UG1.85_Q3.0_A2.0e+07_LT190k_15yr",
    #5.0:  "New_ParabolicHop_72x36_NoEq_DT100_0804_BD0.8_UG1.85_Q5.0_A2.0e+07_LT190k_15yr",
    #10.0: "New_ParabolicHop_72x36_NoEq_DT100_0804_BD0.8_UG1.85_Q10.0_A2.0e+07_LT190k_15yr",
}

TARGET_YEAR = 15
FILE_STRIDE = 1              # 重ければ 2〜5 に

INCLUDE_NIGHT = False        # False: 昼面のみ (cos>0)。True: 夜側も含める

# ピーク検出
PEAK_WINDOW = (90.0, 270.0)
FIT_HALFWIDTH_DEG = 50.0
TAA_STEP = 1.0

SHOW_PLOT = True
SAVE_PNG = True

Q_LABEL = r'Q  [$\times 10^{-20}$ cm$^2$]'

# ==========================================
# 物理定数 (Visualize_SurfaceDensity6_1.py と一致させる)
# ==========================================
EV_TO_J = 1.602176634e-19
K_B = 1.380649e-23
NU_0 = 1.0e13

DT_SIM = 100.0
TEMP_BASE, TEMP_AMP, TEMP_NIGHT = 100.0, 600.0, 100.0
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

# 右下パネル用: 記録された放出量 (budget_statistics_per_taa.csv)
CSV_NAME = "budget_statistics_per_taa.csv"

REGIONS = ['dawn', 'dusk', 'all']
WEIGHTS = ['ALL', 'PSD', 'TD']


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
    taa, y = np.asarray(taa, float), np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    grid = np.arange(0.0, 360.0, TAA_STEP)
    yy = np.interp(grid, taa[ok], y[ok], period=360.0)
    idx = np.where(window_mask(grid, window))[0]
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
# W の計算
# ==========================================
def analyze_W(subdir, orbit, t0, ds, areas):
    path = os.path.join(BASE_DIR, subdir)
    files = list_surface_files(path, TARGET_YEAR)[::FILE_STRIDE]
    if not files:
        raise RuntimeError(f"Year {TARGET_YEAR} の surface_density がありません: {path}")

    u_model, u_mu = parse_u(subdir)
    q_coeff = parse_q(subdir)

    lon_c = np.linspace(-180, 180, N_LON + 1)[:-1] + 360.0 / N_LON / 2.0
    lat_c = np.linspace(-90, 90, N_LAT + 1)[:-1] + 180.0 / N_LAT / 2.0

    taas = []
    rec = {(r, w): [] for r in REGIONS for w in WEIGHTS}

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
        lon_d = lon_sun * (1.0 if ds > 0 else -1.0)     # 負 = dawn

        cos2d = (np.cos(np.deg2rad(lat_c))[:, None] *
                 np.cos(np.deg2rad(lon_sun))[None, :])
        cos_safe = np.maximum(cos2d, 0.0)
        illum = np.where(cos2d > 0.0, 1.0, 0.0)

        day_m = np.ones_like(cos2d, dtype=bool) if INCLUDE_NIGHT else (cos2d > 0.0)
        masks = {
            'dawn': np.broadcast_to((lon_d < 0.0)[None, :], cos2d.shape) & day_m,
            'dusk': np.broadcast_to((lon_d > 0.0)[None, :], cos2d.shape) & day_m,
            'all':  day_m,
        }

        temp = TEMP_BASE + TEMP_AMP * (cos_safe ** 0.25) * np.sqrt(0.306 / AU)
        if INCLUDE_SWS:
            ml = (lon_sun >= SWS_LON_RANGE[0]) & (lon_sun <= SWS_LON_RANGE[1])
            mt = (((lat_c >= SWS_LAT_N[0]) & (lat_c <= SWS_LAT_N[1])) |
                  ((lat_c >= SWS_LAT_S[0]) & (lat_c <= SWS_LAT_S[1])))
            r_sws = np.where(mt[:, None] & ml[None, :],
                             (SWS_FLUX_1AU / AU ** 2) * SWS_YIELD / SWS_REF_DENS, 0.0)
        else:
            r_sws = np.zeros_like(cos2d)
        psd_geom = (F_UV_1AU_M2 / AU ** 2) * cos_safe * illum
        u_bins, q_bins = get_bins(surf.shape[2], u_model, u_mu, q_coeff)

        acc = {k: 0.0 for k in rec}
        for b in range(surf.shape[2]):
            u_j = u_bins[b] * EV_TO_J
            r_psd = psd_geom * q_bins[b]
            r_day = np.where(temp >= 10.0,
                             NU_0 * np.exp(np.maximum(-u_j / (K_B * temp), -700.0)), 0.0)
            r_night = NU_0 * np.exp(max(-u_j / (K_B * TEMP_NIGHT), -700.0))
            r_td = np.where(illum > 0.5, r_day, r_night)

            dens_b = surf[:, :, b].T * areas          # (LAT, LON) [atoms]
            fr = {
                'ALL': 1.0 - np.exp(-np.minimum((r_psd + r_td + r_sws) * DT_SIM, 700.0)),
                'PSD': 1.0 - np.exp(-np.minimum(r_psd * DT_SIM, 700.0)),
                'TD':  1.0 - np.exp(-np.minimum(r_td * DT_SIM, 700.0)),
            }
            for rg in REGIONS:
                m = masks[rg]
                for w in WEIGHTS:
                    acc[(rg, w)] += float(np.sum(dens_b * fr[w] * m))

        for k in rec:
            rec[k].append(acc[k])
        taas.append(float(f['taa']))

    idx = np.argsort(taas)
    return np.array(taas)[idx], {k: np.array(v)[idx] for k, v in rec.items()}


def load_gen_total(subdir, side='dawn'):
    """記録された放出量の軌道合計 [atoms] を返す。無ければ nan。"""
    path = os.path.join(BASE_DIR, subdir, CSV_NAME)
    if not os.path.exists(path):
        return np.nan
    try:
        df = pd.read_csv(path)
    except Exception:
        return np.nan
    suf = {'dawn': '_Dawn', 'dusk': '_Dusk', 'all': ''}[side]
    tot = 0.0
    for p in ('PSD', 'TD', 'SWS', 'MMV'):
        c = f'Gen_{p}{suf}'
        if c not in df.columns:
            c = f'Gen_{p}'
        if c in df.columns:
            tot += float(df[c].sum())
    return tot if tot > 0 else np.nan


# ==========================================
# メイン
# ==========================================
def main():
    orbit, t0 = load_orbit(ORBIT_FILE_PATH)
    ds = dawn_sign(orbit)
    areas = areas_m2()
    print(f"dawn 判定: dawn = lon_sun {'<' if ds > 0 else '>'} 0")
    print(f"領域: {'昼面+夜側' if INCLUDE_NIGHT else '昼面のみ'}")

    res = {}
    for q, sub in sorted(MODELS.items()):
        if not sub:
            continue
        print(f"\n[Q={q}] {sub}")
        try:
            res[q] = analyze_W(sub, orbit, t0, ds, areas)
            print("  OK")
        except Exception as e:
            print(f"  エラー: {e}")

    if not res:
        print("有効なデータがありませんでした。")
        return

    qs = sorted(res.keys())
    peaks = {(rg, w): np.array([peak_taa(res[q][0], res[q][1][(rg, w)]) for q in qs])
             for rg in REGIONS for w in WEIGHTS}
    totals = {(rg, w): np.array([res[q][1][(rg, w)].mean() for q in qs])
              for rg in REGIONS for w in WEIGHTS}

    print("\n" + "=" * 76)
    print("W のピーク TAA [deg]")
    print(f"{'Q':>6} {'dawn ALL':>10} {'dawn PSD':>10} {'dawn TD':>9} "
          f"{'dusk ALL':>10} {'全球 ALL':>10}")
    print("-" * 76)
    for i, q in enumerate(qs):
        print(f"{q:>6} {peaks[('dawn','ALL')][i]:>10.1f} {peaks[('dawn','PSD')][i]:>10.1f} "
              f"{peaks[('dawn','TD')][i]:>9.1f} {peaks[('dusk','ALL')][i]:>10.1f} "
              f"{peaks[('all','ALL')][i]:>10.1f}")
    print("=" * 76)

    with open('W_analysis.csv', 'w', encoding='utf-8') as f:
        f.write('Q,' + ','.join(f'peak_{r}_{w}' for r in REGIONS for w in WEIGHTS)
                + ',' + ','.join(f'mean_{r}_{w}' for r in REGIONS for w in WEIGHTS) + '\n')
        for i, q in enumerate(qs):
            f.write(f"{q}," + ','.join(f"{peaks[(r,w)][i]:.2f}" for r in REGIONS for w in WEIGHTS)
                    + ',' + ','.join(f"{totals[(r,w)][i]:.6e}" for r in REGIONS for w in WEIGHTS) + '\n')
    print("  -> W_analysis.csv")

    # ---------- 図 ----------
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    a1, a2, a3, a4, a5, a6 = axes.ravel()
    cmap = plt.get_cmap('viridis')
    lo, hi = np.log10(min(qs)), np.log10(max(qs))
    col = {q: cmap((np.log10(q) - lo) / (hi - lo) if hi > lo else 0.5) for q in qs}

    def prof(ax, rg, w, title):
        for q in qs:
            taa, d = res[q]
            y = d[(rg, w)]
            mx = np.nanmax(y)
            ax.plot(taa, y / mx if mx else y, '-', color=col[q], lw=2, label=f'Q={q}')
            p = peak_taa(taa, y)
            if np.isfinite(p):
                ax.axvline(p, color=col[q], ls=':', lw=1.1)
        ax.axvline(180, color='gray', ls=':', alpha=0.6)
        ax.set_xlim(0, 360)
        ax.xaxis.set_major_locator(MultipleLocator(60))
        ax.set_xlabel('TAA [deg]')
        ax.set_ylabel('規格化された W')
        ax.set_title(title, fontsize=12)
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=8, ncol=2)

    prof(a1, 'dawn', 'ALL', 'W — Dawn 半球 (全過程)')
    prof(a2, 'dusk', 'ALL', 'W — Dusk 半球 (全過程)')

    a3.plot(qs, peaks[('dawn', 'ALL')], 'o-', color='crimson', lw=2.2, ms=7, label='Dawn')
    a3.plot(qs, peaks[('dusk', 'ALL')], 's-', color='royalblue', lw=2.2, ms=7, label='Dusk')
    a3.plot(qs, peaks[('all', 'ALL')], '^--', color='dimgray', lw=1.8, ms=6, label='全球')
    a3.axhline(180, color='gray', ls=':', lw=1.2, label='遠日点')
    a3.set_xscale('log')
    a3.set_xlabel(Q_LABEL)
    a3.set_ylabel('W のピーク TAA [deg]')
    a3.set_title('W ピークの Q 依存 — Dawn と Dusk', fontsize=12)
    a3.grid(True, which='both', ls='--', alpha=0.4)
    a3.legend(fontsize=9)

    prof(a4, 'dawn', 'PSD', 'W — Dawn / PSD の重みだけ')

    a5.plot(qs, peaks[('dawn', 'ALL')], 'o-', color='black', lw=2.2, ms=7, label='全過程')
    a5.plot(qs, peaks[('dawn', 'PSD')], 's--', color='royalblue', lw=1.8, ms=6, label='PSD の重み')
    a5.plot(qs, peaks[('dawn', 'TD')], '^--', color='crimson', lw=1.8, ms=6, label='TD の重み')
    a5.axhline(180, color='gray', ls=':', lw=1.2, label='遠日点')
    a5.set_xscale('log')
    a5.set_xlabel(Q_LABEL)
    a5.set_ylabel('W のピーク TAA [deg]')
    a5.set_title('過程別の重みで見た W ピーク (Dawn)', fontsize=12)
    a5.grid(True, which='both', ls='--', alpha=0.4)
    a5.legend(fontsize=9)

    gen = {rg: np.array([load_gen_total(MODELS[q], rg) for q in qs])
           for rg in ('dawn', 'dusk', 'all')}
    plotted = False
    for rg, c, mk, lb in (('dawn', 'crimson', 'o-', 'Dawn'),
                          ('dusk', 'royalblue', 's-', 'Dusk'),
                          ('all', 'dimgray', '^--', '全球')):
        v = gen[rg]
        if not np.isfinite(v).all() or v[0] <= 0:
            continue
        a6.plot(qs, v / v[0], mk, color=c, lw=2.2, ms=7, label=lb)
        plotted = True
    if plotted:
        a6.plot(qs, np.array(qs) / qs[0], ':', color='gray', lw=1.5,
                label=r'$\propto Q$ の場合')
        a6.set_xscale('log')
        a6.set_yscale('log')
        a6.set_xlabel(Q_LABEL)
        a6.set_ylabel('放出量の軌道合計 (Q最小で規格化)')
        a6.set_title('記録された放出量の Q 依存\n横ばい = 供給律速', fontsize=12)
        a6.grid(True, which='both', ls='--', alpha=0.4)
        a6.legend(fontsize=9)
    else:
        a6.text(0.5, 0.5, f'{CSV_NAME} が見つかりません', ha='center', va='center',
                transform=a6.transAxes, fontsize=11, color='gray')
        a6.set_axis_off()

    plt.tight_layout()
    if SAVE_PNG:
        tag = 'daynight' if INCLUDE_NIGHT else 'dayonly'
        fig.savefig(f'W_analysis_{tag}.png', dpi=140, bbox_inches='tight')
        print(f"  -> W_analysis_{tag}.png")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()