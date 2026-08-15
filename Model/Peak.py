# -*- coding: utf-8 -*-
"""
在庫と放出率のどちらがピーク位置を作っているか — 因子凍結による分離

放出量は LT ごと・束縛エネルギービンごとの積で書ける。

    P(TAA) = Σ_LT Σ_bin  σ(LT, bin, TAA) × f(LT, bin, TAA)

    σ … 在庫 [atoms]            surface_density の測定値
    f … 1ステップで放出される割合 = 1 - exp(-r·dt)   式から計算

これをそのまま足すと放出量の定義になり、何も検証できない。
そこで片方の TAA 依存を消して比べる。

    P_rate (TAA) = Σ  σ̄(LT,bin)      × f(LT,bin,TAA)   … 放出率だけ動かす
    P_stock(TAA) = Σ  σ(LT,bin,TAA)  × f̄(LT,bin)       … 在庫だけ動かす
    P_both (TAA) = Σ  σ(LT,bin,TAA)  × f(LT,bin,TAA)   … 両方 (= W)

σ̄, f̄ は軌道平均。凍結した側は TAA 依存を持たないので循環しない。

  P_rate のピークが近日点寄り  … 放出率は近日点で最大 (1/r² と高温)
  P_stock のピークが遠日点寄り … 在庫は自転が速い遠日点付近で最大
  P_both のピークがその間      … 積が最大になる TAA

Q を変えたとき、どちらの曲線が動くかでピークシフトの起源が分かる。
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
CSV_NAME = "budget_statistics_per_taa.csv"

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
FILE_STRIDE = 1

SIDE = 'dawn'                 # 'dawn' / 'dusk' / 'all'
LAT_RANGE = None              # 例 (-30.0, 30.0)。None で全緯度
INCLUDE_NIGHT = False         # 夜側も含めるか

# 太陽相対経度のビン数 (σ と f を LT ごとに保持する解像度)
N_LON_BIN = 72

TAA_STEP = 2.0
SMOOTH_DEG = 10.0
PEAK_WINDOW = (90.0, 270.0)
FIT_HALFWIDTH_DEG = 50.0

SHOW_PLOT = True
SAVE_PNG = True
Q_LABEL = r'Q  [$\times 10^{-20}$ cm$^2$]'

# ==========================================
# 物理定数
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
MERCURY_E = 0.205630
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


def resample(taa, y, step=TAA_STEP):
    ok = np.isfinite(y)
    g = np.arange(0.0, 360.0, step)
    if ok.sum() < 3:
        return g, np.full_like(g, np.nan)
    return g, np.interp(g, np.asarray(taa)[ok], np.asarray(y)[ok], period=360.0)


def circ_smooth(y, width_deg=SMOOTH_DEG, step=TAA_STEP):
    w = max(int(round(width_deg / step)), 1)
    if w <= 1:
        return y.copy()
    k = np.ones(w) / w
    ext = np.concatenate([y[-w:], y, y[:w]])
    return np.convolve(ext, k, mode='same')[w:w + len(y)]


def _angdiff(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def peak_taa(taa, y, window=PEAK_WINDOW, half=FIT_HALFWIDTH_DEG):
    taa, y = np.asarray(taa, float), np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    g, yy = resample(taa[ok], y[ok])
    if window is None:
        m = np.ones(len(g), dtype=bool)
    else:
        lo, hi = window
        m = (g >= lo) & (g <= hi) if lo <= hi else (g >= lo) | (g <= hi)
    idx = np.where(m)[0]
    if len(idx) == 0:
        return np.nan
    p0 = float(g[idx[np.argmax(yy[idx])]])
    d = _angdiff(g, p0)
    sel = np.abs(d) <= half
    if sel.sum() < 4:
        return p0
    c = np.polyfit(d[sel], yy[sel], 2)
    if c[0] >= 0:
        return p0
    return float((p0 - c[1] / (2 * c[0])) % 360.0)


def dt_per_taa_deg(taa_deg):
    e = MERCURY_E
    f = np.deg2rad(np.asarray(taa_deg, float))
    dMdf = (1 - e ** 2) ** 1.5 / (1 + e * np.cos(f)) ** 2
    return dMdf * MERCURY_YEAR_SEC / 360.0


def load_emission_rate(subdir, side=SIDE):
    p = os.path.join(BASE_DIR, subdir, CSV_NAME)
    if not os.path.exists(p):
        return None, None
    df = pd.read_csv(p).sort_values('TAA_Bin')
    taa = df['TAA_Bin'].to_numpy(float)
    suf = {'dawn': '_Dawn', 'dusk': '_Dusk', 'all': ''}[side]
    tot = np.zeros_like(taa)
    for pr in ('PSD', 'TD', 'SWS', 'MMV'):
        c = f'Gen_{pr}{suf}'
        if c not in df.columns:
            c = f'Gen_{pr}'
        if c in df.columns:
            tot += df[c].to_numpy(float)
    bw = np.median(np.diff(np.sort(taa))) if len(taa) > 1 else 1.0
    return taa, tot / (dt_per_taa_deg(taa) * bw)


# ==========================================
# σ(LT, bin, TAA) と f(LT, bin, TAA) を作る
# ==========================================
def build_sigma_f(subdir, orbit, t0, ds, areas):
    """
    戻り値:
      taas     … (N_TAA,)
      sigma    … (N_TAA, N_LON_BIN, N_BIN)  在庫 [atoms]
      frac     … (N_TAA, N_LON_BIN, N_BIN)  1ステップで放出される割合 (在庫重み平均)
    どちらも太陽相対経度ビンに揃えてある。
    """
    path = os.path.join(BASE_DIR, subdir)
    files = list_surface_files(path, TARGET_YEAR)[::FILE_STRIDE]
    if not files:
        raise RuntimeError(f"surface_density がありません: {path}")

    u_model, u_mu = parse_u(subdir)
    q_coeff = parse_q(subdir)

    lon_c = np.linspace(-180, 180, N_LON + 1)[:-1] + 360.0 / N_LON / 2.0
    lat_c = np.linspace(-90, 90, N_LAT + 1)[:-1] + 180.0 / N_LAT / 2.0
    lat_m = (np.ones(N_LAT, dtype=bool) if LAT_RANGE is None
             else (lat_c >= LAT_RANGE[0]) & (lat_c <= LAT_RANGE[1]))

    taas, SIG, FRC = [], [], []
    for f in files:
        try:
            surf = np.load(f['path'], allow_pickle=False)
        except Exception:
            continue
        if surf.ndim != 3 or surf.shape[0] != N_LON:
            continue
        surf = np.clip(np.nan_to_num(surf, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        n_bin = surf.shape[2]

        sub_lon, AU = orbit_at(f['time_h'], orbit, t0)
        lon_sun = (lon_c - sub_lon + 180.0) % 360.0 - 180.0
        lon_d = lon_sun * (1.0 if ds > 0 else -1.0)     # 負 = dawn

        cos2d = (np.cos(np.deg2rad(lat_c))[:, None] *
                 np.cos(np.deg2rad(lon_sun))[None, :])
        cos_safe = np.maximum(cos2d, 0.0)
        illum = np.where(cos2d > 0.0, 1.0, 0.0)
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
        u_bins, q_bins = get_bins(n_bin, u_model, u_mu, q_coeff)

        # 半球・昼夜マスク
        day_m = np.ones_like(cos2d, dtype=bool) if INCLUDE_NIGHT else (cos2d > 0.0)
        if SIDE == 'dawn':
            side_col = lon_d < 0.0
        elif SIDE == 'dusk':
            side_col = lon_d > 0.0
        else:
            side_col = np.ones(N_LON, dtype=bool)
        cell_m = day_m & side_col[None, :] & lat_m[:, None]

        # 太陽相対経度ビンへの割り当て
        bin_idx = np.floor((lon_d + 180.0) / (360.0 / N_LON_BIN)).astype(int) % N_LON_BIN

        sig = np.zeros((N_LON_BIN, n_bin))
        num = np.zeros((N_LON_BIN, n_bin))
        for b in range(n_bin):
            u_j = u_bins[b] * EV_TO_J
            r_psd = psd_geom * q_bins[b]
            r_day = np.where(temp >= 10.0,
                             NU_0 * np.exp(np.maximum(-u_j / (K_B * temp), -700.0)), 0.0)
            r_night = NU_0 * np.exp(max(-u_j / (K_B * TEMP_NIGHT), -700.0))
            r_td = np.where(illum > 0.5, r_day, r_night)
            fr = 1.0 - np.exp(-np.minimum((r_psd + r_td + r_sws) * DT_SIM, 700.0))

            st = surf[:, :, b].T * areas * cell_m           # (LAT, LON) [atoms]
            np.add.at(sig, (bin_idx, np.full(N_LON, b)), st.sum(axis=0))
            # f は在庫で重み付けした平均にする (在庫ゼロなら面積重み)
            wsum = (st * fr).sum(axis=0)
            np.add.at(num, (bin_idx, np.full(N_LON, b)), wsum)

        with np.errstate(divide='ignore', invalid='ignore'):
            frc = np.where(sig > 0, num / sig, 0.0)

        taas.append(float(f['taa']))
        SIG.append(sig)
        FRC.append(frc)

    idx = np.argsort(taas)
    return (np.array(taas)[idx], np.array(SIG)[idx], np.array(FRC)[idx])


def frozen_curves(sigma, frac):
    """
    戻り値: P_both, P_rate, P_stock   (それぞれ (N_TAA,))
      P_rate  … σ を軌道平均で固定
      P_stock … f を軌道平均で固定
    """
    sig_bar = sigma.mean(axis=0)          # (LON, BIN)
    frc_bar = frac.mean(axis=0)
    both = np.einsum('tlb,tlb->t', sigma, frac)
    rate = np.einsum('lb,tlb->t', sig_bar, frac)
    stock = np.einsum('tlb,lb->t', sigma, frc_bar)
    return both, rate, stock


# ==========================================
# メイン
# ==========================================
def main():
    orbit, t0 = load_orbit(ORBIT_FILE_PATH)
    ds = dawn_sign(orbit)
    areas = areas_m2()
    print(f"dawn 判定: dawn = lon_sun {'<' if ds > 0 else '>'} 0  / 対象 {SIDE} / "
          f"{'昼夜' if INCLUDE_NIGHT else '昼面のみ'}")

    res = {}
    for q, sub in sorted(MODELS.items()):
        if not sub:
            continue
        print(f"\n[Q={q}] {sub}")
        try:
            taa, sigma, frac = build_sigma_f(sub, orbit, t0, ds, areas)
        except Exception as e:
            print(f"  エラー: {e}")
            continue
        both, rate, stock = frozen_curves(sigma, frac)
        taa_p, P = load_emission_rate(sub)

        g, Bg = resample(taa, both)
        _, Rg = resample(taa, rate)
        _, Sg = resample(taa, stock)
        Bg, Rg, Sg = circ_smooth(Bg), circ_smooth(Rg), circ_smooth(Sg)
        Pg = None
        if P is not None:
            _, Pg = resample(taa_p, P)
            Pg = circ_smooth(Pg)

        res[q] = dict(g=g, both=Bg, rate=Rg, stock=Sg, P=Pg)
        print(f"  両方 {peak_taa(g, Bg):.1f}° / 放出率だけ {peak_taa(g, Rg):.1f}° / "
              f"在庫だけ {peak_taa(g, Sg):.1f}°")

    if not res:
        print("有効なデータがありませんでした。")
        return

    qs = sorted(res.keys())
    rows = []
    print("\n" + "=" * 76)
    print(f"{'Q':>6} {'両方':>9} {'放出率だけ':>11} {'在庫だけ':>10} {'実測P':>9}")
    print("-" * 76)
    for q in qs:
        r = res[q]
        pb = peak_taa(r['g'], r['both'])
        pr = peak_taa(r['g'], r['rate'])
        ps = peak_taa(r['g'], r['stock'])
        pp = peak_taa(r['g'], r['P']) if r['P'] is not None else np.nan
        rows.append((q, pb, pr, ps, pp))
        print(f"{q:>6} {pb:>9.1f} {pr:>11.1f} {ps:>10.1f} {pp:>9.1f}")
    print("=" * 76)
    print("放出率だけ … 在庫を軌道平均で固定。放出率の TAA 依存だけでどこにピークが来るか")
    print("在庫だけ   … 放出率を軌道平均で固定。在庫の TAA 依存だけでどこにピークが来るか")
    print("両方が実測Pに近ければ、この2因子でピーク位置が決まっている")

    with open('frozen_factors.csv', 'w', encoding='utf-8') as f:
        f.write('Q,peak_both,peak_rate_only,peak_stock_only,peak_P\n')
        for r_ in rows:
            f.write(','.join('nan' if not np.isfinite(v) else f'{v:.2f}' for v in r_) + '\n')
    print("  -> frozen_factors.csv")

    # ---------- 図 ----------
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    a1, a2, a3, a4, a5, a6 = axes.ravel()
    cmap = plt.get_cmap('viridis')
    lo, hi = np.log10(min(qs)), np.log10(max(qs))
    col = {q: cmap((np.log10(q) - lo) / (hi - lo) if hi > lo else 0.5) for q in qs}

    def setax(ax, title):
        ax.axvline(180, color='gray', ls=':', alpha=0.6)
        ax.set_xlim(0, 360)
        ax.xaxis.set_major_locator(MultipleLocator(60))
        ax.set_xlabel('TAA [deg]')
        ax.set_ylabel('最大1に規格化')
        ax.set_title(title, fontsize=12)
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=8, ncol=2)

    def draw(ax, key, title):
        for q in qs:
            y = res[q][key]
            if y is None:
                continue
            mx = np.nanmax(y)
            ax.plot(res[q]['g'], y / mx if mx else y, '-', color=col[q], lw=2,
                    label=f'Q={q}')
            p = peak_taa(res[q]['g'], y)
            if np.isfinite(p):
                ax.axvline(p, color=col[q], ls=':', lw=1.1)
        setax(ax, title)

    draw(a1, 'rate', '放出率だけ動かす (在庫を固定)')
    draw(a2, 'stock', '在庫だけ動かす (放出率を固定)')
    draw(a3, 'both', '両方動かす  = W')
    draw(a4, 'P', '実測 P (Gen の記録値)')

    # Q ごとに3本を重ねて比べる (代表として最小Qと最大Q)
    for ax, q in ((a5, qs[0]), (a6, qs[-1])):
        r = res[q]
        for key, c, lb in (('rate', 'crimson', '放出率だけ'),
                           ('stock', 'royalblue', '在庫だけ'),
                           ('both', 'black', '両方')):
            y = r[key]
            mx = np.nanmax(y)
            ax.plot(r['g'], y / mx if mx else y, '-', color=c, lw=2.2, label=lb)
        if r['P'] is not None:
            ax.plot(r['g'], r['P'] / np.nanmax(r['P']), '--', color='seagreen',
                    lw=2.0, label='実測 P')
        setax(ax, f'Q = {q} の内訳')

    plt.tight_layout()
    if SAVE_PNG:
        fig.savefig(f'frozen_factors_{SIDE}.png', dpi=140, bbox_inches='tight')
        print(f"  -> frozen_factors_{SIDE}.png")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()