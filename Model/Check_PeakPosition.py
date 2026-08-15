# -*- coding: utf-8 -*-
"""
柱密度ピーク TAA の Q 依存
- density_grid_t*****_taa***.npy を読み、TAA プロファイルを作る
- ピーク TAA を放物線補間つきで自動検出 (目視読み取りを排除)
- ピークの平坦さ (95%プラトー幅・曲率) と、モンテカルロによる位置の不確かさを出す
- 横軸 Q (対数) × 縦軸 ピーク TAA の図を出力

ピーク位置は定数倍に依らないので cell_volume や正規化面積は掛けていない。
"""

import os
import re
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
from matplotlib.ticker import MultipleLocator

# ==========================================
# 設定
# ==========================================
BASE_DIR = r"./SimulationResult_202607"

# Q値 -> ディレクトリ名。実際のフォルダ名に合わせて書き換える
MODELS = {
    0.1:  "New_ParabolicHop_72x36_NoEq_DT100_0804_BD0.8_UG1.85_Q0.1_A2.0e+07_LT190k_15yr",
    0.27: "Def2_ParabolicHop_72x36_NoEq_DT100_0729_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr",
    1.0:  "ParabolicHop_72x36_NoEq_DT100_0710_BD0.8_UG1.85_Q1.0_A2.0e+07_LT190k_15yr",
    2.0:  "Def2_ParabolicHop_72x36_NoEq_DT100_0729_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr",
    3.0:  "ParabolicHop_72x36_NoEq_DT100_0710_BD0.8_UG1.85_Q3.0_A2.0e+07_LT190k_15yr",
    5.0:  "New_ParabolicHop_72x36_NoEq_DT100_0804_BD0.8_UG1.85_Q5.0_A2.0e+07_LT190k_15yr",
    10.0: "New_ParabolicHop_72x36_NoEq_DT100_0804_BD0.8_UG1.85_Q10.0_A2.0e+07_LT190k_15yr",
}

# 参考線として引く TD のみ run のピーク TAA。不要なら None
TD_ONLY_REFERENCE = None      # 例: 165.0

TARGET_YEAR = 15
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0
MERCURY_YEAR_HOURS = 87.969 * 24

# 解析する領域。'DAWN' / 'DUSK' / 'DAYSIDE_TOTAL'
ANALYSIS_MODES = ['DAWN', 'DUSK']

# TAA を等間隔に張り直す刻み [deg]
TAA_STEP = 1.0
# 平滑化の窓 [点]。奇数。1 で平滑化なし
SMOOTH_WIN = 1
# 不確かさ推定のモンテカルロ試行回数
N_MC = 500
# プラトー幅を測るしきい値 (最大値に対する比)
PLATEAU_LEVEL = 0.95

# ピークを探す TAA 窓 [deg]。None で全域。
# Dusk のように近日点側にも山がある場合、全域だとそちらを拾ってしまうので
# 遠日点側の山に限定する。Dawn / Dusk で同じ窓を使うこと。
PEAK_WINDOW = (90.0, 270.0)

# 誤差棒を「年ごとのばらつき」で出す場合、対象年のリストを入れる。
# 例: [13, 14, 15]。None ならモンテカルロ推定を使う。
ERROR_FROM_YEARS = None

# ピーク推定の方法
#   'parabola3'    … 最大ビンとその左右1点の3点で放物線補間 (仮定なし・ノイズに弱い)
#   'parabola_fit' … ピーク周辺 ±FIT_HALFWIDTH_DEG の点で2次を最小二乗フィット
#   'harmonic'     … 全周期を N_HARMONICS 次までの三角関数でフィットし、窓内の最大を取る
PEAK_METHOD = 'parabola_fit'
FIT_HALFWIDTH_DEG = 50.0
N_HARMONICS = 4

# 3方法すべての結果をコンソールに並べて表示するか (手法依存性の確認用)
COMPARE_METHODS = True

# 図に誤差棒を描くか
SHOW_ERRORBAR = False

# 図を画面に表示するか / PNG に保存するか
SHOW_PLOT = True
SAVE_PNG = False

RNG = np.random.default_rng(0)


# ==========================================
# データ読み込み
# ==========================================
def load_column_density(target_dir, mode, target_year=TARGET_YEAR,
                        grid_res=GRID_RESOLUTION):
    """density_grid から TAA と柱密度(任意単位)の配列を返す。"""
    mid_x = (grid_res - 1) // 2
    mid_y = (grid_res - 1) // 2
    q_off = mid_y // 2

    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"ディレクトリがありません: {target_dir}")

    files = []
    for f in os.listdir(target_dir):
        m = re.search(r'^density_grid_t(\d+)_taa(\d+)\.npy$', f)
        if not m:
            continue
        time_h, taa = int(m.group(1)), int(m.group(2))
        if int(time_h // MERCURY_YEAR_HOURS) + 1 != target_year:
            continue
        files.append((f, taa))

    if not files:
        raise RuntimeError(f"Year {target_year} のデータがありません: {target_dir}")

    taas, vals = [], []
    for fname, taa in files:
        g = np.load(os.path.join(target_dir, fname))
        day = g[mid_x:, :, :].astype(float)
        day[0, :, :] *= 0.5                      # x=0 面は半分だけ数える

        mid_sum = day[:, mid_y, :].sum()
        if mode == 'DAWN':
            v = day[:, :mid_y, :].sum() + 0.5 * mid_sum
        elif mode == 'DUSK':
            v = day[:, mid_y + 1:, :].sum() + 0.5 * mid_sum
        elif mode == 'DAYSIDE_TOTAL':
            v = day.sum()
        elif mode == 'DAWN_OUTER':
            v = day[:, :q_off, :].sum()
        elif mode == 'DUSK_OUTER':
            v = day[:, (grid_res - 1) - q_off:, :].sum()
        else:
            raise ValueError(f"不明なモード: {mode}")

        taas.append(float(taa))
        vals.append(float(v))

    taas = np.array(taas)
    vals = np.array(vals)
    idx = np.argsort(taas)
    return taas[idx], vals[idx]


def resample_uniform(taa, val, step=TAA_STEP):
    """TAA を等間隔に張り直す。周期 360 で補間。"""
    grid = np.arange(0.0, 360.0, step)
    return grid, np.interp(grid, taa, val, period=360.0)


def circ_smooth(y, win=SMOOTH_WIN):
    if win <= 1:
        return y.copy()
    k = np.ones(win) / win
    pad = win // 2
    ext = np.concatenate([y[-pad:], y, y[:pad]])
    return np.convolve(ext, k, mode='valid')


# ==========================================
# ピーク検出
# ==========================================
def window_mask(taa, window=None):
    """探索窓の真偽配列。窓が 0/360 をまたぐ場合も扱う。"""
    if window is None:
        return np.ones(len(taa), dtype=bool)
    lo, hi = window
    if lo <= hi:
        return (taa >= lo) & (taa <= hi)
    return (taa >= lo) | (taa <= hi)


def find_peak(taa, y, window=PEAK_WINDOW):
    """
    窓の中の最大値位置を、放物線補間つきで返す。
    補間には窓の外の隣接点も使う (周期境界)。
    戻り値: (ピークTAA, 放物線の2次係数 a)
    """
    n = len(y)
    step = taa[1] - taa[0]
    m = window_mask(taa, window)
    idx = np.where(m)[0]
    if len(idx) == 0:
        idx = np.arange(n)
    k = int(idx[np.argmax(y[idx])])
    y0, y1, y2 = y[(k - 1) % n], y[k], y[(k + 1) % n]
    den = y0 - 2 * y1 + y2
    d = 0.0
    a = 0.0
    if den != 0:
        d = float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))
        a = den / (2.0 * step ** 2)              # y = a(x-x0)^2 + c の a
    return float((taa[k] + d * step) % 360.0), a


def _angdiff(a, b):
    """a - b を -180..180 に畳む。"""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def find_peak_parabola_fit(taa, y, window=None, half=None):
    """粗いピーク周辺の点をまとめて2次関数に最小二乗フィットし、頂点を返す。"""
    if half is None:
        half = FIT_HALFWIDTH_DEG
    p0, _ = find_peak(taa, y, window)
    d = _angdiff(taa, p0)
    m = np.abs(d) <= half
    if m.sum() < 4:
        return p0, 0.0
    c = np.polyfit(d[m], y[m], 2)
    if c[0] >= 0:
        return p0, 0.0
    return float((p0 - c[1] / (2 * c[0])) % 360.0), float(c[0])


def find_peak_harmonic(taa, y, window=None, n_harm=None):
    """全周期を三角関数でフィットし、窓の中の最大を細かい格子で探す。"""
    if n_harm is None:
        n_harm = N_HARMONICS
    th = np.deg2rad(taa)
    cols = [np.ones_like(th)]
    for k in range(1, n_harm + 1):
        cols += [np.cos(k * th), np.sin(k * th)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)

    fine = np.arange(0.0, 360.0, 0.1)
    tf = np.deg2rad(fine)
    cf = [np.ones_like(tf)]
    for k in range(1, n_harm + 1):
        cf += [np.cos(k * tf), np.sin(k * tf)]
    yf = np.column_stack(cf) @ coef

    mm = window_mask(fine, window)
    idx = np.where(mm)[0]
    if len(idx) == 0:
        idx = np.arange(len(fine))
    k = int(idx[np.argmax(yf[idx])])
    return float(fine[k]), yf


def peak_by(taa, y, method=None, window=PEAK_WINDOW):
    """PEAK_METHOD に従ってピーク TAA を返す。"""
    method = method or PEAK_METHOD
    if method == 'parabola3':
        p, a = find_peak(taa, y, window)
    elif method == 'parabola_fit':
        p, a = find_peak_parabola_fit(taa, y, window)
    elif method == 'harmonic':
        p, _ = find_peak_harmonic(taa, y, window)
        a = 0.0
    else:
        raise ValueError(f"不明な PEAK_METHOD: {method}")
    return p, a


def plateau_width(taa, y, level=PLATEAU_LEVEL, window=PEAK_WINDOW):
    """窓の中で、最大値の level 倍以上になっている TAA の総幅 [deg]。"""
    step = taa[1] - taa[0]
    m = window_mask(taa, window)
    yw = y[m]
    thr = yw.min() + (yw.max() - yw.min()) * level
    return float((yw >= thr).sum() * step)


def peak_uncertainty(taa, y_raw, y_smooth, n_mc=N_MC):
    """
    高周波残差からノイズ振幅を見積もり、それを載せ直して
    ピーク位置がどれだけ揺れるかを測る。
    """
    sigma = float(np.std(y_raw - y_smooth))
    if sigma <= 0:
        return 0.0, 0.0
    peaks = []
    for _ in range(n_mc):
        yy = circ_smooth(y_smooth + RNG.normal(0.0, sigma, size=len(y_smooth)))
        p, _ = peak_by(taa, yy)
        peaks.append(p)
    peaks = np.array(peaks)
    # 周期量なので円周統計で散らばりを出す
    ang = np.deg2rad(peaks)
    R = np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
    circ_std = np.rad2deg(np.sqrt(-2.0 * np.log(max(R, 1e-12))))
    return float(circ_std), sigma


# ==========================================
# メイン
# ==========================================
def peak_of_year(path, mode, year):
    taa_raw, val_raw = load_column_density(path, mode, target_year=year)
    taa, y = resample_uniform(taa_raw, val_raw)
    p, _ = peak_by(taa, circ_smooth(y))
    return p


def year_spread(path, mode, years):
    """複数年それぞれのピークを求め、円周標準偏差を返す。"""
    ps = []
    for yr in years:
        try:
            ps.append(peak_of_year(path, mode, yr))
        except Exception:
            pass
    if len(ps) < 2:
        return np.nan
    ang = np.deg2rad(np.array(ps))
    R = np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(max(R, 1e-12)))))


def analyze(mode):
    rows = []
    profiles = {}

    for q, sub in sorted(MODELS.items()):
        if not sub:
            print(f"[skip] Q={q}: ディレクトリ未設定")
            continue
        path = os.path.join(BASE_DIR, sub)
        try:
            taa_raw, val_raw = load_column_density(path, mode)
        except Exception as e:
            print(f"[エラー] Q={q}: {e}")
            continue

        taa, y = resample_uniform(taa_raw, val_raw)
        ys = circ_smooth(y)

        p_raw, _ = peak_by(taa, y)
        p_sm, a = peak_by(taa, ys)
        width = plateau_width(taa, ys)
        sd, sigma = peak_uncertainty(taa, y, ys)
        if ERROR_FROM_YEARS:
            sd_y = year_spread(path, mode, ERROR_FROM_YEARS)
            if np.isfinite(sd_y):
                sd = sd_y
        noise_pct = sigma / (ys.max() - ys.min()) * 100 if ys.max() > ys.min() else np.nan

        rows.append(dict(Q=q, peak_raw=p_raw, peak_smooth=p_sm, sd=sd,
                         width=width, curv=a, noise_pct=noise_pct, n=len(taa_raw)))
        profiles[q] = (taa, ys)

        msg = (f"Q={q:<5} ピーク {p_sm:6.1f}° (生 {p_raw:6.1f}°)  "
               f"±{sd:4.1f}°  95%幅 {width:5.1f}°  ノイズ {noise_pct:4.1f}%  n={len(taa_raw)}")
        if COMPARE_METHODS:
            alt = "  |  " + " ".join(
                f"{m}={peak_by(taa, ys, m)[0]:.1f}°"
                for m in ('parabola3', 'parabola_fit', 'harmonic'))
            msg += alt
        print(msg)

    return rows, profiles


def plot_result(rows, profiles, mode):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax, ax2 = axes

    qs = np.array([r['Q'] for r in rows])
    pk = np.array([r['peak_smooth'] for r in rows])
    sd = np.array([r['sd'] for r in rows])

    if SHOW_ERRORBAR:
        lab = ('年ごとのばらつき' if ERROR_FROM_YEARS else 'MCノイズ由来')
        ax.errorbar(qs, pk, yerr=sd, fmt='o-', color='crimson', lw=2.0,
                    ms=7, capsize=4, label=f'柱密度ピーク (誤差: {lab})')
    else:
        ax.plot(qs, pk, 'o-', color='crimson', lw=2.0, ms=7, label='柱密度ピーク')
    ax.axhline(180, color='gray', ls=':', lw=1.2, label='遠日点')
    if TD_ONLY_REFERENCE is not None:
        ax.axhline(TD_ONLY_REFERENCE, color='royalblue', ls='--', lw=1.5,
                   label=f'TDのみ ({TD_ONLY_REFERENCE:.0f}°)')
    ax.set_xscale('log')
    ax.set_xlabel('PSD 光脱離断面積 Q  [×10e-20 cm²]')
    ax.set_ylabel('柱密度ピークの TAA [deg]')
    ax.set_title(f'ピーク TAA の Q 依存 ({mode}, {PEAK_METHOD})', fontsize=13)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=9)

    cmap = plt.get_cmap('viridis')
    lo, hi = np.log10(qs.min()), np.log10(qs.max())
    for r in rows:
        q = r['Q']
        taa, ys = profiles[q]
        yn = (ys - ys.min()) / (ys.max() - ys.min())
        c = cmap((np.log10(q) - lo) / (hi - lo) if hi > lo else 0.5)
        ax2.plot(taa, yn, '-', color=c, lw=2.0, label=f'Q={q}')
        ax2.axvline(r['peak_smooth'], color=c, ls=':', lw=1.2)

    ax2.axvline(180, color='gray', ls=':', lw=1.2)
    if PEAK_WINDOW is not None:
        lo, hi = PEAK_WINDOW
        if lo <= hi:
            ax2.axvspan(0, lo, color='gray', alpha=0.12)
            ax2.axvspan(hi, 360, color='gray', alpha=0.12)
        ax2.text(0.02, 0.02, f'灰色 = 探索窓の外 ({lo:.0f}–{hi:.0f}°)',
                 transform=ax2.transAxes, fontsize=9, color='dimgray')
    ax2.set_xlabel('TAA [deg]')
    ax2.set_ylabel('規格化された柱密度')
    ax2.set_title('TAA プロファイル (最小0・最大1に規格化)', fontsize=13)
    ax2.set_xlim(0, 360)
    ax2.xaxis.set_major_locator(MultipleLocator(60))
    ax2.grid(True, ls='--', alpha=0.4)
    ax2.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    if SAVE_PNG:
        out = f'peak_taa_vs_Q_{mode}.png'
        fig.savefig(out, dpi=140, bbox_inches='tight')
        print(f"  -> {out}")
    if not SHOW_PLOT:
        plt.close(fig)
    return fig


def main():
    for mode in ANALYSIS_MODES:
        print(f"\n===== {mode} =====")
        rows, profiles = analyze(mode)
        if not rows:
            print("有効なデータがありませんでした。")
            continue
        plot_result(rows, profiles, mode)

        with open(f'peak_taa_vs_Q_{mode}.csv', 'w', encoding='utf-8') as f:
            f.write('Q,peak_smooth_deg,peak_raw_deg,sd_deg,plateau95_deg,curvature,noise_pct,n_files\n')
            for r in rows:
                f.write(f"{r['Q']},{r['peak_smooth']:.2f},{r['peak_raw']:.2f},"
                        f"{r['sd']:.2f},{r['width']:.1f},{r['curv']:.4e},"
                        f"{r['noise_pct']:.2f},{r['n']}\n")
        print(f"  -> peak_taa_vs_Q_{mode}.csv")

    if SHOW_PLOT:
        plt.show()


if __name__ == '__main__':
    main()