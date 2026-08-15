# -*- coding: utf-8 -*-
"""
band_statistics_per_taa.csv (eff_cos多段バンド × TAA) の解析・可視化スクリプト。

検証したい仮説:
「掃き込み量 × その場の生成効率(局所時刻=eff_cos依存)の積のピークが、
 パラメータ(Q)で経度方向に動く」

出力する図:
  [A] TAA × eff_cosバンド のヒートマップ (Sweep / Gen_PSD / Gen_TD)  ※モデルごと
  [B] eff_cosバンド別に見た、生成量とsweepのTAA分布 (どのバンドがどのTAAでピークか)
  [C] TAAごとの「昼全体のsweep合計」と「昼全体の生成量」— Q比較
  [D] 放出の"局所時刻の重心"(eff_cos加重平均) が TAA に対してどう動くか — Q比較
       (低Qでこの重心がTDの効く側へ手前で頭打ちになる、という予測の直接確認)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401  (日本語ラベル用)
from matplotlib.ticker import MultipleLocator
import os

# ==========================================
# 設定
# ==========================================
BASE_DIR = r"./SimulationResult_202607"

MODELS = {
    "Q2.0 (Standard)": "ParabolicHop_72x36_NoEq_DT100_0714_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr_test",
    "Q0.3 (Weak PSD)": "ParabolicHop_72x36_NoEq_DT100_0714_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr_test",
}

SIDE = "Dawn"          # "Dawn" または "Dusk"
SMOOTH_WINDOW_DEG = 3  # TAA方向の移動平均窓[deg]。0でなし。


# ==========================================
# 読み込み・整形
# ==========================================
def load_band_stats(model_dir, side):
    path = os.path.join(model_dir, "band_statistics_per_taa.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"見つかりません: {path}")
    df = pd.read_csv(path)

    gen_psd_col = f"Gen_PSD_{side}"
    gen_td_col = f"Gen_TD_{side}"
    sweep_col = f"Sweep_{side}"
    df['Gen_PSD'] = df[gen_psd_col]
    df['Gen_TD'] = df[gen_td_col]
    df['Gen_Total'] = df['Gen_PSD'] + df['Gen_TD']
    df['Sweep'] = df[sweep_col]

    bands = df[['Band_Index', 'EffCos_Lo', 'EffCos_Hi']].drop_duplicates().sort_values('Band_Index')
    band_centers = ((bands['EffCos_Lo'] + bands['EffCos_Hi']) / 2.0).values
    band_labels = [f"{lo:.2f}–{hi:.2f}" for lo, hi in zip(bands['EffCos_Lo'], bands['EffCos_Hi'])]
    n_bands = len(bands)
    n_taa = df['TAA_Bin'].nunique()

    # (TAA, band) の2次元配列に整形
    def to_grid(col):
        g = df.pivot(index='TAA_Bin', columns='Band_Index', values=col).sort_index()
        return g.values  # shape (360, n_bands)

    grids = {
        'Gen_PSD': to_grid('Gen_PSD'),
        'Gen_TD': to_grid('Gen_TD'),
        'Gen_Total': to_grid('Gen_Total'),
        'Sweep': to_grid('Sweep'),
    }
    taa_axis = np.sort(df['TAA_Bin'].unique())
    return {
        'grids': grids, 'taa': taa_axis, 'band_centers': band_centers,
        'band_labels': band_labels, 'n_bands': n_bands
    }


def circular_smooth_2d(arr, window_deg):
    """TAA(axis=0, 周期360)方向にのみ移動平均"""
    if window_deg <= 0:
        return arr
    w = int(window_deg)
    if w < 1:
        return arr
    kernel = np.ones(w) / w
    out = np.empty_like(arr)
    n = arr.shape[0]
    for k in range(arr.shape[1]):
        col = arr[:, k]
        ext = np.concatenate([col[-w:], col, col[:w]])
        sm = np.convolve(ext, kernel, mode='same')
        out[:, k] = sm[w:-w]
    return out


# ==========================================
# [A] ヒートマップ
# ==========================================
def plot_heatmaps(data_dict):
    for label, d in data_dict.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, key, title in zip(
            axes,
            ['Sweep', 'Gen_PSD', 'Gen_TD'],
            ['掃き込み供給量 Sweep', 'PSD生成量', 'TD生成量']
        ):
            grid = circular_smooth_2d(d['grids'][key], SMOOTH_WINDOW_DEG).T  # (band, TAA)
            # log表示(0対策)
            with np.errstate(divide='ignore'):
                loggrid = np.log10(np.where(grid > 0, grid, np.nan))
            im = ax.imshow(loggrid, aspect='auto', origin='lower',
                           extent=[0, 360, 0, d['n_bands']], cmap='viridis')
            ax.set_yticks(np.arange(d['n_bands']) + 0.5)
            ax.set_yticklabels(d['band_labels'], fontsize=8)
            ax.set_xlabel('TAA [deg]')
            ax.set_ylabel('eff_cos バンド (ターミネーター←→正午)')
            ax.set_title(f'{title}')
            ax.axvline(180, color='white', ls=':', alpha=0.6)
            ax.xaxis.set_major_locator(MultipleLocator(60))
            plt.colorbar(im, ax=ax, label='log10(atoms)')
        fig.suptitle(f'{label} — {SIDE}側 (TAA × eff_cosバンド)', fontsize=13)
        plt.tight_layout()
        plt.show()


# ==========================================
# [C] 昼全体でのsweep合計 と 生成量 — Q比較
# ==========================================
def plot_dayside_totals(data_dict):
    colors = {}
    palette = ['crimson', 'steelblue', 'darkgreen', 'purple']
    for i, label in enumerate(data_dict.keys()):
        colors[label] = palette[i % len(palette)]

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    for label, d in data_dict.items():
        taa = d['taa']
        c = colors[label]
        sweep_tot = circular_smooth_2d(d['grids']['Sweep'], SMOOTH_WINDOW_DEG).sum(axis=1)
        genpsd_tot = circular_smooth_2d(d['grids']['Gen_PSD'], SMOOTH_WINDOW_DEG).sum(axis=1)
        gentd_tot = circular_smooth_2d(d['grids']['Gen_TD'], SMOOTH_WINDOW_DEG).sum(axis=1)

        axes[0].plot(taa, sweep_tot, color=c, lw=2, label=label)
        axes[1].plot(taa, genpsd_tot, color=c, lw=2, label=label)
        axes[2].plot(taa, gentd_tot, color=c, lw=2, label=label)

    titles = ['昼全体の掃き込み供給量 Sweep', '昼全体のPSD生成量', '昼全体のTD生成量']
    for ax, t in zip(axes, titles):
        ax.set_ylabel('atoms')
        ax.set_title(f'{t} ({SIDE}側)')
        ax.axvline(180, color='gray', ls=':', alpha=0.6, label='遠日点')
        ax.axvline(100, color='orange', ls='--', alpha=0.5, label='TAA=100(掃き込み最大予想)')
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel('TAA [deg]')
    axes[-1].set_xlim(0, 360)
    axes[-1].xaxis.set_major_locator(MultipleLocator(60))
    plt.tight_layout()
    plt.show()


# ==========================================
# [D] 放出の"局所時刻の重心"(eff_cos加重平均)のTAA依存 — Q比較
# ==========================================
def plot_emission_centroid(data_dict):
    colors = {}
    palette = ['crimson', 'steelblue', 'darkgreen', 'purple']
    for i, label in enumerate(data_dict.keys()):
        colors[label] = palette[i % len(palette)]

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for label, d in data_dict.items():
        taa = d['taa']
        c = colors[label]
        bc = d['band_centers']  # (n_bands,)

        gen_total = circular_smooth_2d(d['grids']['Gen_Total'], SMOOTH_WINDOW_DEG)  # (360, n_bands)
        sweep = circular_smooth_2d(d['grids']['Sweep'], SMOOTH_WINDOW_DEG)

        # 各TAAで、生成量をeff_cosバンド中心で加重平均 = 「放出が起きた局所時刻の重心」
        gen_row_sum = gen_total.sum(axis=1)
        gen_centroid = np.where(gen_row_sum > 0,
                                (gen_total * bc[None, :]).sum(axis=1) / np.where(gen_row_sum > 0, gen_row_sum, 1),
                                np.nan)

        sweep_row_sum = sweep.sum(axis=1)
        sweep_centroid = np.where(sweep_row_sum > 0,
                                  (sweep * bc[None, :]).sum(axis=1) / np.where(sweep_row_sum > 0, sweep_row_sum, 1),
                                  np.nan)

        axes[0].plot(taa, gen_centroid, color=c, lw=2, label=f'{label}: 生成の重心')
        axes[1].plot(taa, sweep_centroid, color=c, lw=2, ls='--', label=f'{label}: 掃き込みの重心')

    axes[0].set_ylabel('放出の局所時刻重心\n(eff_cos加重平均)')
    axes[0].set_title(f'放出が起きた局所時刻の重心 ({SIDE}側)\n'
                      f'高い=正午寄り(TD側)、低い=ターミネーター寄り(PSD側)')
    axes[1].set_ylabel('掃き込みの局所時刻重心\n(eff_cos加重平均)')
    axes[1].set_title('掃き込み供給が起きた局所時刻の重心')

    for ax in axes:
        ax.axvline(180, color='gray', ls=':', alpha=0.6)
        ax.axvline(100, color='orange', ls='--', alpha=0.5)
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel('TAA [deg]')
    axes[-1].set_xlim(0, 360)
    axes[-1].xaxis.set_major_locator(MultipleLocator(60))
    plt.tight_layout()
    plt.show()


# ==========================================
# 数値サマリー
# ==========================================
def print_summary(data_dict):
    print("\n" + "=" * 78)
    print(f"=== サマリー ({SIDE}側) ===")
    print("=" * 78)
    for label, d in data_dict.items():
        taa = d['taa']
        sweep_tot = d['grids']['Sweep'].sum(axis=1)
        genpsd_tot = d['grids']['Gen_PSD'].sum(axis=1)
        gentd_tot = d['grids']['Gen_TD'].sum(axis=1)
        gen_tot = genpsd_tot + gentd_tot

        def peak(y):
            ys = circular_smooth_2d(y[:, None], SMOOTH_WINDOW_DEG)[:, 0]
            return taa[np.nanargmax(ys)]

        print(f"\n[{label}]")
        print(f"  掃き込みピーク TAA     : {peak(sweep_tot):.0f} deg")
        print(f"  PSD生成ピーク TAA      : {peak(genpsd_tot):.0f} deg")
        print(f"  TD生成ピーク TAA       : {peak(gentd_tot):.0f} deg")
        print(f"  総生成ピーク TAA       : {peak(gen_tot):.0f} deg")
        print(f"  PSD/TD 総量比          : {genpsd_tot.sum()/gentd_tot.sum():.3f}"
              if gentd_tot.sum() > 0 else "  (TD=0)")
    print("=" * 78)


if __name__ == "__main__":
    data = {}
    for label, subdir in MODELS.items():
        full = os.path.join(BASE_DIR, subdir)
        try:
            data[label] = load_band_stats(full, SIDE)
            print(f"[読込] {label} ({data[label]['n_bands']}バンド)")
        except Exception as e:
            print(f"[エラー] {label}: {e}")

    if len(data) >= 1:
        print_summary(data)
        plot_heatmaps(data)
        plot_dayside_totals(data)
        plot_emission_centroid(data)