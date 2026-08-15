# -*- coding: utf-8 -*-
"""
渋滞域の「均され度」の定量化 と 渋滞先端到達TAA vs TDピークの照合

--- 背景 ---
TD単独のピークは左(≈162.5°)にある。これは「渋滞域(高密度在庫)が、
TDの効く高eff_cos領域にドカンと供給されるTAA」に対応する、という仮説。
PSDが強いと、この渋滞域が(ターミネーター際で消費されて)均され、
見かけの自転由来の遠日点(180°)ピークが相対的に強くなる。
PSDが弱いと渋滞域がくっきり残り、左右非対称な左ピークが強くなる。

--- このスクリプトで測るもの ---
[A] 均され度の3指標 (Stockのeff_cos方向プロファイルから、各TAAで):
    1. 歪度 skewness  : 在庫がeff_cos方向に偏っているか(渋滞=偏り大)
    2. ピーク/平均比  : max/mean (渋滞=尖って大, 均され=1に近い)
    3. 重心の分散     : 在庫の広がり(集中=渋滞 小, 広がり=均され 大)
    これらを Q2.0 と Q0.3 で TAA に対して比較。
    → PSDが強いQ2.0の方が「均され」ている(尖り小・分散大)はず。

[B] 渋滞先端の到達TAA vs TDピーク:
    各TAAで在庫が「急落し始めるeff_cos」(=渋滞先端の位置)を検出し、
    それがTDの効く閾値(eff_cos_TD)を超えるTAAを求める。
    そのTAAがTD放出ピーク(≈162.5°)と一致するかを照合。

Stock (band_statistics_per_taa.csv の時間平均在庫) を使用。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
from matplotlib.ticker import MultipleLocator
import os

# ==========================================
# 設定
# ==========================================
BASE_DIR = r"./SimulationResult_202607"

MODELS = {
    "Q2.0 (Standard)": "ParabolicHop_72x36_NoEq_DT100_0716_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr_test2",
    "Q0.3 (Weak PSD)": "ParabolicHop_72x36_NoEq_DT100_0716_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr_test2",
}
COLORS = {"Q2.0 (Standard)": "crimson", "Q0.3 (Weak PSD)": "steelblue"}

SIDE = "Dawn"
SMOOTH_WINDOW_DEG = 3

# [B] TDが効き始めるとみなす eff_cos 閾値(渋滞先端がここを超えたら大放出)
EFF_COS_TD_THRESHOLD = 0.5
# [B] 「急落」判定: バンド間の log10 減少がこの値を超えたら渋滞先端とみなす
DROP_LOG_THRESHOLD = 0.15


# ==========================================
# 読み込み
# ==========================================
def load_stock_and_gen(model_dir, side):
    path = os.path.join(model_dir, "band_statistics_per_taa.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"見つかりません: {path}")
    df = pd.read_csv(path)

    stock_col = f"Stock_{side}" if f"Stock_{side}" in df.columns else f"Sweep_{side}"
    if 'Step_Count' in df.columns:
        sc = df['Step_Count'].replace(0, np.nan)
        df['Stock'] = df[stock_col] / sc
    else:
        df['Stock'] = df[stock_col]
    df['Stock'] = df['Stock'].fillna(0.0)
    df['Gen_TD'] = df[f'Gen_TD_{side}']
    df['Gen_PSD'] = df[f'Gen_PSD_{side}']

    bands = df[['Band_Index', 'EffCos_Lo', 'EffCos_Hi']].drop_duplicates().sort_values('Band_Index')
    bc = ((bands['EffCos_Lo'] + bands['EffCos_Hi']) / 2.0).values

    def grid(col):
        return df.pivot(index='TAA_Bin', columns='Band_Index', values=col).sort_index().values

    return {
        'stock': grid('Stock'),
        'gen_td': grid('Gen_TD'),
        'gen_psd': grid('Gen_PSD'),
        'band_centers': bc,
        'taa': np.sort(df['TAA_Bin'].unique()),
        'n_bands': len(bc),
    }


def circular_smooth(y, w):
    if w <= 0:
        return y
    w = int(w)
    k = np.ones(w) / w
    ext = np.concatenate([y[-w:], y, y[:w]])
    return np.convolve(ext, k, mode='same')[w:-w]


# ==========================================
# [A] 均され度の3指標
# ==========================================
def smoothing_metrics(stock_row, bc):
    """在庫のeff_cosプロファイルから3つの均され度指標を計算。
    在庫を「eff_cos上の重み分布」とみなす(昼側のみ、在庫>0)。"""
    w = np.clip(stock_row, 0, None)
    if w.sum() <= 0:
        return np.nan, np.nan, np.nan

    # 重み付き平均・分散・歪度(eff_cosを変数として)
    mean = np.sum(w * bc) / np.sum(w)
    var = np.sum(w * (bc - mean)**2) / np.sum(w)
    std = np.sqrt(var) if var > 0 else 0.0
    if std > 0:
        skew = np.sum(w * (bc - mean)**3) / np.sum(w) / (std**3)
    else:
        skew = 0.0

    # ピーク/平均比
    mean_stock = w.mean()
    peak_over_mean = w.max() / mean_stock if mean_stock > 0 else np.nan

    return skew, peak_over_mean, var


def analyze_smoothing(models, side, smooth_deg=3):
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    metric_names = ['歪度 skewness\n(偏り; 渋滞=大)',
                    'ピーク/平均比\n(尖り; 渋滞=大)',
                    '重心の分散\n(広がり; 均され=大)']

    for label, subdir in models.items():
        try:
            d = load_stock_and_gen(os.path.join(BASE_DIR, subdir), side)
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            continue
        bc = d['band_centers']
        taa = d['taa']
        c = COLORS.get(label)

        skew_arr = np.full(len(taa), np.nan)
        pom_arr = np.full(len(taa), np.nan)
        var_arr = np.full(len(taa), np.nan)
        for ti in range(len(taa)):
            skew_arr[ti], pom_arr[ti], var_arr[ti] = smoothing_metrics(d['stock'][ti, :], bc)

        for ax, arr in zip(axes, [skew_arr, pom_arr, var_arr]):
            ax.plot(taa, circular_smooth(arr, smooth_deg), '-', color=c, lw=2, label=label)

    for ax, name in zip(axes, metric_names):
        ax.axvline(180, color='gray', ls=':', alpha=0.6, label='遠日点(180)')
        ax.axvline(162, color='green', ls='--', alpha=0.5, label='TD単独ピーク(162)')
        ax.set_ylabel(name)
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('TAA [deg]')
    axes[-1].set_xlim(0, 360)
    axes[-1].xaxis.set_major_locator(MultipleLocator(60))
    fig.suptitle(f'渋滞域の均され度 3指標 — Stock ({side}側)\n'
                 'PSD強(Q2.0)ほど均される(尖り小・分散大)はず', fontsize=13)
    plt.tight_layout()
    plt.show()


# ==========================================
# [B] 渋滞先端の到達TAA vs TDピーク
# ==========================================
def analyze_frontier(models, side, smooth_deg=3):
    fig, ax = plt.subplots(figsize=(11, 6))

    print("\n" + "=" * 70)
    print(f"=== 渋滞先端の到達eff_cos と TDピーク照合 ({side}側) ===")
    print("=" * 70)

    for label, subdir in models.items():
        try:
            d = load_stock_and_gen(os.path.join(BASE_DIR, subdir), side)
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            continue
        bc = d['band_centers']
        taa = d['taa']
        c = COLORS.get(label)

        # 各TAAで在庫が急落し始めるeff_cos(=渋滞先端)を検出
        frontier = np.full(len(taa), np.nan)
        for ti in range(len(taa)):
            s = np.clip(d['stock'][ti, :], 1e-300, None)
            logs = np.log10(s)
            # 低eff_cosから見て、初めて大きく落ちるバンド境界のeff_cos
            for b in range(1, len(bc)):
                if logs[b-1] - logs[b] > DROP_LOG_THRESHOLD:
                    frontier[ti] = bc[b]  # 落ちた先のeff_cos(渋滞先端はこの手前まで)
                    break
            else:
                frontier[ti] = bc[-1]  # 急落なし=SSPまで在庫あり

        frontier_sm = circular_smooth(frontier, smooth_deg)
        ax.plot(taa, frontier_sm, '-', color=c, lw=2, label=f'{label}: 渋滞先端eff_cos')

        # TDピークTAA
        gen_td_total = d['gen_td'].sum(axis=1)
        td_peak_taa = taa[np.nanargmax(circular_smooth(gen_td_total, smooth_deg))]

        # 渋滞先端がTD閾値を超える最初のTAA(近日点側から)
        cross_taa = None
        for ti in range(len(taa)):
            if frontier_sm[ti] >= EFF_COS_TD_THRESHOLD:
                cross_taa = taa[ti]
                break

        ax.axvline(td_peak_taa, color=c, ls=':', alpha=0.7)
        print(f"\n[{label}]")
        print(f"  TD放出ピーク TAA            : {td_peak_taa:.0f}°")
        print(f"  渋滞先端がeff_cos={EFF_COS_TD_THRESHOLD}を超えるTAA: "
              f"{cross_taa if cross_taa is not None else '該当なし':}")
        if cross_taa is not None:
            print(f"  → 両者の差: {abs(td_peak_taa - cross_taa):.0f}° "
                  f"({'近い(仮説支持)' if abs(td_peak_taa-cross_taa)<25 else '離れている'})")

    ax.axhline(EFF_COS_TD_THRESHOLD, color='green', ls='--', alpha=0.6,
               label=f'TD効き始め閾値 eff_cos={EFF_COS_TD_THRESHOLD}')
    ax.axvline(162, color='black', ls='-.', alpha=0.4, label='TD単独ピーク(162)')
    ax.set_xlabel('TAA [deg]')
    ax.set_ylabel('渋滞先端の eff_cos (在庫が急落し始める位置)')
    ax.set_title(f'渋滞先端の到達eff_cos vs TAA ({side}側)\n'
                 '点線=各モデルのTD放出ピークTAA')
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    print("=" * 70)


if __name__ == "__main__":
    # [A] 均され度 3指標
    analyze_smoothing(MODELS, SIDE, smooth_deg=SMOOTH_WINDOW_DEG)

    # [B] 渋滞先端 vs TDピーク
    analyze_frontier(MODELS, SIDE, smooth_deg=SMOOTH_WINDOW_DEG)